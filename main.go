package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/responses"
	"github.com/openai/openai-go/shared"
)

const prompt = "Pick a randmom number between 1-10"

var numberPattern = regexp.MustCompile(`\b(?:10|[1-9])\b`)

type runConfig struct {
	model       string
	calls       int
	parallelism int
	timeout     time.Duration
}

type sampleResult struct {
	index  int
	number int
	text   string
	err    error
}

func main() {
	cfg := runConfig{}
	flag.StringVar(&cfg.model, "model", "gpt-5-mini", "OpenAI model to call")
	flag.IntVar(&cfg.calls, "calls", 200, "number of model calls to make")
	flag.IntVar(&cfg.parallelism, "parallel", 20, "maximum concurrent model calls")
	flag.DurationVar(&cfg.timeout, "timeout", 5*time.Minute, "overall timeout")
	flag.Parse()

	if err := validateConfig(cfg); err != nil {
		fmt.Fprintf(os.Stderr, "config error: %v\n", err)
		os.Exit(2)
	}
	if os.Getenv("OPENAI_API_KEY") == "" {
		fmt.Fprintln(os.Stderr, "OPENAI_API_KEY is not set")
		os.Exit(2)
	}

	ctx, cancel := context.WithTimeout(context.Background(), cfg.timeout)
	defer cancel()

	client := openai.NewClient()
	results := collectSamples(ctx, client, cfg)
	if err := printSummary(results, cfg.calls); err != nil {
		fmt.Fprintf(os.Stderr, "run failed: %v\n", err)
		os.Exit(1)
	}
}

func validateConfig(cfg runConfig) error {
	if cfg.calls <= 0 {
		return errors.New("calls must be greater than 0")
	}
	if cfg.parallelism <= 0 {
		return errors.New("parallel must be greater than 0")
	}
	if cfg.timeout <= 0 {
		return errors.New("timeout must be greater than 0")
	}
	if strings.TrimSpace(cfg.model) == "" {
		return errors.New("model must not be empty")
	}
	return nil
}

func collectSamples(ctx context.Context, client openai.Client, cfg runConfig) []sampleResult {
	jobs := make(chan int)
	results := make(chan sampleResult, cfg.calls)

	workers := min(cfg.parallelism, cfg.calls)
	var wg sync.WaitGroup
	wg.Add(workers)
	for worker := 0; worker < workers; worker++ {
		go func() {
			defer wg.Done()
			for index := range jobs {
				number, text, err := callModel(ctx, client, cfg.model)
				results <- sampleResult{
					index:  index,
					number: number,
					text:   text,
					err:    err,
				}
			}
		}()
	}

	go func() {
		defer close(jobs)
		for index := 1; index <= cfg.calls; index++ {
			select {
			case <-ctx.Done():
				return
			case jobs <- index:
			}
		}
	}()

	go func() {
		wg.Wait()
		close(results)
	}()

	collected := make([]sampleResult, 0, cfg.calls)
	for result := range results {
		collected = append(collected, result)
	}
	sort.Slice(collected, func(i, j int) bool {
		return collected[i].index < collected[j].index
	})
	return collected
}

func callModel(ctx context.Context, client openai.Client, model string) (int, string, error) {
	resp, err := client.Responses.New(ctx, responses.ResponseNewParams{
		Model: shared.ResponsesModel(model),
		Input: responses.ResponseNewParamsInputUnion{
			OfString: openai.String(prompt),
		},
		Instructions:    openai.String("Return only one integer from 1 through 10."),
		MaxOutputTokens: openai.Int(16),
		Store:           openai.Bool(false),
	})
	if err != nil {
		return 0, "", err
	}

	text := strings.TrimSpace(resp.OutputText())
	number, err := parseNumber(text)
	if err != nil {
		return 0, text, err
	}
	return number, text, nil
}

func parseNumber(text string) (int, error) {
	matches := numberPattern.FindAllString(text, -1)
	if len(matches) == 0 {
		return 0, fmt.Errorf("could not find a number from 1 through 10 in %q", text)
	}

	// If the model includes explanatory text that repeats the range, the selected
	// number is usually the final number in the response.
	number, err := strconv.Atoi(matches[len(matches)-1])
	if err != nil {
		return 0, err
	}
	if number < 1 || number > 10 {
		return 0, fmt.Errorf("parsed number out of range: %d", number)
	}
	return number, nil
}

func printSummary(results []sampleResult, expected int) error {
	counts := make(map[int]int, 10)
	var failures []sampleResult
	for _, result := range results {
		if result.err != nil {
			failures = append(failures, result)
			continue
		}
		counts[result.number]++
	}

	if len(results) != expected {
		failures = append(failures, sampleResult{
			err: fmt.Errorf("only collected %d of %d results", len(results), expected),
		})
	}

	successes := len(results) - countFailures(results)
	fmt.Printf("prompt: %q\n", prompt)
	fmt.Printf("successful samples: %d/%d\n\n", successes, expected)
	printHistogram(counts, successes)

	if len(failures) == 0 {
		return nil
	}

	fmt.Fprintln(os.Stderr, "\nfailures:")
	for _, failure := range failures {
		label := "run"
		if failure.index > 0 {
			label = fmt.Sprintf("run %d", failure.index)
		}
		fmt.Fprintf(os.Stderr, "  %s: %v", label, failure.err)
		if failure.text != "" {
			fmt.Fprintf(os.Stderr, " (output: %q)", failure.text)
		}
		fmt.Fprintln(os.Stderr)
	}
	return fmt.Errorf("%d failed sample(s)", len(failures))
}

func countFailures(results []sampleResult) int {
	failures := 0
	for _, result := range results {
		if result.err != nil {
			failures++
		}
	}
	return failures
}

func printHistogram(counts map[int]int, total int) {
	maxCount := 0
	for number := 1; number <= 10; number++ {
		maxCount = max(maxCount, counts[number])
	}

	const maxBarWidth = 50
	for number := 1; number <= 10; number++ {
		count := counts[number]
		barWidth := 0
		if maxCount > 0 {
			barWidth = count * maxBarWidth / maxCount
		}

		percent := 0.0
		if total > 0 {
			percent = float64(count) * 100 / float64(total)
		}

		fmt.Printf("%2d | %-50s %3d %5.1f%%\n", number, strings.Repeat("#", barWidth), count, percent)
	}
}
