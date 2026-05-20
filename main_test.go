package main

import "testing"

func TestParseNumber(t *testing.T) {
	tests := []struct {
		name string
		text string
		want int
	}{
		{name: "bare number", text: "7", want: 7},
		{name: "ten", text: "10", want: 10},
		{name: "uses final number when prose repeats range", text: "Between 1 and 10, I pick 4.", want: 4},
		{name: "ignores embedded digits", text: "run 2 selected 9", want: 9},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parseNumber(tt.text)
			if err != nil {
				t.Fatalf("parseNumber() unexpected error: %v", err)
			}
			if got != tt.want {
				t.Fatalf("parseNumber() = %d, want %d", got, tt.want)
			}
		})
	}
}

func TestParseNumberRejectsMissingNumber(t *testing.T) {
	if _, err := parseNumber("no numeric answer"); err == nil {
		t.Fatal("parseNumber() expected an error")
	}
}

func TestValidateConfig(t *testing.T) {
	cfg := runConfig{
		model:       "gpt-5-mini",
		calls:       200,
		parallelism: 20,
		timeout:     1,
	}
	if err := validateConfig(cfg); err != nil {
		t.Fatalf("validateConfig() unexpected error: %v", err)
	}
}
