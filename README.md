# llmrandom

Calls `gpt-5-mini` 200 times with:

```text
Pick a randmom number between 1-10
```

Then prints a terminal histogram of the numbers returned.

## Run

```sh
export OPENAI_API_KEY="..."
go run . -calls 200 -parallel 20
```

Useful flags:

- `-model`: model name, default `gpt-5-mini`
- `-calls`: number of samples, default `200`
- `-parallel`: max concurrent API calls, default `20`
- `-timeout`: total run timeout, default `5m`
