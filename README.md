# AMB — Agent Memory Benchmark

We built AMB because we wanted to be honest about how Hindsight performs — and because no existing benchmark gave us the full picture. AMB is fully open: datasets, prompts, scoring logic, and results.

Live leaderboard: **[agentmemorybenchmark.ai](https://agentmemorybenchmark.ai)**

## The problem with existing benchmarks

LoComo and LongMemEval are solid datasets, but they were designed for an era of 32k context windows. State-of-the-art models now have million-token context windows — on most instances, a naive "dump everything into context" approach scores competitively, not because it's a good memory architecture, but because retrieval has become the easy part. The benchmarks can no longer tell them apart.

Both datasets were also built around chatbot use cases. Agents today don't just answer questions about conversation history — they research, plan, execute multi-step tasks, and build knowledge across many interactions. AMB adds datasets that focus on agentic tasks: memory across tool calls, knowledge built from document research, preferences applied to multi-step decisions.

## What AMB measures

A memory system that scores 90% accuracy but costs $10 per user per day is not better than one that scores 82% and costs $0.10. AMB starts from accuracy because it's the hardest to fake, and tracks speed and token cost alongside it.

The only credible benchmark result is one you can reproduce yourself. AMB publishes everything: the evaluation harness, judge prompts, answer generation prompts, and the exact models used. Small changes to any of these can swing accuracy scores by double digits — we publish all of them.

## How it works

1. **Ingest** — documents from a dataset are loaded into a memory provider
2. **Retrieve** — for each query the memory provider retrieves relevant context
3. **Generate** — a Gemini model produces an answer from the retrieved context
4. **Judge** — a second Gemini call scores the answer against gold answers

Retrieval time is tracked separately from generation; ingestion time is also recorded.

One dataset breaks this shape on purpose. **PrecisionMemBench** ([upstream](https://github.com/tenurehq/precisionmembench)) asserts, per case, exactly which memories must and must not come back — noise is a hard failure, not an invisible cost — so it runs with `--mode retrieval`: no answer LLM, no judge, and the score is the fraction of cases whose returned belief-ID set satisfies every assertion.

```bash
uv run amb run --dataset precisionmembench --split single-turn --memory hindsight-cloud --mode retrieval
```

## Setup

```bash
# Copy and fill in your API key
cp .env.example .env   # or just create .env with:
# GEMINI_API_KEY=...
```

### Which LLM scores the runs

**Every benchmark round is read and judged by the same model, so the rows are
comparable.** For the hippocampus rounds that model is
`MiniMax-M3.1-Flash` on the MiniMax token plan (a subscription, so a
500-question round costs no marginal money and cannot run out mid-run the way
a pay-per-token balance does).

```bash
export OPENAI_API_KEY=$MINIMAX_API_KEY
export OPENAI_BASE_URL=https://api.minimaxi.com/v1
export OMB_ANSWER_LLM=openai  OMB_ANSWER_MODEL=MiniMax-M3.1-Flash
export OMB_JUDGE_LLM=openai   OMB_JUDGE_MODEL=MiniMax-M3.1-Flash
# MiniMax accepts response_format=json_schema and then ignores it, answering
# in prose; without this the harness fails at json.loads() instead of falling
# back to json_object.
export OMB_JSON_MODE=json_object
# A request timeout is not optional: without one a single hung connection
# blocks the whole run.
export OMB_LLM_TIMEOUT=180
```

`scripts/run_mml_round.sh <provider>` runs a round with that configuration.
Rows scored by a different model are a separate round and must not be mixed:
the deepseek-flash rows (hippocampus 78.0% on LME-S 500) predate this rule
and are kept only as a historical comparison.

## Usage

```bash
# List available datasets, memory providers, and modes
uv run amb providers

# List domains for a dataset
uv run amb domains --dataset personamem

# Run a benchmark
uv run amb run --dataset personamem --domain 32k --memory bm25

# Limit scale for a quick test
uv run amb run --dataset personamem --domain 32k --memory bm25 --query-limit 20

# Oracle mode: ingest only gold documents (tests generation quality in isolation)
uv run amb run --dataset personamem --domain 32k --memory bm25 --oracle

# Dataset statistics
uv run amb dataset-stats --dataset personamem

# Browse results in the browser
uv run amb view
```

## Results

Results are saved to `outputs/{dataset}/{memory}/{mode}/{domain}.json` and can be explored with `uv run amb view`.

## Requirements

- Python ≥ 3.11
- `GEMINI_API_KEY` in `.env` or environment
- For MemBench: set `MEMBENCH_DATA_PATH` to your local data directory
