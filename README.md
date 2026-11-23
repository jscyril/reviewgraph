# ReviewGraph: a multi-agent PR reviewer

A LangGraph app that reviews GitHub pull requests and posts its findings as inline review comments. A supervisor agent decides which specialists the PR needs. Every changed file then gets a security, performance and/or style reviewer, all running in parallel, and each reviewer can read other files in the repo for context before deciding. The findings are deduplicated and ranked, a human can approve them, and they're posted as one GitHub review.

It runs as a **GitHub Action** on every PR, or as a **CLI**. It works with **Gemini, Groq, OpenRouter or Anthropic**.

```mermaid
graph LR
  START --> fetch[fetch PR<br/>diff + head SHA] --> supervisor
  supervisor -- "Send × (file, specialist)" --> review
  subgraph review [reviewer agent, one per file × specialist]
    direction TB
    think[LLM] -- read_file --> gh[(GitHub API<br/>at head SHA)] --> think
    think --> out[structured findings]
  end
  supervisor -. nothing to review .-> report
  review --> report[dedupe + rank]
  report -- --post --> approve{{human approval<br/>interrupt}}
  report -- --post --yes --> post
  approve -- approved --> post[post GitHub review<br/>inline comments]
  approve -- rejected --> END
  report --> END
  post --> END
```

## What it shows

| Concept | Where |
|---|---|
| Supervisor / specialist multi-agent design | `supervisor` node picks the specialists; `review` runs them |
| Dynamic parallel fan-out with `Send`, merged by an `operator.add` reducer | `route` → one `review` per (file, specialist) |
| Tool-using agents (ReAct loop) | reviewers call `read_file` through the GitHub API at the PR's head commit |
| Structured output into Pydantic, per-provider strategy | native JSON schema (Gemini, Anthropic) or tool calling (Groq, OpenRouter) |
| Human-in-the-loop with `interrupt` + checkpointer | `approve` node pauses; resume with `Command(resume=...)`, optionally dropping findings |
| Evals | planted-bug benchmark: recall and false positives per provider (`evals/`) |
| Observability | LangSmith tracing with run names, tags and metadata |

## GitHub Action

Add this to any repo as `.github/workflows/pr-review.yml`, and add your provider's API key as a repository secret:

```yaml
name: PR Review
on:
  pull_request:
    types: [opened, reopened, ready_for_review]
permissions:
  contents: read
  pull-requests: write
jobs:
  review:
    if: ${{ !github.event.pull_request.draft }}
    runs-on: ubuntu-latest
    steps:
      - uses: jscyril/reviewgraph@main
        with:
          provider: gemini            # or groq / openrouter / anthropic
          # model: gemini-2.5-pro     # optional override
        env:
          GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}   # GROQ_API_KEY / OPENROUTER_API_KEY / ANTHROPIC_API_KEY
```

The review is posted as PR comments, and the full report also appears in the workflow run summary. In CI, the approval step is skipped (`--yes`).

## CLI

```bash
uv sync
export GEMINI_API_KEY=...   # or GROQ_API_KEY / OPENROUTER_API_KEY / ANTHROPIC_API_KEY
export GITHUB_TOKEN=...     # needed to post; optional for reading public PRs (raises rate limits)

uv run pr-reviewer https://github.com/OWNER/REPO/pull/123                 # print the report
uv run pr-reviewer https://github.com/OWNER/REPO/pull/123 --post          # review, approve or drop findings, then post
uv run pr-reviewer changes.diff --provider groq                           # local diff; reviewers read files from --repo-dir (default .)
```

| `--provider` | API key env var | Default model | Structured output |
|---|---|---|---|
| `gemini` | `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) | `gemini-pro-latest` | native JSON schema |
| `groq` | `GROQ_API_KEY` | `openai/gpt-oss-120b` | tool calling |
| `openrouter` | `OPENROUTER_API_KEY` | `openai/gpt-oss-120b` | tool calling |
| `anthropic` | `ANTHROPIC_API_KEY` | `claude-opus-5-5` | native JSON schema |

## Evals

There are 9 files with one planted bug each, covering SQL injection, path traversal, a hardcoded secret, pickle deserialization, XSS, an N+1 query, a blocking sleep in async code, a quadratic loop and a swallowed exception. There are also 2 clean files where any finding is a false positive.

```bash
uv run python -m evals --provider gemini --provider groq --provider openrouter
```

The run prints a comparison table and saves per-case results to `evals/results/<provider>.json`. A bug counts as caught when a finding lands in the same file within 3 lines of it.

## Tracing (LangSmith)

```bash
export LANGSMITH_TRACING=true LANGSMITH_API_KEY=...
```

Every run appears as `pr-review`, tagged with the provider. Eval runs appear as `eval:<case>`. Each reviewer's tool calls show up as child spans.

## Test

```bash
uv run pytest
```

The tests run offline with a stub LLM and a fake GitHub. They cover the diff parser, fan-out per file and specialist, the tool loop, path-traversal protection, removing duplicates, the approval interrupt (approve, drop and reject), splitting inline comments from body findings, and the eval scorer.

## Limits

- Each push doesn't trigger a new review, because the workflow runs only on open, reopen and ready-for-review. Re-run it from the Actions tab or add `synchronize` to the trigger list. If you add it, you'll get a new review on every push.
- Fork PRs get no secrets, so the action can't run on them.
- PR text goes into the prompts, so a malicious PR could try to steer the reviewers. The reviewers can only read repo files and write comments. They never see your API keys.
- Cost grows with files × specialists. Lockfiles are skipped. Use `max-concurrency` to stay under provider rate limits.
