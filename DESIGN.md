# Moonleaf — Design

**A wind tunnel for LLM inference infrastructure.**

Moonleaf is a single offline binary with two co-equal subsystems:

1. **A measurement client** — a load generator for OpenAI-compatible streaming endpoints with instrument-grade rigor: coordinated-omission-correct open-loop load, honest token/chunk semantics, and a built-in calibration mode that characterizes the tool's own measurement error.
2. **An inference simulator** — an OpenAI-compatible server whose latency is *emergent*, not injected: a real-time discrete-event model of continuous batching, prefill/decode asymmetry, and KV-cache memory pressure, parameterized by GPU profiles.

Together they form a wind tunnel: generate controlled traffic, fly it through a modeled backend, and measure with calibrated instruments — no GPU, no network, no external services. Everything reproduces from a seed with `cargo run`.

---

## Positioning

Existing tools each cover part of this space:

| Tool                       | Language  | LLM-aware metrics | Realistic offline backend | Measurement calibration |
| -------------------------- | --------- | ----------------- | ------------------------- | ----------------------- |
| guidellm (vllm-project)    | Python    | yes               | no                        | no                      |
| genai-perf (NVIDIA)        | Python    | yes               | no                        | no                      |
| inference-benchmarker (HF) | Rust      | yes               | no                        | no                      |
| vllm bench serve           | Python    | yes               | no (needs vLLM)           | no                      |
| oha / wrk / k6             | Rust/C/Go | no                | no                        | no                      |
| **moonleaf**               | Rust      | yes               | **yes**                   | **yes**                 |

The differentiators are the last two columns. Every existing mock is a latency injector; moonleaf's simulator reproduces the *dynamics* of inference serving (TTFT knees, batch-dependent inter-token latency, preemption stalls) from first principles. And no existing tool ships a way to prove its own measurements are honest.

The simulator is also independently useful: a GPU-free, drop-in OpenAI-compatible backend with realistic behavior, for testing gateways, routers, and autoscalers.

---

## Interfaces

### CLI

| Command             | Purpose                                                                                                                                                  |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `moonleaf run`      | Benchmark an endpoint: `--target <URL>` for any OpenAI-compatible server, or `--sim <profile>` to auto-start the simulator in-process on a loopback port |
| `moonleaf sim`      | Run the simulator standalone (for gateway/router testing, or cross-network benchmarking)                                                                 |
| `moonleaf compare`  | Same workload against two targets (or two sim profiles); side-by-side table with deltas                                                                  |
| `moonleaf validate` | Calibration: benchmark the simulator's injector engine with known distributions, report measured-vs-injected error per percentile                        |

### HTTP (simulator)

- `POST /v1/chat/completions` — SSE streaming, `stream_options: {include_usage}`, terminating `[DONE]`
- `GET /v1/models`, `GET /healthz`
- Introspection (optional extension, never required by the client):
  - `GET /sim/stats` — queue depth, batch size, KV utilization time series
  - `x-sim-*` response headers — per-request queue wait and prefill time, so client-observed TTFT can be decomposed into queue + prefill

### Non-goals

- Non-OpenAI API dialects; non-streaming benchmarking
- Real tokenization — the simulator uses a deterministic ~4-chars/token rule and reports usage truthfully against it; the client prefers server-reported usage
- TUI dashboards; plotting (JSONL output feeds any external plotting tool)
- Virtual-clock "pure simulation" (capacity planning without HTTP) — interesting, but the measurement instrument is half the product and instruments need real I/O
- KV swap modeling (preemption uses recompute only)

---

## Measurement principles

These are the load-bearing rules; violating any of them silently invalidates results.

1. **The client observes from the outside.** All primary metrics come from the client's own socket observations — timestamps of SSE chunks as they arrive. Simulator introspection is diagnostic ground truth, never a metric source. The client must work identically against backends that offer no cooperation.
2. **Open-loop load is coordinated-omission-correct.** Requests fire on a pre-materialized schedule (constant, Poisson, ramp, burst) regardless of in-flight completions, and latency is measured from the *scheduled* send time. Closed-loop mode (fixed concurrency) exists as a deliberate lab clamp — it probes the server's operating curve (e.g., ITL vs batch size) — and its latency-hiding behavior at saturation is documented, not papered over.
3. **Chunks are not tokens.** Servers may pack multiple tokens per SSE chunk. What the client measures directly is inter-*chunk* latency and it is reported as such. Token counts come from server-reported `usage` when available, estimation otherwise (flagged in output).
4. **Metric definitions match vLLM's** for cross-tool comparability: TTFT (request send → first content chunk), TPOT ((E2E − TTFT) / (output_tokens − 1)), ITL (per-chunk gaps), E2E, plus request/token throughput and first-class error rates (timeout, HTTP error, malformed SSE).
5. **The instrument is calibrated.** `validate` runs the client against the injector engine with known injected distributions and reports the measurement error per percentile. The tool's error bounds are documented, not assumed.
6. **Everything reproduces from a seed.** One global seed derives the full request plan and all per-request content; two runs with the same seed produce identical workloads.

---

## Client pipeline

1. **Workload plan.** A seeded plan of per-request specs: arrival time (open-loop), input/output token targets, per-request seed. Specs are tens of bytes each — prompts are never materialized in the plan. Prompt text (meaningless filler with controlled token count) is synthesized deterministically from the per-request seed at send time. Sources: inline prompt, prompt file, or the synthetic generator with token-count distributions (fixed/uniform/lognormal), using `max_tokens` and `ignore_eos` where supported.
2. **Arrival scheduling.** Closed-loop: N workers in send→await→send loops (arrival times cannot be pre-planned; contents still come from the plan). Open-loop: requests spawn at scheduled times, unbounded in-flight.
3. **Request task.** reqwest streaming response → incremental SSE parser (frames split at arbitrary byte boundaries, multi-line `data:`, keep-alive comments, CRLF, `[DONE]`) → monotonic timestamp per chunk → usage extraction.
4. **Aggregation.** Each completed request becomes one `RequestRecord` (all timestamps, token counts, error class) sent over an mpsc channel to a single aggregator task owning the HDR histograms and counters. No shared mutable state, no lock contention; warmup exclusion and live progress fall out naturally.
5. **Output.** Terminal summary, JSON summary, JSONL raw records (every chunk timestamp — anything is recomputable offline), live progress line.

## Simulator

Two engines behind the same HTTP surface:

**Injector engine.** Sampled TTFT / inter-chunk latency / output-length distributions. Exists as calibration ground truth for `validate` and as the "dumb mock" baseline that the dynamics engine is contrasted against.

**Dynamics engine.** A real-time discrete-event simulation:

- **GPU profile** (built-in presets such as `a100-7b-fp16`, `a100-7b-int4`, `l4-8b-int4`, or custom TOML): weights bytes, HBM bandwidth, prefill throughput, VRAM, KV bytes/token, KV block size, max batch tokens. Derived: KV block budget = (VRAM − weights) / block bytes.
- **State:** FIFO waiting queue, running batch, per-sequence KV block allocations, sampled target output length per request.
- **Scheduler loop** (single task; one iteration per simulated GPU step):
  1. Admit waiting requests while KV blocks and the batch-token budget allow (continuous batching).
  2. Take a bounded chunk of pending prefill work (chunked prefill interleaving).
  3. Compute the step duration from a roofline-style cost model: memory time (weights read once per step regardless of batch size — the reason batching works — plus KV reads scaling with batch) vs compute time (prefill tokens this step); step = max of the two.
  4. Sleep for the step duration, then emit one token to each decoding sequence's SSE stream; retire finished sequences and free their blocks.
  5. On block exhaustion, preempt the most recently admitted sequence back to the queue with recompute cost.

Nothing is tuned to "look right": TTFT knees under rising load, ITL growth with batch size, preemption stalls under long-output pressure, and quantization gains (¼ weight bytes, 4× KV budget) all emerge from the loop.

**Fidelity claim: queueing-accurate, not FLOP-accurate.** The simulator models scheduling, batching, and memory-pressure *dynamics*, parameterized by published hardware numbers. It predicts the shapes of curves and their causes, not absolute production latencies.

**Deployment shapes.** Same code, two shapes: in-process background task on a loopback port (`run --sim` — zero setup, used by all case studies; traffic still crosses real localhost HTTP/TCP, never in-process calls), or a standalone process via `moonleaf sim` (isolates CPU, allows real network in the path).

---

## Architecture

- **Cargo workspace:** `crates/core` (protocol types, SSE parser, workload plan, metrics), `crates/sim` (both engines, axum server), `crates/cli` (the `moonleaf` binary).
- **Concurrency:** tokio; per-request tasks; one aggregator task owning all mutable metric state, fed by mpsc. The simulator's scheduler loop is a single task driven by timers.
- **Key crates:** tokio, axum, reqwest, hdrhistogram, serde/serde_json, clap, tracing, rand (seedable).
- **Testing:** SSE parser fixture tests with pathological byte splits; integration tests run the client against the in-process simulator; determinism tests (same seed → identical plan and sim behavior).

---

## Case studies

Six written studies, each reproducible offline with a single command, each explaining its curves via the modeled internals:

1. **Coordinated omission, demonstrated** — closed-loop vs open-loop against the same saturated simulator; the hidden latency made visible.
2. **The TTFT knee** — open-loop rate sweep; queue growth past the service capacity.
3. **ITL vs batch size** — closed-loop concurrency steps; memory-bandwidth sharing during decode.
4. **KV pressure and preemption** — long-output workload exhausting the block budget; preemption stalls in the tail.
5. **FP16 vs INT4** — `compare` across two profiles of the same GPU; throughput and TTFT deltas from smaller weights and larger KV budget.
6. **Chunked prefill on/off** — mixed long/short-prompt workload; short-request TTFT rescued from prefill head-of-line blocking.

## Engineering bar

CI (fmt, clippy, tests) green on every commit; release binaries for Linux and macOS; README with the positioning table, metric definitions, architecture diagram, and a demo GIF.
