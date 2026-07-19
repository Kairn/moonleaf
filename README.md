# Moonleaf :first_quarter_moon_with_face:

**A wind tunnel for LLM inference infrastructure.**

Moonleaf is a single offline binary that combines:

- **A benchmark client** for OpenAI-compatible streaming endpoints — LLM-aware metrics (TTFT, inter-token latency, TPOT) measured with instrument-grade rigor: coordinated-omission-correct open-loop load, honest token/chunk semantics, and a built-in calibration mode that characterizes the tool's own measurement error.
- **An inference simulator** — an OpenAI-compatible server whose latency *emerges* from a real-time model of continuous batching, prefill/decode asymmetry, and KV-cache memory pressure, parameterized by GPU profiles. No GPU required. Also useful standalone as a realistic backend for testing gateways, routers, and autoscalers.

Generate controlled traffic, fly it through a modeled backend, measure with calibrated instruments — entirely offline, reproducible from a seed.
