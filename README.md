# Holistic Agent Observability (Local & Frugal)

<div align="center">
  <img src="https://github.com/user-attachments/assets/04398c09-a162-4da4-91c9-9a8e62b663f3" alt="Graph flow and reasoning timeline" width="49%" />
  <img src="https://github.com/user-attachments/assets/113805dd-7e68-4143-8577-f3285437650f" alt="Metrics overview and sustainability impact" width="49%" />
  <br />
  <sub><i>Left: Graph + reasoning timeline. Right: Metrics + sustainability (CodeCarbon).</i></sub>
  <br />
</div>


End-to-end, local-first tracing, QA, and sustainability insights for LLM agents using only Ollama and open models.

- Complete graph view of traces (React Flow) from user question to tool calls and generations
- Human-interpretable reasoning timeline and insights
- NLI-based groundedness (entailment/neutral/contradicted) with local cross-encoders
- CodeCarbon-based sustainability panel with per-phase breakdown (Reasoning vs Rest)
- Single command to run everything locally

## Tech Stack
- Backend: FastAPI + Uvicorn, Pydantic models, CodeCarbon (offline/process mode)
- Agent: Python + LangChain ChatOllama (local LLMs), modular tools
- NLI: Local HuggingFace cross-encoders (e.g., `cross-encoder/nli-deberta-v3-small`)
- Frontend: React + TypeScript + Vite, React Flow for graphs
- Observability: Langfuse traces, local caching of processed summaries
- Env/Build: `uv` for Python, `npm` for frontend

## Bring Your Own Agent (BYOA)
You can analyze traces from any agent framework. All we need is tracing data:

- Langfuse (recommended): point the UI at your Langfuse project and paste API keys (or set env vars). We fetch traces and process them entirely locally.
- LangSmith: similar concept; add support or export traces to Langfuse-compatible format.

Once connected, load your traces in the UI, click “Process trace,” and the toolkit computes groundedness, insights, and sustainability (CodeCarbon) locally—no paid API calls.

## Requirements
- Python 3.11+
- Node 18+ (for the UI dev server)
- Ollama installed locally (and desired models pulled)
- Optional: Langfuse API access and local `.env` for credentials if you want to load remote traces

## Setup
- Copy `.env.example` to `.env` at the repo root and fill placeholders (no secrets committed).
- Copy `observability/.env.example` to `observability/.env` for the frontend.
- Pull desired Ollama models locally (for example: `ollama pull qwen3:4b`).

## Quick Start (single command)
1. Install package (build and install locally):
   ```bash
   uv build
   uv pip install -U ./dist/glass_agent-0.1.0-py3-none-any.whl
   ```
2. Start both backend and frontend together:
   ```bash
   glass-agent up --reload
   ```
   - Backend: http://127.0.0.1:8000
   - Frontend: http://localhost:5173

## How to run and test
- Start everything locally:
   ```bash
   glass-agent up --reload
   ```
   Then open the UI at http://localhost:5173
- CLI quick check:
   ```bash
   glass-agent ask "What is my emergency fund status?"
   ```
   Confirms the agent runs and writes a CodeCarbon sidecar under `data/carbon/`.
- UI workflow: connect Langfuse, load a trace, click “Process trace,” then verify the graph, timeline, and Sustainability panel render.
- Backend-only check:
   ```bash
   glass-agent serve --reload
   ```
   Visit http://127.0.0.1:8000/docs to confirm the API is up.

## Importing Traces
1) Configure provider in the UI (Langfuse for now).
2) Provide API key either:
   - Via environment variables (preferred), or
   - Paste into the UI settings (stored locally in your browser).
3) Click “Connect” to load recent traces, pick one, and click “Process trace.” Cached results can be reloaded without reprocessing.

## Alternative CLI commands
- Run only the backend:
  ```bash
  glass-agent serve --host 127.0.0.1 --port 8000 --reload
  ```
- Ask the agent once from CLI:
  ```bash
  glass-agent ask "Give me a progress report on my emergency fund and travel goals."
  ```
- Interactive demo (REPL):
  ```bash
  glass-agent demo
  ```

## Configuration
- Environment variables (see `.env` template if present):
  - `VITE_TRACE_SUMMARY_URL` (frontend) — defaults to `http://localhost:8000/trace-summary`
  - Langfuse keys as needed for trace retrieval (optional). Example:
    - `VITE_LANGFUSE_PUBLIC_KEY` and `VITE_LANGFUSE_SECRET_KEY`
    - `VITE_LANGFUSE_HOST` (if self-hosted)
- Models:
  - LLMs via Ollama: `qwen3:4b`, `llama3.1:8b`, `gpt-oss:20b` (configurable in UI)
  - NLI cross-encoders (HuggingFace) for groundedness scoring

## Sustainability Metrics (CodeCarbon)
- We measure carbon emissions using `OfflineEmissionsTracker` in process/offline mode to avoid system prompts.
- Each agent run persists a sidecar JSON to `data/carbon` keyed by user-question hash.
- Backend merges sidecar with trace durations to estimate a per-phase split:
  - Reasoning (LLM thinking steps)
  - Rest (planning, tools, other processing)
- UI shows total mg CO₂e and 3 significant-figure values per phase.
- Cached per-trace metrics prevent recomputation on replays.

## Reproducibility & Local-First
- No paid APIs required; everything runs locally with Ollama + local cross-encoders.
- Deterministic sidecar files for per-run carbon metrics in `data/carbon`.
- LocalStorage caches processed summaries so re-renders don’t call the backend again.

## Why run analysis locally?
- Privacy: keep sensitive traces and prompts on your machine.
- Frugality: no pay-per-token costs; use Ollama-hosted models.
- Repeatability: re-run, cache, and compare changes without recomputation.
- Debuggability: single dev machine reproduces the exact UI and metrics.

## Example use cases
- Red-teaming and leakage checks on regulated data (e.g., banking, healthcare) without sending to third-party APIs.
- Prompt/template iteration: spot contradiction hotspots and regressions fast with cached metrics.
- Tooling audits: verify tool misuse, loops/timeouts, and repeated error patterns in long traces.
- Sustainability reviews: compare reasoning styles/models vs. carbon cost before rollouts.
- Offline demos and reproducible experiments for stakeholders.

## Development
- Backend live-reload is enabled with `--reload` in the CLI command.
- Frontend hot-reloads via Vite.
- Code formatting and linting are project/tooling dependent (not enforced by this README).

## Folder Structure (high level)
```
agent/                      # Agent, tools, CLI
observability_backend/      # FastAPI server computing summaries & metrics
observability/              # React UI
prompts/                    # Sample banking prompts
data/carbon/                # CodeCarbon sidecar JSONs (created at runtime)
```

## Troubleshooting
- Ollama model errors: ensure models are pulled locally (e.g., `ollama pull qwen3:4b`).
- Port conflicts: change ports with `glass-agent up --port 8001 --ui-port 5174`.
- No carbon on first run: ensure you run an agent call (or sidecar exists) so the backend can join per-trace emissions.
- Langfuse auth: set env vars (host and API keys) and verify connectivity.

## FAQs
- How do I swap LLMs? Use the UI “Settings” panel to pick a different Ollama model (and NLI model). Defaults live in code (`agent/config.py`).
- Can I use models that aren’t in Ollama? Yes, but you’ll need to adapt the agent to your runtime and ensure traces still land in Langfuse.
- Can I run headless? Yes. Start the backend with `glass-agent serve` and POST to `/trace-summary` directly.
- Why CodeCarbon offline/process mode? It avoids elevated privileges on macOS and still gives consistent, reproducible estimates per run.
- How is the carbon split computed? We apportion the total per-trace emissions by the durations of reasoning vs other observations.
- Do I need the internet? Only to fetch dependencies or remote traces; all analysis and model inference can run locally.

## Environment variables quick reference
- Frontend
  - `VITE_TRACE_SUMMARY_URL`: Backend endpoint for summarization (default `http://localhost:8000/trace-summary`).
  - `VITE_LANGFUSE_HOST`: Langfuse base URL (e.g., `http://localhost:3000` or your cloud host).
  - `VITE_LANGFUSE_PUBLIC_KEY`: Langfuse public key for Basic auth.
  - `VITE_LANGFUSE_SECRET_KEY`: Langfuse secret key for Basic auth.
- Backend/Agent (optional)
  - `AGENT_MODEL`: Default Ollama model id (e.g., `qwen3:4b`).
  - `AGENT_NAME`: Label for the agent when persisting sidecar metadata.
  - Any provider-specific vars you introduce.

