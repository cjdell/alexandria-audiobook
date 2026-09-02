# AGENTS

Guidance for AI agents working in this repository. Read this before making changes.

## What this is

Alexandria — a WebUI that turns books into audiobooks. FastAPI backend in
`app/app.py` (serves the UI on port 4200); pipeline workers run as subprocesses:
`generate_script.py` → `review_script.py` → `generate_personas.py` → TTS
(`app/tts.py`) → export. Script generation chunks the book and asks an
OpenAI-compatible LLM for strict-JSON script entries; Qwen3-TTS voices them
(local engine or external server).

## Config — easy to regress, read carefully

- The UI server resolves its settings file via the `ALEXANDRIA_CONFIG_PATH`
  env var when set, else `<repo>/app/config.json`.
- The worker subprocesses (cwd = `app/`) MUST resolve config the same way.
  `generate_script.py`, `review_script.py` and `generate_personas.py` therefore
  all use:

  ```python
  config_path = os.environ.get("ALEXANDRIA_CONFIG_PATH") or os.path.join(
      os.path.dirname(os.path.abspath(__file__)), "config.json")
  ```

  Historically they hardcoded `<app>/config.json`, silently fell back to the
  `localhost:11434/v1` (Ollama) defaults, and the WebUI LLM settings were
  ignored. Never reintroduce a hardcoded path that skips the env var — fix all
  three workers together.
- `POST /api/config` rewrites `config.json` from its own schema
  (`config.model_dump()`), so hand-added keys are dropped on the next settings
  save. Anything that must survive a UI save belongs in code, not only in the
  config file.

## LLM provider gotchas

- The app talks to any OpenAI-compatible endpoint
  (`llm.base_url`, `llm.api_key`, `llm.model_name` in config).
- Reasoning models (DeepSeek `deepseek-v4-*`) enable hidden "thinking" by
  default: they burn the whole `max_tokens` budget on reasoning, emit no
  `content`, and every JSON parse fails (`finish_reason=length`). The workers
  therefore wrap `client.chat.completions.create` (right after client creation)
  to inject `extra_body={"thinking": {"type": "disabled"}}` whenever `base_url`
  contains `deepseek.com`; config `llm.extra_body` can override. Keep the check
  host-scoped — llama.cpp and other endpoints must not receive that field.
- LLM tasks demand strict JSON output and retry up to 3 attempts. A truncated
  response is a token-budget/reasoning problem, not a network problem: raise
  `generation.max_tokens` (config) and/or ensure thinking is disabled.
- `--single-speaker` mode in `generate_script.py` bypasses the LLM entirely.

## TTS

- `app/tts.py` (qwen-tts) loads Qwen3-TTS Base / CustomVoice / VoiceDesign
  models, plus LoRA adapters. Dtype comes from `_compute_dtype()`: bf16 on
  NVIDIA CUDA and discrete ROCm GPUs; fp32 on AMD APU iGPUs (device names like
  `Radeon 660M`/`680M` — bf16 HIP GEMMs SIGSEGV there on gfx1035/Rembrandt with
  ROCm 6.4). Don't revert to a blanket `bf16 if "cuda" in device`.

## Docker / ROCm

- `Dockerfile` is upstream's CUDA build (`pytorch/pytorch:2.8.0-cuda12.8-...`).
- `Dockerfile.rocm` produces the AMD variant used on aibox: swaps in
  `torch==2.8.0+rocm6.4`, installs matching ROCm `torchaudio`, drops
  `torchvision`. Rebuild both whenever app code changes:

  ```bash
  sudo podman build -t localhost/alexandria:latest .
  sudo podman build -f Dockerfile.rocm -t localhost/alexandria:rocm .
  ```

  (Container definition, volumes and the `HSA_OVERRIDE_GFX_VERSION` env live in
  the separate `infrastructure-nix-flake` repo, not here.)

## Dev loop / validation

- No automated tests. Validate with `python -m py_compile` on changed files and
  a real worker run on a small text file:
  `cd app && python -u generate_script.py /tmp/sample.txt` (one small LLM call).
- Runtime state files (`annotated_script.json`, `chunks.json`,
  `voice_config.json`) are written next to `app/` (repo root when workers run
  with cwd=`app/`) and are gitignored.
- Match surrounding style; keep diffs small and targeted.
