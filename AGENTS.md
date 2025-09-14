# Repository Guidelines

## Project Structure & Module Organization
- Python source: `reservoir/` (entrypoints: `__main__.py`, `cli.py`).
- Utilities: `utils/` (GPU helpers, metrics, plotting, preprocessing).
- Tests: `tests/` (`test_*.py`).
- Configs: `configs/` (JSON demos: `sine_wave_demo_config.json`, `lorenz_demo_config.json`).
- Scripts: `scripts/` (CUDA setup, rebuild helpers). Docs in `docs/`, example outputs in `outputs/`.

## Build, Test, and Development Commands
- Install deps (Python 3.13+): `uv sync`
- Run CLI: `uv run reservoir-cli --help`
- Run module: `uv run python -m reservoir --config configs/sine_wave_demo_config.json`
- CPU tests: `uv run poe test` (equivalent: `uv run pytest -q`)
- GPU smoke test (skips if no GPU): `uv run poe test-gpu`
- GPU demos (CUDA 12): `uv run poe demo-sine-gpu` or `uv run poe demo-lorenz-gpu`

## Coding Style & Naming Conventions
- Follow PEP 8 with 4‑space indentation and type hints.
- Naming: modules/functions/variables `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE_CASE`.
- Docstrings: concise summary + args/returns; prefer NumPy style where helpful.
- Keep files focused; colocate helpers in `utils/`; core RC logic in `reservoir/`.

## Testing Guidelines
- Framework: `pytest`. Place tests under `tests/` and name `test_*.py` with functions `test_*`.
- Add unit tests for new logic and edge cases; prefer deterministic seeds.
- GPU-specific checks belong in `tests/test_gpu_*.py`; guard heavy tests with markers if needed.
- Run locally: `uv run pytest -q`; for CUDA paths use Poe tasks above.

## Commit & Pull Request Guidelines
- Commits: short, imperative subject (≤72 chars). Examples: `add poe tasks`, `update structure`, `fix metrics edge case`.
- PRs: clear description, rationale, and scope; link issues. Include:
  - What changed and why
  - How to test (commands used)
  - Screenshots/plots for demos when relevant
  - Notes on GPU/CUDA requirements

## Security & Configuration Tips
- CUDA 12 required for GPU: ensure drivers match. Poe tasks set `JAX_PLATFORMS=cuda` and disable XLA preallocation.
- Avoid committing large artifacts; write outputs to `outputs/` and add to `.gitignore` if new patterns arise.
