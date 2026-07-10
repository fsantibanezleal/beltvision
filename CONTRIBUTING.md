# Contributing

Thanks for working on beltvision. This is a reusable library, so the bar is a stable
public API, real tests, and a clean import boundary between the slim classical core
and the heavy optional engines.

## Golden rules

- English only, in code, comments, docs, and commit messages.
- Never commit secrets.
- Never use `curl` in scripts or checks. Use Python `httpx` if a check needs HTTP.
- Keep the core dependencies minimal. A heavy engine (torch, onnxruntime, ultralytics,
  anomalib, transformers) belongs in the `[dl]` extra and must be imported lazily,
  never at module import time.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows
# source .venv/bin/activate       # Linux / macOS
python -m pip install --upgrade pip
pip install -e ".[dev]"           # editable install with test + lint tooling
```

To work on the heavy precompute methods, also install the extra:

```bash
pip install -e ".[dev,dl]"
```

### Developing alongside a consumer product

A consuming product depends on beltvision. For a local edit-test loop across both, do
an editable install of this package into the consumer's environment:

```bash
# from the consumer repo, pointing at your local checkout:
pip install -e ../beltvision
```

The consumer's pinned dependency uses a git tag ref until beltvision publishes to
PyPI; the editable install above is a development-only override.

## Run the tests and linters

```bash
python -m ruff check beltvision tests
python -m pytest -q
```

## Local validation gate (must pass before you open a PR)

1. `python -c "import beltvision; print(beltvision.__version__)"` imports cleanly.
2. `python -m ruff check beltvision tests` is clean.
3. `python -m pytest -q` passes.
4. `beltvision synth_tear_gt --quick --out ./derived` runs end-to-end.

## Branch and PR flow

```
task/<slug>  ->  develop  ->  main
```

- Branch from `develop` as `task/<short-slug>`.
- Keep commits focused; write English commit messages in the imperative mood.
- Update `CHANGELOG.md` under `[Unreleased]` with every user-facing change.
- Open the PR against `develop`. `main` is release-only and is tagged `vX.YY.ZZZ`.

## Releases

A release is a tag `vX.YY.ZZZ` on `main`. Publishing to PyPI is automated by GitHub
Actions via OIDC Trusted Publishing (no stored token); see
`.github/workflows/publish-pypi.yml`. The project stays on `0.x` while the API is
still stabilizing.
