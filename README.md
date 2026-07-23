# beltvision

A reusable computer-vision inspection engine for industrial conveyor belts. It bundles
the classical vision methods (CLAHE preprocessing, belt-edge geometry, anomaly,
segmentation, granulometry, tracking) together with the machinery that makes an
inspection reproducible and auditable: a named staged pipeline, two data contracts, a
measured live/precompute lane gate, and a validated artifact manifest.

[![CI](https://github.com/fsantibanezleal/beltvision/actions/workflows/ci.yml/badge.svg)](https://github.com/fsantibanezleal/beltvision/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/beltvision.svg)](https://pypi.org/project/beltvision/)
[![License](https://img.shields.io/github/license/fsantibanezleal/beltvision.svg)](LICENSE)
[![Version](https://img.shields.io/github/v/tag/fsantibanezleal/beltvision.svg?label=version&sort=semver)](https://github.com/fsantibanezleal/beltvision/tags)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21519891-blue)](https://doi.org/10.5281/zenodo.21519891)

The PyPI badge goes live on the first published release. Until the API stabilizes,
install from a git tag ref (see Install).

## What it is / is NOT

It IS:

- A domain-agnostic inspection engine for belt imagery, importable with a slim
  classical stack (numpy, opencv, scikit-image, scipy).
- A frozen six-stage pipeline (`preprocess`, `feature_extraction`, `train`, `infer`,
  `evaluate`, `export`) whose stage signatures are stable and whose bodies are the
  rework surface for heavier methods.
- Two explicit data contracts: an ingestion gate (bad input is rejected, borderline
  input is flagged, clean input is accepted) and an artifact manifest that a viewer
  can replay without re-running the pipeline.
- A measured lane gate that decides where a capability runs (in a browser, on a CPU
  server, or offline as a precompute artifact) from measured numbers, never a label.

It is NOT:

- A trained-model zoo. The heavy deep-learning engines (torch, onnxruntime,
  ultralytics, anomalib, transformers) are optional extras imported lazily inside the
  precompute lane, never at import time.
- A web service or an application. It is a library other applications consume.
- A dataset. The bundled cases are a registry plus deterministic synthetic scenes;
  real frames are brought by the consumer.

## Install

Dependencies are declared entirely in `pyproject.toml` as three levels (extras) - there are
no `requirements-*.txt` files:

```bash
# Core - classical CV only (numpy / opencv / scikit-image / scipy). Tiny, pure CPU:
pip install beltvision

# CPU RUNTIME level ("web running") - CPU torch + onnxruntime + light learned models.
# This is what the app's CPU venv (VPS emulation) installs:
pip install "beltvision[dl]"

# GPU TRAINING / PRECOMPUTE level - all the heavy models, run locally on a CUDA GPU.
# The CUDA torch build is the only thing an extra cannot pin (it needs a package index),
# so pass that index once at install time:
pip install "beltvision[gpu]" --extra-index-url https://download.pytorch.org/whl/cu124
```

Two venvs, one per level: a CPU `.venv` (VPS emulation, `[dl]`) and a `.venv-gpu`
(`[gpu]`, CUDA). Verified on an RTX 5000 Ada (sm_89), torch 2.6.0+cu124.

Until the API stabilizes and the first release is published to PyPI, pin a git tag:

```bash
pip install "beltvision @ git+https://github.com/fsantibanezleal/beltvision@v0.1.0"
```

## Quickstart

Run one synthetic control case through all six stages (no external data required):

```bash
beltvision synth_tear_gt --quick --out ./derived
# equivalently:
python -m beltvision.pipeline synth_tear_gt --quick --out ./derived
```

From Python:

```python
from beltvision.pipeline import run_case

manifest = run_case("synth_tear_gt", quick=True, out_root="./derived")
print(manifest["case_id"], [m["lane"] for m in manifest["methods"]])
```

Validate raw input against the ingestion contract:

```python
from beltvision.io.contract import validate_image
from beltvision.io.schema import IngestionParams

result = validate_image(frame_bgr, IngestionParams(px_per_mm=3.2))
if not result.accepted:
    raise ValueError(result.reasons)
```

## Architecture

Three lanes decouple where a capability can run:

- Live-web: small models that a browser runtime can drive within a tight budget.
- Live-server: models a CPU server can hold and run within a latency budget.
- Precompute: everything heavier, run offline and shipped as a committed artifact.

The lane is not declared; it is measured. `beltvision.core.gate.classify_lane` takes
model bytes, inference milliseconds, trace bytes, and web-drivability and returns the
lane plus the numbers it decided from. That verdict rides into the manifest, so a
consumer can render the lane story of a run one to one.

The pipeline threads a single `StageContext` through six frozen stages. Each stage is
a pure-ish function `(StageContext) -> dict` that records its timing into a compact
trace. Stage names and signatures are frozen; stage bodies are the rework surface
where heavier methods attach.

## Data and contract

Contract 1 (ingestion) is the bring-your-own-data gate. The policy is explicit and
lives in one place:

- Reject: undecodable or corrupt input, wrong channel count, out-of-range resolution,
  or a payload over the size cap.
- Flag: decodable and in range but low-contrast (dusty or hazy) below the contrast
  floor. The frame is still processed and the flag rides along so downstream
  confidences can be honestly down-weighted.
- Accept: decodable, in range, adequate contrast.

Contract 2 (the artifact manifest) is the compact, validated description of one run.
Every numeric claim a consumer shows must trace back to a field in the manifest, and
the schema is versioned so a drifted consumer mirror fails its own build.

## Methods

The engine ships a lightweight, deterministic method ladder on the slim classical
stack (CLAHE, Canny/Sobel edge geometry, a patch-Gaussian anomaly baseline) that runs
live and offline in a test. The heavier methods (learned anomaly baselines,
transformer and detection backbones, segmentation and tracking engines) attach to the
same stages in the precompute lane through the `[dl]` extra. Method results carry
references to the underlying techniques.

## Versioning and changelog

This project follows a three-segment version scheme and Keep a Changelog. See
[CHANGELOG.md](CHANGELOG.md). Each release is tagged `vX.YY.ZZZ`. The project stays on
`0.x` while the extracted API is still stabilizing.

## License

Apache-2.0. See [LICENSE](LICENSE).

## Author

Felipe Santibanez-Leal.
