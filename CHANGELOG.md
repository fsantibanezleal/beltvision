# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and the project uses a
three-segment `X.YY.ZZZ` version scheme with a `vX.YY.ZZZ` git tag per release.

## [Unreleased]

## [0.01.000] - 2026-07-10

### Added

- Initial extraction of the reusable belt-inspection engine into its own package.
- Two data contracts: ingestion (`beltvision.io`) with the accept/flag/reject outlier
  policy, and the artifact manifest (`beltvision.core.manifest`, Contract 2) with a
  versioned schema, validation, and index roll-up.
- The measured live/precompute lane gate (`beltvision.core.gate`) that classifies a
  capability from measured numbers rather than a label.
- The frozen six-stage pipeline (`preprocess`, `feature_extraction`, `train`, `infer`,
  `evaluate`, `export`) and its `StageContext`, seeded RNG, and stage trace.
- A case registry, deterministic synthetic belt scenes, and framework-free ONNX
  artifact descriptors.
- A console entry point (`beltvision`) to run one case or all cases.
- Minimal core dependencies (numpy, opencv-python-headless, scikit-image, scipy) with
  a `[dl]` extra for the heavy precompute engines and a `[dev]` extra for tooling.
- Test suite covering the ingestion contract, the gate, the manifest, and an
  end-to-end pipeline smoke run.
