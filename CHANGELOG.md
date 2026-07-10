# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and the project uses a
three-segment `X.YY.ZZZ` version scheme with a `vX.YY.ZZZ` git tag per release.

## [Unreleased]

## [0.02.000] - 2026-07-10

### Added

- The LIVE-tier computer-vision method ladder (`beltvision.methods`): a registry mapping
  method-id -> callable + tier + capability + reference, with `run(method_id, image,
  **params)`, `run_ladder`, and `to_manifest_method`. 15 CLAHE-first, tier-tagged methods
  returning JSON-safe results, each measuring its own gate inputs (bytes, ms,
  web-drivability) so `core.gate.classify_lane` tags its lane. Learned methods degrade
  gracefully to `{"status": "weights_absent", ...}` (never an exception) when an optional
  weight is missing, reporting the expected model size so the lane stays honest.
  - Preprocess (mandatory first stage): `preprocess.clahe_lab` - LAB L-channel CLAHE +
    bilateral denoise + a dark-channel/global-contrast dust-haze severity score.
  - Geometry (classical): `geometry.hough_edges` (Canny + HoughLinesP near-vertical
    candidates), `geometry.ransac_edges` (RANSAC deg-2 polynomial left/right edge fit),
    `geometry.radon_orientation` (Radon dominant orientation), `geometry.misalignment`
    (centreline deviation, width profile, skew, flags), `geometry.kalman_edge` (per-camera
    constant-velocity Kalman edge tracker that persists across calls), `geometry.obb`
    (cv2.minAreaRect oriented boxes for the belt and per region).
  - Granulometry (classical): `granulometry.watershed_psd` - CLAHE -> Otsu -> distance
    transform -> watershed -> regionprops -> equivalent diameters -> D10/D50/D80 +
    oversize% + Rosin-Rammler (Weibull) fit + the PSD curve, honest relative-px units when
    uncalibrated.
  - Segmentation: `segmentation.slic` (SLIC superpixels, classical, live) and
    `segmentation.mobile_sam` (MobileSAM/FastSAM automatic masks, learned, `[dl]`).
  - Anomaly (learned): `anomaly.padim_lite` (per-patch Gaussian + Mahalanobis heatmap,
    live on CPU with no weight) and `anomaly.conv_ae` (conv-AE L1 architecture + ONNX/torch
    inference path; weights trained in the precompute lane).
  - Detection (learned): `detection.onnx_detector` - ONNX Runtime CPU detector (RT-DETR /
    permissive YOLO) returning boxes + scores + labels.
  - Tracking: `tracking.optical_flow` (Farneback belt-speed + motion direction) and
    `tracking.bytetrack_associate` (ByteTrack-style two-stage IoU associator over detector
    boxes; the detector is the cost).
- `beltvision.models`: optional-weight provisioning that locates weights on disk and, opt-in,
  downloads them with httpx (never curl). It is the root of the `weights_absent` contract and
  keeps the default runtime and tests fully offline.
- The `infer` stage now runs the full live ladder and folds each result into the Contract 2
  manifest with a measured lane verdict.
- Tests: per-method run + JSON-serializability, the missing-weight graceful path, per-capability
  content checks, and the download/provisioning contract. `httpx` added to the `[dl]` extra.

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
