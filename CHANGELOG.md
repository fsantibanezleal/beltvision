# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and the project uses a
three-segment `X.YY.ZZZ` version scheme with a `vX.YY.ZZZ` git tag per release.

## [Unreleased]

## [0.11.001] - 2026-07-13

### Changed

- **`analyze_scene` now uses the robust cascade** for the three sections that were trash on
  real frames: `belt_geometry` → `robust.belt_band`, `damage` → `robust.damage`, `edges` →
  `robust.edge_condition`. The belt band is estimated ONCE and threaded into damage + edges
  (the cascade). Each section's `metrics` now carries `per_pipeline` + `confidence` so a viewer
  can show the fused result and each pipeline's contribution; the old mask-derived
  `belt_geometry`/`belt_damage`/`edge_condition` (medial-axis diagonal, "0 regions",
  `applicable:false`) are no longer wired into the scene.

### Fixed

- **Import cycle** `analysis → render → methods.semantic → methods/__init__ → constrained →
  render`: `render` now imports `CLASS_COLORS_BGR` lazily inside the two overlay functions that
  use it, so `beltvision.analysis` imports cleanly (the cycle was latent — the deploy gate only
  exercised precomputed replay, not live `analyze_scene`).

## [0.11.000] - 2026-07-13

### Added

- **`beltvision.methods.robust` — staged (cascade) multi-pipeline belt analysis.** Replaces the
  single fragile mask-derived analyses (which produced a diagonal-nonsense centreline, "0
  regions / severity none", and `applicable: false` on real frames). Each stage runs SEVERAL
  complementary pipelines fused with an agreement-based confidence, and always shows informative
  content — never a bare "not applicable" or an invented single line.
  - `orientation_consensus(gray)` — global belt-travel alignment fused from Radon (noise/
    illumination-robust), FFT dominant peak, and the structure tensor; agreement = confidence.
  - `belt_band(image)` — two belt limits via **signed oriented-gradient projection onto the
    belt-normal axis** (the two most-prominent opposite-polarity peaks = the parallel belt
    edges), cross-checked by constrained Hough, fused. **Centreline = the MIDLINE of the two
    limits** (never a medial axis); width = their perpendicular separation. Confident on clean
    belts, honest low-confidence candidates + Studio deferral on ambiguous frames (validated on
    the real COLA 34 water-curtain frame and clean iron-ore frames).
  - `damage(image, band=...)` — RGB **anomaly ensemble** (illumination residual + wavelet
    texture-removal + FFT band-stop + morphological black/top-hat) INSIDE the validated band,
    fused to a heatmap + flagged regions; states the RGB-only limitation honestly (no laser/
    depth/labelled defects) and flags likely-loaded (content-occluded) bands.
  - `edge_condition(image, band=...)` — border strength/continuity/roughness/notches sampled
    ALONG the validated limits (not a raw mask boundary); conservative RGB heuristic; gates to
    n/a when the band is low-confidence.
  - `band_mask_from_edges(shape, edge_a, edge_b)` helper. 8 regression tests; ruff clean.

## [0.10.001] - 2026-07-13

### Fixed

- **Belt-axis orientation prior with one ROI per belt edge (`roi.orientation_band`).** When a
  user draws a separate ROI strip on each belt edge (the required per-ROI workflow),
  `orientation_band` rasterised all belt annotations jointly and took a single oriented box.
  Two vertical strips placed left/right form a box wider than tall, so the joint long edge
  came out **horizontal** — the constrained Hough then searched the wrong orientation and
  found only one line, so `extract_belt_edges` failed. Now `_per_annotation_orientation`
  measures each annotation's own long-edge angle and takes the period-180 circular mean
  (`_axis_circular_mean`), recovering the true belt axis regardless of how the strips are
  arranged. Proven on the real COLA 34 frame: belt axis 90° (was ~horizontal), 10 in-band
  lines across two ROIs, belt edges found, width 325 px, one edge per ROI + centreline.
- **`__version__`** was left at 0.9.0 while `pyproject` moved to 0.10.0; both are now 0.10.1.

## [0.10.000] - 2026-07-13

### Fixed

- **Node DAG threading (`pipeline_graph._edge_map`).** A detect node fed a non-binary image
  rebuilt its edge map from `ctx["frame"]` (the raw original), silently discarding every
  upstream CLAHE / denoise / ROI stage. It now preprocesses the actual node input, so the
  pipeline threads correctly.

### Added

- **Per-ROI detection.** `pipeline_graph.apply_roi` now rasterises each drawn annotation
  separately and exposes them as `ctx["roi_masks"]`. When more than one ROI is present, the
  `hough_constrained` and `ransac_line_constrained` ops run **once per region** and merge the
  results, instead of detecting on a single unioned mask — so a left-edge ROI and a
  right-edge ROI are analysed independently.
- **Belt-edge pair extraction.** `constrained.extract_belt_edges(segments, theta_center_deg)`
  selects the two belt edges from a set of in-band lines via a gap-maximising 2-cluster split
  (longest representative per cluster, so noise loses to a true edge), and reports
  `edge_a` / `edge_b`, belt `width_px` (and mm when a scale is set), and the belt centreline.
  Exposed as the `belt_edges` pipeline op and wired into the `belt_detection` template
  (`… → hough_constrained → belt_edges → measure_lines`).
- Regression tests: `extract_belt_edges` pair/width/noise-robustness/degenerate cases;
  per-ROI mask population; the `belt_edges` op inside a pipeline.

## [0.09.000] - 2026-07-11

### Added

- **Guided classical-CV engine (Phase 1): detectors now run on a preprocessed, thresholded,
  ROI-masked, orientation-constrained input, never the raw frame.** This fixes the defect
  where firing Hough / RANSAC on the raw frame locked onto noise and "found lines in all
  directions". New modules:
  - **`beltvision.methods.roi`** - rasterises user annotations
    (`freehand` / `polygon` / `rect` / `line` + `label`) to boolean masks at frame
    resolution (`rasterize`, `combine_by_label`), derives the belt-band prior
    (`belt_limit_prior` -> band mask + guide orientation + two guide edge lines from an
    `expected-belt-limits` / `belt-section` region or two guide lines), and returns the
    orientation gate a constrained detector consumes (`orientation_band(view, annotations)`
    -> `(theta_center_deg, theta_band_deg)`: top/end near the annotated/detected axis,
    lateral near-horizontal, PCA/Radon fallback).
  - **`beltvision.methods.constrained`** - the fix itself:
    `preprocess_for_lines` (gray -> CLAHE -> gaussian/median/bilateral denoise ->
    canny/log/sobel edge -> restrict to ROI), `gradient_orientation_gate` (keep only edge
    pixels whose Sobel gradient orientation is within the belt-normal band),
    `hough_constrained` (skimage `hough_line` with a THETA VECTOR limited to the band ->
    `hough_line_peaks` -> straight in-band segments only; a line outside the band is
    un-representable in the accumulator), and `ransac_line_constrained` (RANSAC over ROI
    edge points with an orientation-band `is_model_valid` reject -> straight in-band lines
    only, no cross-direction junk). Each returns the uniform overlay-carrying method record.
    Registered in the ladder `REGISTRY` as `geometry.hough_constrained` /
    `geometry.ransac_line_constrained` (family `constrained_lines`).
  - **`beltvision.methods.transforms`** - classical transforms as records + reusable
    array helpers: FFT (`fft_spectrum`, `fft_orientation`, `fft_filter`
    directional/band/low/high/notch + inverse, `phot` phase-only anomaly map, numpy.fft,
    no new dep) and wavelet (`dwt_decompose`, `dwt_reconstruct` subband selection for
    texture removal / anomaly enhancement, `wavelet_denoise` translation-invariant
    BayesShrink). Registered under a new `transform` capability with families
    `frequency_transform` / `wavelet`.
  - **`beltvision.methods.measure`** - pure, exact-number measurement (ImageJ Analyze
    model): `angle_between`, `segment_length`, `polygon_area`, `polygon_perimeter`,
    `count_objects`, `density`, `calibrate_scale` (px-per-mm) and px<->mm / px2<->mm2
    converters.
  - **`beltvision.pipeline_graph`** - the Pipeline Studio node DAG engine. `OP_REGISTRY`
    (34 ops across source/roi, preprocess, transform, binarize/morphology, detect, measure)
    reuses the real method implementations. `run_pipeline(spec, image, rois, priors)`
    executes the graph topologically, threads each node's output image to its consumers and
    captures EVERY node's overlay + metrics; one node failing is recorded in `errors` and
    never aborts. Three correctly-staged `TEMPLATES` (`belt_detection`, `belt_condition`,
    `material_on_belt`) with `list_templates()` / `get_template()` / `op_catalog()`.
- **`PyWavelets` (`pywt`) added to the beltvision CORE dependencies** - it is a light CPU
  RUNTIME op (DWT), so it belongs in core, not the `[gpu]` extra.
- **Tests** (`test_roi`, `test_constrained`, `test_transforms`, `test_measure`,
  `test_pipeline_graph`): ROI rasterisation exactness; the constrained Hough returns in-band
  lines ONLY on a synthetic cross-hatch while a raw full-theta Hough returns out-of-band
  lines; exact measurement numbers; every transform yields a real PNG overlay + finite
  metric; and every pipeline template runs end-to-end returning per-node overlays. Proven on
  the real COLA 34 frame: the constrained Hough finds 6 clean near-vertical belt lines (mean
  axis 91deg, all in-band) where the raw `features.hough_lines_p` finds 453 scattered noise
  lines (dominant axis 1.7deg).

## [0.08.000] - 2026-07-11

### Added

- **`beltvision.methods.foundation`** - a REAL beyond-SOTA (open-vocabulary / foundation-model)
  method tier that runs on the GPU in the offline precompute lane and returns each result as the
  same uniform overlay-carrying record the classical / sota methods do (maturity `tier =
  "beyond_sota"`), so a serving product replays the committed overlay with zero live compute.
  Six genuinely-loading-and-running methods, each with a legible drawn overlay + a scalar metric:
  - `features.dinov2` - DINOv2 (ViT-B/14, `facebook/dinov2-base`) dense self-supervised patch
    features, PCA(1-3) -> RGB feature map upsampled to the frame (metric: explained variance).
  - `anomaly.dinov2_knn` - per-patch kNN cosine distance over the DINOv2 feature grid -> a
    foundation-feature anomaly heatmap (AnomalyDINO-style, a PatchCore analogue; metric:
    max nearest-neighbour distance), peak restricted to the belt footprint when available.
  - `depth.depth_anything_v2` - Depth-Anything-V2-Small monocular relative depth -> colorised
    belt-surface relief overlay (metric: relative depth range).
  - `detection.owlv2` - OWLv2 (`google/owlv2-base-patch16-ensemble`) open-vocabulary detection
    of belt-domain objects (foreign object / wood / metal tool / person) by text prompt -> boxes
    + labels + scores (metric: n detections).
  - `segmentation.grounded_sam` - GroundingDINO (`grounding-dino-tiny`) open-vocab boxes ->
    SAM (`sam-vit-base`) masks, colored + text-labelled (metric: n masks; belt coverage in extra).
  - `segmentation.sam2` - SAM 2 (`sam2_b.pt`, ultralytics) prompt-free automatic mask generation
    with a mask-boundary overlay (metric: n masks), a stronger segmenter than the MobileSAM sota
    entry (both kept).

  Heavy libs (torch / transformers / ultralytics / PIL) are lazily imported inside each function,
  each loaded model is cached module-wide across the batch, inputs are long-side-capped for VRAM,
  and everything runs on `device='cuda'`. A model that will not run RAISES so the precompute
  wrapper records it in `errors` and skips it - never a fabricated or empty-overlay entry.
- **`precompute_methods(..., include_foundation=True)`** - a new gate that runs the beyond-SOTA
  block only when `include_foundation` is set AND `device` is a CUDA device, so the slim CPU
  runtime never loads a foundation model. Each foundation method is guarded like the others (one
  failure -> `errors`, never abort).
- **Beyond-SOTA methods in the ladder `REGISTRY`** - the six foundation methods are registered
  with `tier="beyond_sota"` so `methods_by_tier()["beyond_sota"]`, `method_index()` and `families()`
  surface them. Their LIVE registry callable degrades to a graceful `weights_absent` on the CPU/VPS
  runtime (the model is never hosted live; catalogue cases replay the precomputed overlay), so
  `run_ladder` stays fast and honest. A new `depth` capability joins the ladder.

### Changed

- `pyproject` `[gpu]` extra: added `pillow` (used directly by the foundation preprocessing).

## [0.07.001] - 2026-07-11

### Added

- **`beltvision.precompute.precompute_methods`** - an offline helper that runs the FULL
  per-method toolbox on one still frame and returns each method as a uniform, JSON-safe record
  carrying a drawn overlay (base64 PNG), a scalar metric, and its maturity `tier` + `family`,
  so a serving product can precompute + replay every analysis with no live gaps. It runs the
  19-operator classical feature bench (`features.run_all`) + the consolidated geometry read
  (`geometry.analysis`), the 4-class semantic map, unsupervised anomaly (PaDiM-lite
  self-reference, the trained conv-AE reconstruction residual, and the PaDiM + PatchCore-lite
  frozen-backbone banks - each a per-position colour heatmap), watershed granulometry on the
  segmented content, dense optical flow, and MobileSAM automatic masks. The 4 semantic layers
  are segmented ONCE and shared; each method is wrapped so one failure is recorded and skipped,
  never aborting. Torch / onnxruntime / ultralytics are imported lazily; GPU is used for the
  frozen-backbone banks + MobileSAM when `device='cuda'`.
- **Reusable overlay renderers** (`beltvision.render`): `heatmap_overlay` (normalize +
  upsample + colour-map a per-patch anomaly score grid, with a named peak marker),
  `flow_overlay` (dense optical-flow magnitude heat + a sparse arrow field), `granulometry_overlay`
  (colour each watershed particle + its boundary) and `masks_overlay` (SAM automatic-mask boxes)
  - so the anomaly / flow / granulometry / SAM methods, whose live results carry numbers but no
  drawn overlay, get a legible legend + result-bar overlay in the precompute lane.

### Changed

- `precompute.backbone.ResNetPatchFeatures` accepts a `device` argument and extracts the frozen
  ResNet-18 patch features on that device (CPU default; `'cuda'` for the precompute lane).

## [0.07.000] - 2026-07-11

### Added

- **Classical feature / edge / keypoint / texture toolbox** (`beltvision.methods.features`).
  A broad bench of textbook classical operators, each implemented for real on
  opencv / scikit-image / numpy and each returning a visually DISTINCT drawn overlay (base64
  PNG) plus a single scalar/count metric, a family and a citable reference. CLAHE-first, and
  where a belt region is relevant they run INSIDE the segmented belt-footprint mask. The 19
  operators span six families: edge operators (Canny, Sobel, Scharr, Laplacian,
  Laplacian-of-Gaussian, Prewitt, Roberts-cross, morphological-gradient - metric edge
  density); lines/boundaries (HoughLinesP with a dominant-angle read, a RANSAC straight-LINE
  fit of the two belt-mask boundaries via skimage `LineModelND`, Radon orientation);
  superpixels (SLIC); shape (oriented bounding box via `minAreaRect`, external contours);
  corners/keypoints (Harris, Shi-Tomasi good-features, ORB); and texture (a Gabor filter bank
  with a dominant-orientation read, and a Local Binary Pattern map with a texture-entropy
  metric). `features.run_all(image, mask=None, methods=None)` segments the belt ONCE and runs
  the whole bench, returning `{"methods": [{id, name, family, tier, reference, metric_name,
  metric_value, overlay_b64}, ...]}`. Every operator is also individually registered in the
  method REGISTRY.
- **Consolidated straight-line geometry analysis** (`geometry.analysis`). One legible overlay
  that surfaces the corrected STRAIGHT-LINE beltline (a least-squares straight centreline plus
  two quasi-parallel straight edges - never a reintroduced parabola/curve) with the belt
  orientation angle, OBB (angle + w/h), belt width (px) and edge parallelism (deg), and
  cross-checks the belt axis against Hough, a RANSAC boundary-line fit and Radon on the same
  frame, with a numeric read-out panel + legend. It never withholds under low confidence: it
  always draws the estimate and labels the confidence.
- **Maturity-tier tagging + grouping helpers.** Every REGISTRY method now carries a `tier` in
  `{classical, sota, beyond_sota}` (the front-end grouping axis, distinct from the per-method
  compute tier and the measured lane): classical = the whole feature/edge bench plus
  granulometry / optical-flow / Kalman geometry; sota = the trained belt segmenter
  (`semantic_layers`), the PaDiM / Conv-AE anomaly methods, MobileSAM and the ONNX detector;
  beyond_sota is reserved for the open-vocab GroundedSAM / DINOv2 / AnomalyDINO frontier
  (registered when present). New helpers `methods_by_tier()`, `families()` and
  `method_index()` let a UI group the ladder by tier or family. A per-method `family` axis was
  added to `MethodSpec`.
- **Feature-bench + straight-geometry tests** (`tests/test_features.py`): `features.run_all`
  returns >= 16 methods, each with a real non-empty PNG overlay + a finite metric and a
  distinct overlay; every feature method is registered, tier-tagged and runs to an `ok`
  envelope; and the consolidated geometry keeps the centreline STRAIGHT (least-squares
  curvature < 0.05 on a straight synthetic belt at multiple orientations) while never
  withholding an estimate.

## [0.06.000] - 2026-07-11

### Added

- **Temporal / dynamic sequence-video engine** (`beltvision.precompute.dynamic`). Given an
  ordered frame sequence (or frames sampled from a video), `analyze_sequence(...)` runs the
  per-frame ladder (CLAHE -> semantic segmentation -> belt geometry -> content coverage) PLUS
  the genuinely temporal analyses that only exist across frames: dense Farneback optical flow
  for a per-frame belt SPEED + material-flow direction; a ByteTrack-style associator over
  per-frame contrast-blob boxes for object/particle TRACKING with stable ids; belt-footprint
  centroid DRIFT (a lateral wander trend vs a baseline) plus belt-axis angle over time;
  content COVERAGE over time; and an EVENT stream (object appear/disappear, foreign
  appear/clear, belt stop/start). Deterministic given the seed; the classical ops and the
  association carry no randomness.
- **Annotated output video + metric timelines.** `precompute_sequence(...)` renders one
  legible overlay per frame (belt edges/centreline or footprint, tracked fragments with ids,
  a belt-flow arrow and a live metrics HUD) and encodes them to a compact H.264 `annotated.mp4`
  via `imageio` + `imageio-ffmpeg` (bundled ffmpeg - no system install, no network). It writes
  `timelines.json` (`belt_speed[t]`, `flow_direction[t]`, `coverage[t]`, `drift[t]`,
  `track_count[t]`, `n_foreign[t]`, `haze[t]`, `moving[t]`, `events[t]`) and a `manifest.json`
  (engine/seed/device, frame size, fps/hold/duration, video bytes/codec, the temporal-method
  list and a metrics summary). The slim runtime never imports this module - it replays the
  committed mp4 + timelines; `imageio` is imported lazily only inside `encode_video`.
- **Temporal-smoke tests** (`tests/test_dynamic.py`): a tiny synthetic sliding-window
  sequence exercises the per-frame ladder + optical-flow speed recovery + blob tracking +
  drift + events deterministically (always-on, classical only), and an `importorskip`-gated
  test proves the mp4 encode/decode round-trip in the precompute lane.

## [0.04.000] - 2026-07-10

### Added

- **View-aware 3-stage pipeline.** `beltvision.recognize_view(image)` (Stage 1) is a
  classical scene classifier that predicts the `view_type` (`end_return`, `top_carrying`,
  `side_profile`, `oblique_cctv`) from colour-agnostic scene features (centre isotropic
  material fill vs oriented belt-streak coherence, dust/haze, orientation) with a
  confidence and per-view scores. `beltvision.views.VIEW_ANALYSES` (Stage 2) maps a view to
  the analyses that inform it. `beltvision.analyze_scene(...)` (Stage 3) orchestrates the
  whole tool on a frame and returns results grouped BY ANALYSIS, each with a legible overlay
  (base64 PNG), a plain-language summary and JSON metrics.
- **4-class semantic segmentation backbone** (`segmentation.semantic_layers`): every pixel
  is labelled `belt` / `content` / `foreign` / `external` (content = the transported
  material of ANY domain: ore, aggregate, food, packages, recycling...). An always-on
  classical colour+texture+coherence prior (live-thin, CPU) plus an opt-in open-vocab path
  (MobileSAM automatic masks labelled by CLIP zero-shot, offline/precompute). Never
  `weights_absent` - it degrades to the classical prior and reports the engine used. All
  downstream analyses derive from these layers.
- **Orientation-agnostic belt geometry from the mask** (`geometry.belt_geometry`): the belt
  axis comes from the belt-mask structure orientation (works at any orientation - vertical,
  horizontal, diagonal), the centreline is the medial line of the band (straight or curved,
  no forced parametric model), edges follow the mask boundary, width is measured
  perpendicular to the local tangent, and misalignment is the angular difference between the
  belt axis and the supporting-structure axis (detected from the external layer). Degrades
  to an honest low-confidence result on a broad/ambiguous region instead of drawing garbage.
- **Derived analyses** (`beltvision.methods.analyses`): belt damage/integrity (rips/holes/
  wear inside the belt), edge/border condition, surface irregularity + dust/haze, content
  quantity (coverage %, load, granulometry PSD INSIDE the content mask only) and foreign
  objects. Interpretable overlays (`beltvision.render`) with a legend + one-line result.
- **Labelled synthetic ground truth** (`cases.synthetic.synth_scene` / `GT_SUITE`): scenes
  at multiple orientations (vertical, horizontal, 30/45deg diagonal) plus a curved path and
  a misaligned lateral case, each emitting the exact belt mask, centreline, edges,
  orientation, injected damage/foreign boxes and belt-vs-support angle. A BLOCKING gate
  (`tests/test_scene_gt.py`) asserts the pipeline recovers this within tolerance (belt IoU,
  orientation error <= 8deg at every orientation, centreline RMSE, misalignment sign+angle).
- `requirements-precompute-gpu.txt` + a `[gpu]` extra: the local CUDA precompute/training
  lane (device='cuda'), separate from the CPU VPS-emulation runtime.

### Changed

- **Removed the degree-2 polynomial ("parabola") edge fit and the axis-assuming
  misalignment method** (`geometry.ransac_edges`, `geometry.misalignment`). A belt edge is
  never modelled as a forced curve; all belt geometry is derived from the segmented mask
  shape. No `np.polyfit(deg>=2)` remains in the geometry path.
- Bumped to 0.4.0.

## [0.03.000] - 2026-07-10

### Added

- `beltvision.precompute`: the offline PRECOMPUTE lane that trains the belt-specific learned
  anomaly models on REAL normal frames, exports the conv-AE to ONNX so the weight-gated live
  method becomes real, fits the PaDiM / PatchCore-lite banks, and produces an honest held-out
  learned-vs-classical benchmark. Deterministic given the seed, reproducible via
  `python -m beltvision.precompute --data <dir> --models-out <dir> --bench-out <dir>`.
  - `dataset.build_split`: leakage-safe split (normal-only training; all foreign-object /
    anomaly frames test-only, MVTec discipline; a subset of normal frames held out as
    negatives), with a `assert_leakage_free` guard so no frame is in train and test.
  - `train.train_conv_ae`: trains the conv-AE (L1 reconstruction) on CLAHE grayscale normal
    frames and exports it to ONNX (opset 17), with a torch-vs-onnxruntime parity check.
  - `train.fit_padim` / `fit_patchcore`: PaDiM per-position Gaussian and a PatchCore-lite
    random coreset over frozen ResNet-18 layer2+layer3 features; compact `.npz` banks.
  - `benchmark.compute_benchmark`: image-level AUROC + average precision per method
    (classical residual, padim_lite, conv-AE, PaDiM, PatchCore-lite), plus the robustness
    axis (AUROC drop under a synthetic dust/haze perturbation) and the cost axis (model
    bytes + measured CPU ms), written as a machine-readable `benchmark.json` and labeled a
    small-sample proxy with the exact N per class.
- `anomaly.conv_ae` now returns REAL anomaly output (status `ok`) when the trained
  `conv_ae.onnx` is present in the beltvision weights dir; the live method is no longer
  permanently `weights_absent`.
- `models.download`: the MobileSAM (Apache-2.0) weight URL is now verified and its sha256 is
  pinned, so the opt-in httpx fetch enables `segmentation.mobile_sam` to run for real; a
  genuinely missing weight still degrades gracefully to `weights_absent`.
- Tests: leakage-safe split checks (always-on, classical) plus a conv-AE ONNX export smoke
  and a benchmark-compute smoke on a tiny fixture (torch-gated: they skip cleanly on a slim
  runtime venv without a working torch bridge).

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
