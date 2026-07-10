"""Model-artifact descriptors.

The pipeline trains heavy models in the precompute lane and exports them to compact
ONNX artifacts committed under ``models/``. This subpackage describes such an
artifact (bytes, opset, input shape, measured CPU latency) so the gate and the
manifest can reason about it without importing any deep-learning framework.
"""
