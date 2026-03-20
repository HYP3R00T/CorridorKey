"""Inference stage — public surface.

Entry points:
    load_model(config)                    -> nn.Module
    run_inference(frame, model, config)   -> InferenceResult

Contracts:
    InferenceConfig   — checkpoint path, device, precision, optimization mode
    InferenceResult   — alpha [1,1,H,W], fg [1,3,H,W], FrameMeta
"""

from corridorkey_new.inference.config import InferenceConfig, OptimizationMode
from corridorkey_new.inference.contracts import InferenceResult
from corridorkey_new.inference.loader import load_model
from corridorkey_new.inference.orchestrator import run_inference

__all__ = [
    "load_model",
    "run_inference",
    "InferenceConfig",
    "InferenceResult",
    "OptimizationMode",
]
