# Inference Engine

Runtime engine for running the CorridorKey keying model on individual frames.

This module exposes `CorridorKeyEngine`, which loads a trained checkpoint and
runs the full keying pipeline including preprocessing, model inference, despilling,
matte cleanup, and compositing.

::: corridorkey_core.inference_engine
