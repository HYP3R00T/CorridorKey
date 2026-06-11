# Inference

Runs the neural network on a preprocessed frame and returns the predicted alpha matte and foreground colour.

## Reference

::: corridorkey.stages.inference.orchestrator.run_inference

::: corridorkey.stages.inference.contracts.InferenceResult

## Usage

Inference is typically called through `Engine` or `run_clip()`. Direct use requires loading a model first:

```python
from corridorkey import load_config
from corridorkey.stages.inference.loader import load_model
from corridorkey.stages.inference.orchestrator import run_inference

config = load_config()
inference_config = config.to_inference_config(device="cuda")

model = load_model(inference_config)

result = run_inference(preprocessed_frame, model, inference_config)

print(result.alpha.shape)   # [1, 1, img_size, img_size] — GPU tensor
print(result.fg.shape)      # [1, 3, img_size, img_size] — GPU tensor
```

## Output tensors

`InferenceResult.alpha` and `InferenceResult.fg` are PyTorch tensors on the inference device (GPU). They are transferred to CPU by the postprocessor before numpy operations.

- `alpha`: shape `[1, 1, H, W]`, float16 or float32, values in [0, 1]
- `fg`: shape `[1, 3, H, W]`, float16 or float32, values in [0, 1]

## AlphaGenerator protocol

The alpha slot is pluggable. See [Device and Backends](../../../architecture/device-and-backends.md) for the full protocol specification and optional config wiring.
