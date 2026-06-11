# Preprocessor

Reads one frame at a time, resizes it to the model's input resolution, normalises pixel values, and stacks the source frame with the alpha hint into a 4-channel tensor.

## Reference

::: corridorkey.stages.preprocessor.orchestrator.preprocess_frame

## Usage

```python
from corridorkey import load_config
from corridorkey.stages.preprocessor.orchestrator import preprocess_frame
from corridorkey.stages.loader.validator import list_frames

config = load_config()
preprocess_config = config.to_preprocess_config(device="cuda", resolved_img_size=2048)

image_files = list_frames(manifest.frames_dir)
alpha_files = list_frames(manifest.alpha_frames_dir)

preprocessed = preprocess_frame(
    manifest,
    frame_index=0,
    config=preprocess_config,
    image_files=image_files,
    alpha_files=alpha_files,
)

print(preprocessed.tensor.shape)   # [1, 4, img_size, img_size]
print(preprocessed.meta.frame_index)
print(preprocessed.meta.source_h, preprocessed.meta.source_w)
```

## Output tensor layout

The output tensor has shape `[1, 4, H, W]` where H and W are `img_size`. The four channels are:

- Channels 0-2: source frame (RGB, normalised to [0, 1])
- Channel 3: alpha hint (grayscale, normalised to [0, 1])

## Pre-loading frame lists

Pass `image_files` and `alpha_files` to avoid re-scanning the directory on every frame. Use `list_frames()` from `corridorkey.stages.loader.validator` to get a sorted list of frame paths.
