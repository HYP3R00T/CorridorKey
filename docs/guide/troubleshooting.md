# Troubleshooting

## CUDA out of memory

The model ran out of VRAM during inference.

- Switch to the `tiled` preset: `ck /clips` then select `tiled` in the wizard.
- Reduce `img_size` in your config: try `1024` or `1536`.
- Enable `half_precision` in `[preprocess]`.
- Close other GPU-heavy applications before running.

## Model not found

```text
ModelError: Model verification failed
```

Run `ck init` to download the model:

```shell
ck init
```

If the download fails, check your internet connection and disk space at `~/.config/corridorkey/models/`.

## Clip skipped: already complete

CorridorKey found the expected output frames already present in `Output/`. This is intentional — it skips clips that are already done.

To reprocess a clip, delete its `Output/` directory and run again.

## Clip skipped: no input frames

The `Input/` directory exists but contains no recognised files. Check that your frames are JPEG or PNG, or that your video file is in a format FFmpeg can decode.

## Frame count mismatch

```text
FrameMismatchError: Clip 'X': frame count mismatch — N input frames vs M alpha frames
```

The number of frames in `Input/` and `AlphaHint/` do not match. Regenerate your alpha hint frames so the counts are equal.

## No alpha generator registered

```text
AlphaGeneratorError: Clip 'X' requires alpha frames but no AlphaGenerator is registered
```

The clip has no `AlphaHint/` folder and no alpha generator plugin is registered. Either:

- Add an `AlphaHint/` folder with pre-generated alpha frames.
- Register an `AlphaGenerator` plugin if you are using the library directly.

## Poor edge quality

- Increase `img_size` to `2048` (requires 12+ GB VRAM).
- Try the `quality` or `max_quality` preset.
- Check `postprocess.despill_strength` — very high values can erode edges.
- Enable `postprocess.hint_sharpen` (on by default) — it uses the alpha hint to sharpen edges after upscaling.

## Green fringing on foreground

- Increase `postprocess.despill_strength` (default `0.5`, max `1.0`).
- Check that `postprocess.source_passthrough` is `true` (default) — it replaces model FG in opaque interior regions with original source pixels, eliminating dark fringing.

## Logs

Logs are written to `~/.config/corridorkey/logs/`. Check there for detailed error messages when a clip fails.

## Reset everything

To delete all config, models, and logs and start fresh:

```shell
ck reset
```

Then run `ck init` to set up again.
