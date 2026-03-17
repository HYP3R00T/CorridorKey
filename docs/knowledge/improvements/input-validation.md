# Input Validation

CorridorKey validates clip inputs before loading the inference engine. Validation runs in two tiers: instant checks that cost milliseconds, and sample decode checks that cost a few seconds.

## Why Tiered Validation

Validating every frame upfront on a large video would take as long as processing it. The solution is to validate fast and early, then validate a small sample to catch the most common problems.

Tier 1 and Tier 2 together take 2-3 seconds regardless of video length and catch the vast majority of problems before the engine loads.

## Tier 1 - Instant Checks

These run before anything else and cost milliseconds.

- Input asset path exists and is readable.
- Alpha asset path exists and is readable (if a mask folder is provided).
- Output directory can be created.
- Enough free disk space for the expected output.
- GPU VRAM meets the minimum requirement (if CUDA is available).
- Mask file count matches frame count (directory listing only, no file reading).

If any Tier 1 check fails, processing stops immediately with a clear error message. The engine is never loaded.

## Tier 2 - Sample Decode

These run after Tier 1 passes and cost a few seconds.

- Decodes the first frame, the last frame, and one random middle frame.
- Verifies all three have consistent resolution.
- Warns if dtypes are inconsistent across the sample.

Video files are either valid or they are not. Partial corruption is rare and usually affects a contiguous block, not random frames. Sampling three frames catches the common cases without reading the full sequence.

## Tier 3 - Not Yet Implemented

Tier 3 would validate each frame as it enters the processing queue. If a frame fails to decode, it would be logged and skipped rather than crashing the job. This is not currently implemented.

## ValidationResult

`validate_job_inputs` returns a `ValidationResult` with three fields:

- `ok` - true if all checks passed and the job can proceed.
- `errors` - fatal problems that must be resolved before processing.
- `warnings` - non-fatal issues the user should be aware of.

Errors stop processing. Warnings are printed but do not stop processing.

## VRAM Threshold

The default minimum VRAM requirement is 6.0 GB. This is a conservative floor for the 2048-resolution model. The actual requirement depends on `optimization_mode`:

- `speed` mode requires more VRAM (full 2048x2048 refiner pass).
- `lowvram` mode can run on less (tiled refiner).

The threshold can be overridden by passing `min_vram_gb` to `validate_job_inputs`.

## Source Code

- Validation function: `validate_job_inputs` in [corridorkey/validators.py](https://github.com/edenaion/CorridorKey/blob/main/packages/corridorkey/src/corridorkey/validators.py)
- Called from: `process` command in [corridorkey-cli/commands/process.py](https://github.com/edenaion/CorridorKey/blob/main/packages/corridorkey-cli/src/corridorkey_cli/commands/process.py)

## Related Documents

- [Configuration](../configuration/index.md) - Parameters that affect validation thresholds.
- [Optimization modes](optimization-modes.md) - How VRAM requirements vary by mode.
