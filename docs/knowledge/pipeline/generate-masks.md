# Generate Masks

Stage 2 generates alpha hint masks for a frame sequence using an external mask generator. If masks already exist on disk, this stage is skipped entirely.

## Status

Stage 2 is not yet implemented. The stage function exists as a placeholder and raises `NotImplementedError` when called without a generator. The pipeline checks for existing masks before calling this stage and skips it if the mask folder is already populated.

## Purpose

The alpha hint mask tells the model roughly where the foreground is. It is not a precise matte, just a directional hint. The model uses it as a starting point and handles fine detail from there.

Without a mask, the model receives a blank input (all 0.5, meaning everything is unknown). Results are still usable but less accurate, particularly at edges.

## How It Will Work

When implemented, stage 2 will:

1. Check whether the output mask directory is already populated. If yes, skip.
2. Delegate to an external generator that implements the `AlphaGenerator` protocol.
3. The generator writes mask frames to the output directory.
4. Stage 1 reads those mask frames when processing each frame.

The core inference pipeline is unchanged regardless of which generator is used. The generator is a separate concern.

## Planned Generator Options

| Generator | Best for | Temporal consistency |
|---|---|---|
| Chroma key | Clean green screens, controlled lighting | None - frame by frame |
| GVM / VideoMaMa | Difficult footage, uneven screens | Partial |
| SAM2 | Any footage, maximum consistency | High - tracks subject across full video |

SAM2 is the strongest option for temporal consistency. It maintains memory of previous frames and tracks the subject across the full video. Consistent input masks produce consistent output mattes, which reduces flicker in the final result.

Using SAM2 also means the pipeline is not limited to green screen footage. Any footage where a segmentation model can identify the subject becomes a valid input.

## Mask Quality

What makes a good mask:

- A reasonable unknown band (approximately 20-40 pixels wide at 2048 resolution) around subject edges.
- Clean separation between definite foreground and definite background.
- Temporally consistent - the same pixel should not flip between foreground and background across adjacent frames.

What makes a bad mask:

- Unknown band too tight or too loose.
- Noisy - random pixels flipping between foreground and background.
- Wrong threshold - green screen pixels classified as foreground, or subject pixels as background.

## Source Code

- Stage function: `stage_2_generate_masks` in [corridorkey/stages.py](https://github.com/edenaion/CorridorKey/blob/main/packages/corridorkey/src/corridorkey/stages.py)
- Generator protocol: `AlphaGenerator` in [corridorkey/protocols.py](https://github.com/edenaion/CorridorKey/blob/main/packages/corridorkey/src/corridorkey/protocols.py)

## Related Documents

- [Load frame](load-frame.md) - Stage 1, which reads the mask files this stage produces.
- [Temporal consistency](../improvements/index.md) - Why mask quality affects output flicker.
