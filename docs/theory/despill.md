# Despill

## What green spill is

When a subject is filmed in front of a green screen, green light reflects off the screen and onto the subject. This contamination — called spill — appears as a green tint on the edges of the subject, particularly on hair, skin, and light-coloured clothing.

Spill is a physical phenomenon. No matter how well the green screen is lit, some reflected green light will reach the subject. The amount depends on the distance between the subject and the screen, the screen's reflectivity, and the lighting setup.

## How despill works

Despill suppresses the green channel in pixels where green is dominant relative to red and blue. The suppression is applied selectively — only in regions where the alpha matte indicates the subject is near the background boundary, where spill is most visible.

The `despill_strength` setting controls how aggressively the green channel is reduced:

- `0.0` — no despill. The foreground colour is unchanged.
- `0.5` — moderate suppression (default). Removes visible spill without affecting the subject's natural colours.
- `1.0` — full suppression. Removes all green dominance. Can produce a magenta tint on subjects with naturally green tones.

## Tuning despill

Start at the default (`0.5`) and adjust based on the footage:

- If the subject has a visible green tint on edges, increase `despill_strength`.
- If the subject's natural colours look shifted toward magenta, decrease it.
- For subjects with green clothing or props, lower values preserve the natural colour better.

Despill is applied after the alpha matte is finalised, so it does not affect the matte quality.
