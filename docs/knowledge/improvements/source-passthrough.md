# Source Passthrough

Source passthrough replaces the model's foreground prediction with original source pixels in fully opaque interior regions. Only the edge transition band uses the model's fg prediction.

## The Problem It Solves

The model predicts foreground colour for every pixel. At edges - where green screen separation, hair strands, and semi-transparency matter - the model's prediction is the right choice. In the solid interior of the subject (face, body, clothing), the model's prediction can introduce subtle colour shifts compared to the original source.

The original source pixels are always the ground truth for interior regions. There is no reason to use a model prediction where the subject is fully opaque and the original pixel is available.

## How It Works

1. The alpha matte is eroded inward by `edge_erode_px` pixels. This defines the interior region - pixels far enough from any edge that they are definitely fully opaque.
2. The eroded mask is blurred by `edge_blur_px` pixels to create a smooth transition zone between interior and edge.
3. The blend mask is used to mix source pixels and model fg: interior pixels (blend mask near 1.0) take source pixels, edge pixels (blend mask near 0.0) take model fg.

The result is that the model's fg prediction is only used where it matters - at edges and in semi-transparent regions. The interior retains full source fidelity.

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `source_passthrough` | false | Enable or disable the feature. |
| `edge_erode_px` | 3 | Pixels to erode the interior mask inward. Acts as a safety buffer to avoid using source pixels where green spill might still contaminate the edge region. |
| `edge_blur_px` | 7 | Gaussian blur radius for the transition seam. Larger values produce a softer, less visible boundary between source and model fg. |

## When to Use It

Source passthrough is most beneficial when:

- The subject has complex colour in the interior (skin tones, detailed clothing, fine patterns).
- The model's fg prediction introduces visible colour shifts in solid areas.
- Maximum colour fidelity in the interior is required.

It is less critical when:

- The subject is uniformly coloured and the model's prediction is accurate.
- The footage has significant green spill that extends into the interior - in that case, the model's despilled prediction may actually be better than the raw source.

## Relationship to the MLX Backend

The MLX backend applies a similar blending operation automatically because the MLX model's colour fidelity in opaque interior regions is lower than the Torch model. On the Torch path, source passthrough is opt-in via the `source_passthrough` parameter.

## Related Documents

- [Postprocess](../pipeline/postprocess.md) - Stage 5, where source passthrough runs.
- [PostprocessParams contract](../contracts/postprocess-params.md)
- [Configuration](../configuration/index.md) - The `source_passthrough`, `edge_erode_px`, and `edge_blur_px` parameters.
