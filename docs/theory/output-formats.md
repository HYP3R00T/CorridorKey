# Output Formats

## EXR vs PNG

CorridorKey can write outputs as EXR or PNG. The right choice depends on your compositing workflow.

**EXR** is a floating-point format designed for visual effects work. It stores values in linear light, supports values above 1.0 (HDR), and preserves full precision through compositing operations. Use EXR when your compositor expects linear light input.

**PNG** is an integer format. CorridorKey writes 16-bit PNG for the `processed/` output and 8-bit PNG for `comp/`. PNG is sRGB-encoded and clips values above 1.0. Use PNG when you need a widely compatible format or when your compositor expects sRGB input.

## Linear light vs sRGB

Camera footage is typically encoded in sRGB (or a log curve). Compositing operations — colour grading, light wraps, colour corrections — are mathematically correct only in linear light. Working in sRGB introduces errors that are visible as incorrect blending at edges.

CorridorKey's `processed/` output is in linear light when written as EXR. The `comp/` output is always sRGB — it is a preview image for visual review, not for compositing.

## Output directory contents

| Folder | Format | Colour space | Purpose |
|---|---|---|---|
| `alpha/` | PNG or EXR | Linear | Alpha matte — grayscale |
| `fg/` | PNG or EXR | sRGB | Straight foreground colour |
| `processed/` | PNG (16-bit) or EXR (float32) | Linear | Premultiplied RGBA — primary compositor input |
| `comp/` | PNG (8-bit) | sRGB | Checkerboard preview — visual review only |

## EXR compression

When writing EXR, CorridorKey uses `dwaa` compression by default. DWAA is a lossy wavelet codec that gives good compression ratios with fast decode. For lossless EXR, use `zip` or `piz`.

Available codecs: `none`, `rle`, `zips`, `zip`, `piz`, `pxr24`, `dwaa`, `dwab`.

Set `writer.exr_compression` in your config to change the codec.

## Choosing formats for your workflow

For compositing in Nuke, Fusion, or After Effects with a linear workflow:

```yaml
writer:
  alpha_format: exr
  fg_format: exr
  processed_format: exr
  comp_enabled: true   # keep for review
```

For a simpler workflow or when your compositor expects sRGB PNG:

```yaml
writer:
  alpha_format: png
  fg_format: png
  processed_format: png
```
