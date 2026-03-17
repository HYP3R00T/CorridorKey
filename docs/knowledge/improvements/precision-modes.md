# Precision Modes

`precision` controls the floating point format used during the model forward pass. This affects VRAM usage, inference speed, and numerical stability.

## The Four Modes

| Mode | Format | Range | Precision | When to use |
|---|---|---|---|---|
| `auto` (default) | bf16 on Ampere+/Apple Silicon, fp16 otherwise | Varies | Varies | Right choice for most users |
| `fp16` | float16 | 65,504 | ~3 decimal digits | Older GPUs without bf16 support |
| `bf16` | bfloat16 | ~3.4x10^38 | ~2 decimal digits | Ampere+ and Apple Silicon |
| `fp32` | float32 | ~3.4x10^38 | ~7 decimal digits | Debugging, maximum numerical safety |

## Why Precision Matters

A floating point number has two parts: an exponent that controls the range of values it can represent, and a mantissa that controls how many significant digits it carries.

float16 is fast and memory-efficient but has a limited range. Values above 65,504 become infinity. Values too small become zero. In practice this means certain operations - softmax, layer norm, exponentials - can overflow or underflow in float16 and produce NaN.

bfloat16 was developed by Google Brain to address this. It uses the same number of bits as float16 but allocates more to the exponent and fewer to the mantissa. The result is the same value range as float32 with slightly less precision. Neural networks tolerate the precision reduction well because they were trained with noise anyway. The overflow problem that affects float16 largely disappears.

## How Autocast Works

The forward pass uses mixed precision. The framework decides per-operation whether to use the configured precision or fall back to float32, based on numerical stability requirements.

Operations that run in reduced precision: matrix multiplications, convolutions, linear layers.

Operations that always run in float32: softmax, layer norm, batch norm, exponentials, logarithms.

This means `precision` controls the weight format and the default for compute-heavy operations. It does not force every operation into reduced precision.

## TF32

On Ampere+ GPUs, TF32 is enabled automatically for matrix multiplications. TF32 is not a storage format - it is a compute mode that uses 10-bit mantissa math for matrix multiplications while storing values as float32. It is transparent to the user and provides a speed improvement on Ampere+ hardware with no configuration required.

## Auto-Detection

`auto` mode detects the GPU architecture at engine creation time:

- NVIDIA Ampere or newer (compute capability 8.0+) - selects bf16.
- Apple Silicon (MPS or MLX backend) - selects bf16.
- Older NVIDIA GPUs - selects fp16.
- CPU - selects fp32.

## Related Documents

- [Configuration](../configuration/index.md) - The `precision` config parameter.
- [Infer](../pipeline/infer.md) - Stage 4, where precision takes effect.
