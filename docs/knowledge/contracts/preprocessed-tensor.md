# PreprocessedTensor

Output of stage 3 (preprocess). Input to stage 4 (infer).

Carries the model-ready tensor on the target device, along with metadata needed to upsample results back to source resolution in stage 5.

## Fields

| Field | Type | Description |
|---|---|---|
| `tensor` | `torch.Tensor [1, 4, img_size, img_size]` float32 | Channels 0-2 are ImageNet-normalised image. Channel 3 is the mask. On the target device. |
| `img_size` | int | Square resolution the tensor was prepared at. Matches the model's training resolution (2048). |
| `device` | str | Torch device string the tensor lives on ("cuda", "mps", "cpu"). |
| `source_h` | int | Original frame height in pixels. Carried through for upsampling in stage 5. |
| `source_w` | int | Original frame width in pixels. Carried through for upsampling in stage 5. |

## Notes

The tensor has four channels, not three. The fourth channel is the alpha hint mask, concatenated after the normalised image. The model was trained to expect this four-channel input.

`source_h` and `source_w` are not used by stage 4. They are carried through so stage 5 can upsample the model's predictions back to the original frame dimensions without needing to pass the original frame separately.

## Related Documents

- [Preprocess](../pipeline/preprocess.md) - Stage 3, which produces this contract.
- [Infer](../pipeline/infer.md) - Stage 4, which consumes this contract.
