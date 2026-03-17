# Pipeline

CorridorKey processes footage through six sequential stages. Each stage has a defined input contract and a defined output contract. What happens inside a stage is an implementation detail. The contracts are the stable interface.

## The Six Stages

| Stage | Name | Package | What it does |
|---|---|---|---|
| 1 | [Load frame](load-frame.md) | corridorkey | Reads one image frame and its alpha hint mask from disk. Normalises both to float32 sRGB. |
| 2 | [Generate masks](generate-masks.md) | corridorkey | Generates alpha hint masks for a frame sequence using an external generator. Not yet implemented. |
| 3 | [Preprocess](preprocess.md) | corridorkey-core | Resizes, normalises, and stacks image and mask into a model-ready tensor. |
| 4 | [Infer](infer.md) | corridorkey-core | Runs the model forward pass. Produces raw alpha and foreground predictions. |
| 5 | [Postprocess](postprocess.md) | corridorkey-core | Despeckles, despills, composites, and applies source passthrough. |
| 6 | [Write outputs](write-outputs.md) | corridorkey | Writes all enabled output images for one processed frame to disk. |

## Package Boundary

Stages split across two packages based on one rule: does the stage touch the filesystem?

Stages 1, 2, and 6 touch the filesystem. They live in `corridorkey`.

Stages 3, 4, and 5 are pure compute. They take arrays and tensors in and return arrays and tensors out. They live in `corridorkey-core` and have no filesystem dependency.

This boundary means the core package can be tested in isolation with synthetic arrays, and the application layer can change how frames are loaded or written without touching inference logic.

## Data Flow

A single frame moves through the pipeline in this order:

1. Stage 1 reads the frame and mask from disk. Returns a `FrameData`.
2. Stage 2 is skipped if masks already exist on disk.
3. Stage 3 takes the image and mask arrays from `FrameData` and returns a `PreprocessedTensor`.
4. Stage 4 takes the `PreprocessedTensor` and returns a `RawPrediction`.
5. Stage 5 takes the `RawPrediction` and the original source image and returns a `ProcessedFrame`.
6. Stage 6 takes the `ProcessedFrame` and writes files to disk.

See [Contracts](../contracts/index.md) for the full definition of each data structure.

## Related Documents

- [Contracts](../contracts/index.md) - Data structures between stages.
- [Configuration](../configuration/index.md) - Parameters for each stage.
- [Improvements](../improvements/index.md) - Known gaps and future work per stage.
