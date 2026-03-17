# Improvements

Design decisions, trade-offs, and known gaps in the current implementation.

Each document covers one area: what is implemented, what is not, and why the current approach was chosen.

## Documents in This Section

- [Optimization modes](optimization-modes.md) - How the CNN refiner tiling strategy is selected and why.
- [Precision modes](precision-modes.md) - Floating point format selection and the trade-offs between fp16, bf16, and fp32.
- [Source passthrough](source-passthrough.md) - Why original source pixels produce better interior colour than the model's fg prediction.
- [Input validation](input-validation.md) - Tiered validation before the engine loads.

## Known Gaps

The following improvements are documented in the pipeline contract but not yet implemented.

### Letterboxing and Encoder Tiling

The current resize strategy squishes frames to 2048x2048 regardless of aspect ratio. Letterboxing (padding the shorter dimension to preserve aspect ratio) and encoder tiling (processing high-resolution frames as overlapping 2048x2048 tiles) are the most impactful remaining quality improvements for non-square and high-resolution footage.

The CNN refiner already tiles at 512x512 with cosine blending. The same approach needs to be extended to the encoder.

### Temporal Consistency

The model processes each frame independently. Small variations in input between frames cause small variations in output, which accumulates into visible flicker in the matte. Using SAM2 as the mask generator is the strongest solution - consistent input masks produce consistent output mattes. Temporal smoothing (blending each frame's alpha with its neighbours) is a lower-complexity fallback.

### Prefetch Buffer and Pinned Memory

The GPU idles while the CPU prepares each frame. A prefetch buffer with CPU worker threads and pinned memory transfers would keep the GPU running continuously. This is the single biggest throughput improvement available without changing the model.

### Bounded Queue for Long Videos

There is no explicit memory management for long videos. A bounded queue with a configurable RAM budget would prevent out-of-memory failures on large jobs.

### Checkpointing

If processing crashes midway through a long video, the run must restart from the beginning. Writing output frames as they complete and checking which frames already exist before starting would allow resuming interrupted jobs.

### Video Reassembly

The pipeline writes frame sequences only. Optional video export for the comp preview output would be the most useful addition - it is the one output people want to scrub through in a player.
