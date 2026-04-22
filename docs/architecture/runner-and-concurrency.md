# Runner and Concurrency

## The assembly line

Stages 3-6 (preprocess, inference, postprocess, write) run as a threaded assembly line. Three worker types run concurrently:

```
_MultiClipPreprocessWorker
    -> preprocess_queue (bounded)
        -> _InferenceWorker[device:0]  -+
        -> _InferenceWorker[device:1]  -+-> inference_queue (bounded)
        -> _InferenceWorker[device:N]  -+
                                            -> _MultiClipPostWriteWorker
```

**`_MultiClipPreprocessWorker`** — one thread. Reads frames from disk, preprocesses them, and pushes `_FrameWork` items onto the preprocess queue. Applies RAM throttling: if available RAM drops below a headroom threshold, it pauses before reading the next frame.

**`_InferenceWorker`** — one thread per device. Pulls `_FrameWork` items from the preprocess queue, runs the model, and pushes `_InferenceWork` items onto the inference queue. Each worker owns its model instance and CUDA stream.

**`_MultiClipPostWriteWorker`** — one thread. Pulls `_InferenceWork` items from the inference queue, postprocesses them in a thread pool (one thread per GPU, minimum 4), and writes outputs to disk.

## Bounded queues

Both inter-stage queues are bounded. The preprocess queue depth defaults to `max(2, n_gpus * 2)` and the inference queue depth to the same. Bounded queues prevent the preprocess worker from reading the entire clip into memory before inference starts.

## Multi-clip batching

When multiple clips are processed in a single `run()` call, all clips share the same queue pair. The preprocess worker feeds frames from all clips in order. Inference workers pull from the shared queue without waiting for clip boundaries, so a fast GPU is never idle between short clips.

## Cancellation

`Engine.cancel()` sets a `threading.Event`. All workers check this event before processing each frame. When set, workers drain their current item and stop. The Engine raises `JobCancelledError` for the current clip and reports it in `JobStats.clips_cancelled`.

Frames already written are not removed. A cancelled run can be resumed by running again — already-complete clips are skipped automatically.

## Model cache

For single-device runs, the process-level model cache (`ModelCache`) reuses a loaded model across multiple `engine.run()` calls. The cache key is a hash of the `InferenceConfig`. If the config matches the cached model, it is returned without reloading.

For multi-device runs, each device loads its own model independently. The cache is not used for multi-device runs.

## DeferredTransfer

After inference, the alpha and foreground tensors are on the GPU. `DeferredTransfer` starts an async copy to CPU memory using a dedicated CUDA stream, overlapping the GPU-to-CPU transfer with the next inference step. The postwrite worker calls `transfer.resolve()` to wait for the copy to complete before postprocessing.

This overlap is the primary source of throughput improvement on CUDA — the GPU is running inference on frame N+1 while frame N is being transferred to CPU and written to disk.
