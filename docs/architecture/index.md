# Architecture

How CorridorKey is built internally. This section is for contributors and maintainers.

| Document | What it covers |
|---|---|
| [Pipeline Overview](pipeline-overview.md) | The six stages and how they connect |
| [Stage Contracts](stage-contracts.md) | How stages communicate through immutable data contracts |
| [Device and Backends](device-and-backends.md) | Device resolution, the backend protocol, multi-GPU dispatch |
| [Runner and Concurrency](runner-and-concurrency.md) | The threaded assembly line, queues, and model cache |
