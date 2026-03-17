# Knowledge

Theory, design decisions, and internals for the CorridorKey pipeline.

This section is for anyone who wants to understand how CorridorKey works, not just how to use it. It covers the six pipeline stages, the data contracts between them, the configuration system, and the reasoning behind key implementation decisions.

No prior knowledge of the codebase is required. Code references link to source files where relevant.

## Documents in This Section

- [Pipeline](pipeline/index.md) - The six stages that transform a raw frame into compositing-ready outputs.
- [Contracts](contracts/index.md) - The data structures that flow between stages.
- [Configuration](configuration/index.md) - Every parameter, its default, and when to change it.
- [Improvements](improvements/index.md) - Trade-offs, future work, and design decisions.
