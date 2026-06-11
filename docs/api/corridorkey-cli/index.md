# corridorkey-cli

The `ck` command-line interface for CorridorKey. Provides the `ck` command for processing clips from a terminal.

## Install

```shell
pip install corridorkey-cli
```

This installs the `ck` command and the `corridorkey` core library as a dependency.

## Commands

| Command | Description |
|---|---|
| [`ck [clips_dir]`](commands.md#ck-clips_dir) | Scan, configure, and process clips (default wizard) |
| [`ck init`](commands.md#ck-init) | One-time setup: health check, config file, model download |
| [`ck config`](commands.md#ck-config) | Show resolved configuration with source attribution |
| [`ck config --write`](commands.md#ck-config) | Write config to `~/.config/corridorkey/corridorkey.yaml` |
| [`ck reset`](commands.md#ck-reset) | Delete `~/.config/corridorkey` (config, models, logs) |

## Entry point

The package exposes a single entry point:

```python
from corridorkey_cli import main
main()
```

`main()` is the `ck` console script target. It wraps the Typer app and handles `KeyboardInterrupt` cleanly.
