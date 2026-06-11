# Commands

## ck [clips_dir]

Scan a clips directory, prompt for engine settings, and process all clips. This is the default command — running `ck` without a subcommand invokes the wizard.

```shell
ck /path/to/clips          # interactive wizard
ck /path/to/clips --yes    # non-interactive, use config defaults
ck                         # prompts for clips directory
```

**Arguments:**

| Argument | Description |
|---|---|
| `clips_dir` | Path to the clips directory (optional — wizard prompts if omitted) |

**Options:**

| Option | Description |
|---|---|
| `--yes`, `-y` | Skip all prompts and use config defaults. Requires `clips_dir`. |

**What the wizard does:**

1. Displays the current resolved config with source attribution.
2. Prompts for a clips directory (if not provided as an argument).
3. Shows the engine settings panel and asks you to pick a preset or enter values manually.
4. Confirms the selected values.
5. Runs the pipeline and shows live progress.

**Engine presets:**

| Preset | refiner_mode | model_precision | img_size |
|---|---|---|---|
| `full_frame` | full_frame | float16 | 1024 |
| `balanced` | auto | auto | 1536 |
| `quality` | full_frame | bfloat16 | 2048 |
| `max_quality` | full_frame | float32 | 2048 |
| `tiled` | tiled | float16 | 1024 |

Select `manual` to set each value individually.

---

## ck init

One-time setup. Runs an environment health check, creates the config file if absent, and offers to download the inference model.

```shell
ck init
```

**Health checks:**

| Check | What it verifies |
|---|---|
| Python >= 3.13 | Python version |
| compute device | GPU backend, vendor, and available VRAM |
| config file | Presence of `~/.config/corridorkey/corridorkey.yaml` |
| inference model | Presence of `~/.config/corridorkey/models/CorridorKey_v1.0.pth` |
| platform | OS and architecture |

If the model is not found, `ck init` offers to download it with a progress bar. The download can be skipped — the model path is shown so you can place it manually.

---

## ck config

Show the resolved configuration with source attribution. Each field shows its current value and where it came from (defaults, global config file, project config file, or environment variable).

```shell
ck config           # display only
ck config --write   # write to ~/.config/corridorkey/corridorkey.yaml
```

**Options:**

| Option | Description |
|---|---|
| `--write` | Write the resolved config to the global config file |

Config is resolved from (lowest to highest priority):

1. Built-in defaults
2. `~/.config/corridorkey/corridorkey.yaml`
3. `./corridorkey.yaml` (current working directory)
4. `CK_*` environment variables

---

## ck reset

Delete `~/.config/corridorkey` — removes the config file, downloaded models, and logs.

```shell
ck reset        # prompts for confirmation
ck reset --yes  # skip confirmation
```

**Options:**

| Option | Description |
|---|---|
| `--yes` | Skip the confirmation prompt |

After resetting, run `ck init` to set up again.

This operation is irreversible. The inference model will need to be downloaded again.
