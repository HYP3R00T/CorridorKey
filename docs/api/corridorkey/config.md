# Config

## Loading config

::: corridorkey.infra.load_config

::: corridorkey.infra.load_config_with_metadata

## CorridorKeyConfig

The single entry point for all pipeline configuration. Load once at startup with `load_config()`, then use the bridge methods to produce stage configs.

::: corridorkey.infra.config.pipeline.CorridorKeyConfig

## Settings models

Each section of the config file maps to a settings model. These are Pydantic models — all fields are validated on construction.

::: corridorkey.infra.config.preprocess.PreprocessSettings

::: corridorkey.infra.config.inference.InferenceSettings

::: corridorkey.infra.config.postprocess.PostprocessSettings

::: corridorkey.infra.config.writer.WriterSettings

## Device utilities

::: corridorkey.infra.resolve_device

::: corridorkey.infra.resolve_devices

::: corridorkey.infra.detect_gpu

::: corridorkey.infra.clear_device_cache

## Config file utilities

These functions are re-exported from `utilityhub_config` and manage the config file on disk.

**`ensure_config_file(config, app_name, format="yaml") -> Path`**
Create the config file at `~/.config/<app_name>/<app_name>.yaml` if it does not exist. Returns the path. Safe to call on every startup.

**`get_config_path(app_name, format="yaml") -> Path`**
Return the expected config file path without creating it.

**`write_config(config, app_name, format="yaml") -> Path`**
Write the current config to disk, overwriting any existing file. Returns the path written.

## Model hub

::: corridorkey.infra.default_checkpoint_path

## Example: loading and overriding config

```python
from corridorkey import load_config

# Load from file + env vars
config = load_config()

# Override specific fields at runtime
from corridorkey.infra.config import CorridorKeyConfig
config = CorridorKeyConfig.model_validate({
    **config.model_dump(),
    "device": "cuda:1",
    "inference": {
        **config.inference.model_dump(),
        "model_precision": "float16",
    },
})
```

## Example: showing config with source attribution

```python
from corridorkey import load_config_with_metadata

config, metadata = load_config_with_metadata()
for field_name in config.model_fields:
    source = metadata.get_source(field_name)
    print(f"{field_name}: {getattr(config, field_name)} (from {source.source})")
```
