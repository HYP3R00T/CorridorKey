# Adding a Stage

This guide walks through adding a new pipeline stage. Follow the existing pattern — every stage has the same structure.

## 1. Create the stage folder

```shell
mkdir packages/corridorkey/src/corridorkey/stages/my_stage
touch packages/corridorkey/src/corridorkey/stages/my_stage/__init__.py
touch packages/corridorkey/src/corridorkey/stages/my_stage/contracts.py
touch packages/corridorkey/src/corridorkey/stages/my_stage/orchestrator.py
```

## 2. Define the contracts

In `contracts.py`, define the stage's input and output types as frozen Pydantic models or frozen dataclasses.

```python
# contracts.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class MyStageResult:
    """Output contract of my_stage."""
    data: np.ndarray
    stem: str
```

Keep contracts minimal — only the fields downstream stages actually need.

## 3. Write the orchestrator

In `orchestrator.py`, write the stage function. It takes the previous stage's output contract and a config, and returns the new contract.

```python
# orchestrator.py
from __future__ import annotations
import logging
from corridorkey.stages.my_stage.contracts import MyStageResult

logger = logging.getLogger(__name__)

def my_stage(input_data, config) -> MyStageResult:
    """One-line description of what this stage does.

    Args:
        input_data: Output of the previous stage.
        config: Stage configuration.

    Returns:
        MyStageResult with processed data.

    Raises:
        MyStageError: If processing fails.
    """
    ...
```

## 4. Add a config dataclass

If the stage needs configuration, add a config dataclass in the stage folder and a corresponding settings model in `infra/config/`.

```python
# my_stage/config.py
from dataclasses import dataclass

@dataclass
class MyStageConfig:
    strength: float = 1.0
```

```python
# infra/config/my_stage.py
from pydantic import BaseModel, Field

class MyStageSettings(BaseModel):
    strength: float = Field(default=1.0, ge=0.0, le=1.0)
```

Add the settings model to `CorridorKeyConfig` and add a `to_my_stage_config()` bridge method.

## 5. Add an error type

Add a typed exception to `errors.py`:

```python
class MyStageError(CorridorKeyError):
    """Raised when my_stage fails."""
    def __init__(self, frame_index: int, detail: str) -> None:
        self.frame_index = frame_index
        super().__init__(f"my_stage failed at frame {frame_index}: {detail}")
```

Export it from `corridorkey/__init__.py` and add it to `__all__`.

## 6. Export from the stage `__init__.py`

```python
# stages/my_stage/__init__.py
from corridorkey.stages.my_stage.contracts import MyStageResult
from corridorkey.stages.my_stage.orchestrator import my_stage

__all__ = ["MyStageResult", "my_stage"]
```

## 7. Write tests

Create `packages/corridorkey/tests/unit/stages/my_stage/` and add tests for the orchestrator and any pure helper functions.

```shell
mkdir packages/corridorkey/tests/unit/stages/my_stage
touch packages/corridorkey/tests/unit/stages/my_stage/__init__.py
touch packages/corridorkey/tests/unit/stages/my_stage/test_orchestrator.py
```

## 8. Add docs

Add a page to `docs/api/corridorkey/stages/my-stage.md` and register it in `zensical.toml`.

## Checklist

- `contracts.py` with frozen input/output types
- `orchestrator.py` with the stage function and docstring
- Config dataclass and settings model (if needed)
- Error type in `errors.py` and exported from `__init__.py`
- Exported from stage `__init__.py`
- Unit tests
- Docs page registered in `zensical.toml`
