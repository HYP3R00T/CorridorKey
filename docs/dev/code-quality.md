# Code Quality

## Linting and formatting

CorridorKey uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting.

```shell
# Check for lint errors
uv run ruff check

# Fix auto-fixable lint errors
uv run ruff check --fix

# Format code
uv run ruff format

# Check formatting without writing
uv run ruff format --check
```

Ruff configuration is in `ruff.toml` at the workspace root.

## Type checking

```shell
uv run ty check
```

Type checking uses [ty](https://github.com/astral-sh/ty). All public functions and methods must have type annotations.

## Pre-commit hooks

Pre-commit hooks run Ruff and other checks automatically on `git commit`.

```shell
# Install hooks (once)
uv run pre-commit install

# Run all hooks manually
uv run pre-commit run --all-files
```

Hook configuration is in `.pre-commit-config.yaml`.

## Editor config

`.editorconfig` sets indentation (4 spaces), line endings (LF), and trailing whitespace rules. Most editors pick this up automatically.

## Quality gate order

When making changes, run quality gates in this order:

```shell
uv run ruff format          # 1. format
uv run ruff check --fix     # 2. lint
uv run ty check             # 3. type check
uv run pytest               # 4. tests
```

All four must pass before opening a pull request.
