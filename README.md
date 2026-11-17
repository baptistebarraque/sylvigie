# Sylvigie

Satellite monitoring of French forests: detects clear-cuts and dieback from
Sentinel-1/2 time series and delivers alerts on an interactive map.

## Status

Phase 1 (`sylvigie-detect`), roadmap step 1: repository skeleton. The CLI
lists every pipeline stage; stages not built yet say which roadmap step will
implement them.

## Setup

Requires [uv](https://docs.astral.sh/uv/). It installs the pinned Python
version (see `.python-version`) and all dependencies:

```sh
uv sync
```

## Usage

```sh
uv run sylvigie --help
```

## Development

```sh
uv run pytest                                # tests (synthetic data, no network)
uv run ruff check . && uv run ruff format .  # lint and format
```

## Data

Nothing under `data/` is committed. Every file there must be reproducible
from a CLI command and a zone config in `configs/`.
