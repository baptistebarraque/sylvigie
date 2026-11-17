# Sylvigie

Satellite monitoring of French forests: detects clear-cuts and dieback from
Sentinel-1/2 time series and delivers alerts on an interactive map.

## Status

Phase 1 (`sylvigie-detect`), roadmap step 2a: choice of the pilot zone. The
CLI lists every pipeline stage; stages not built yet say which roadmap step
will implement them.

## Data sources and licences

- Clear-cut reference: SUFOSAT v3.0.1, Mermoz et al. (2024), IEEE JSTARS,
  doi:10.1109/JSTARS.2024.3429012, [Zenodo record 17253008](https://zenodo.org/records/17253008).
  The record's metadata says CC BY 4.0 but its description says CC BY-NC 4.0:
  treated as non-commercial.
- Forest map (BD Forêt V2) and park boundaries (PatriNat): IGN Géoplateforme.

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

Pick a pilot window (roadmap step 2a): download SUFOSAT and the park
boundary, rank candidate windows and write a review map and table under
`data/processed/<zone>/zone/`:

```sh
uv run sylvigie suggest-zone --config configs/morvan.yaml
```

Copy the chosen candidate's bbox into the config's `bbox` field.

## Development

```sh
uv run pytest                                # tests (synthetic data, no network)
uv run ruff check . && uv run ruff format .  # lint and format
```

## Data

Nothing under `data/` is committed. Every file there must be reproducible
from a CLI command and a zone config in `configs/`.
