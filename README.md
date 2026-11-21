# Sylvigie

Satellite monitoring of French forests: detects clear-cuts and dieback from
Sentinel-1/2 time series and delivers alerts on an interactive map.

## Status

Phase 1 (`sylvigie-detect`), roadmap step 3: Sentinel-2 time-series cube of
each pilot window. The CLI lists every pipeline stage; stages not built yet
say which roadmap step will implement them.

## Data sources and licences

- Clear-cut reference: SUFOSAT v3.0.1, Mermoz et al. (2024), IEEE JSTARS,
  doi:10.1109/JSTARS.2024.3429012, [Zenodo record 17253008](https://zenodo.org/records/17253008).
  The record's metadata says CC BY 4.0 but its description says CC BY-NC 4.0:
  treated as non-commercial.
- Forest maps (BD Forêt V2, Masque Forêt 2021-2023) and park boundaries
  (PatriNat): IGN Géoplateforme.
- Imagery: Copernicus Sentinel-2 L2A, served by Microsoft Planetary
  Computer. CRSWIR follows the fordead package (INRAE).

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

Pick the pilot windows (roadmap step 2a): download SUFOSAT, the park
boundary and BD Forêt, then shortlist candidate windows and write review
maps (PNG and interactive HTML over IGN orthophotos) under
`data/processed/<zone>/zone/`:

```sh
uv run sylvigie suggest-zone --config configs/morvan.yaml
```

Two shortlists come out: **representative** windows (R1, R2, ...), closest
to the whole park in cut rate and conifer share, and **densest** windows
(D1, D2, ...). Copy the chosen bboxes into the config's `windows` field:
`dev` (where thresholds and models are tuned) and `eval` (where headline
metrics are measured). For the Morvan, `dev` = D1 and `eval` = R1.

Build the reference layers of each window (roadmap step 2b) on a 10 m grid
aligned with Sentinel-2, covering the window plus a margin
(`grid.margin_m`) that is processed but never evaluated:

```sh
uv run sylvigie grid --config configs/morvan.yaml
```

It writes to `data/interim/<zone>/<window>/`:

| File | Content |
| --- | --- |
| `eval_area.tif` | 1 = window (evaluated), 0 = margin |
| `forest_mask.tif` | 1 = forest: union of BD Forêt V2 and IGN Masque Forêt 2021-2023 |
| `forest_type.tif` | species group from BD Forêt V2 (codes in the file's metadata) |
| `sufosat_date.tif` | SUFOSAT clear-cut date, days since 1970-01-01, 0 = none |
| `sufosat_prob.tif` | SUFOSAT clear-cut probability (%), 0 = none |
| `summary.csv` | area and clear-cut area by forest type and year |
| `reference_map.html` | the layers over IGN orthophotos, for visual checks |

Build the Sentinel-2 L2A time series of each window (roadmap step 3) from
Microsoft Planetary Computer. The run is resumable: it keeps a per-day
manifest in `data/raw/s2/<zone>/<window>/` and skips finished days.

```sh
uv run sylvigie ingest --config configs/morvan.yaml --window dev --subset-km 5  # develop
uv run sylvigie ingest --config configs/morvan.yaml                             # both windows
```

1. Search the catalogue for the period (`period`), drop duplicate items
   (ESA reprocessings), group items by solar day.
2. Screen each day on its 20 m scene classification (SCL): keep it when at
   least `s2.min_clear_fraction` of the window's forest pixels are clear.
3. Download the bands of kept days onto the window grid, remove the +1000
   offset of processing baselines >= 04.00 item by item, fuse tiles.
4. Gather the days into `data/interim/<zone>/<window>/s2_cube.zarr`
   (Zarr v3): int16 reflectance x 10 000 per band plus SCL, each at its
   native resolution (`y, x` at 10 m for B04 and B08; `y20, x20` at 20 m for
   B05, B8A, B11, B12 and SCL), chunked to hold the whole time series of
   2.56 x 2.56 km blocks. 20 m bands are resampled to 10 m and the cloud mask
   is applied when reading (`sylvigie.preprocess.indices`), which halves the
   cube on disk and lets the mask change without downloading again.

Plot index time series (NBR, NDVI, CRSWIR) of typical pixels picked from the
step 2b layers: stable conifer, stable broadleaf, Douglas fir and fir/spruce
clear-cuts of the calibration year:

```sh
uv run sylvigie plot-series --config configs/morvan.yaml --window dev --subset-km 5
```

## Development

```sh
uv run pytest                                # tests (synthetic data, no network)
uv run ruff check . && uv run ruff format .  # lint and format
```

## Data

Nothing under `data/` is committed. Every file there must be reproducible
from a CLI command and a zone config in `configs/`.
