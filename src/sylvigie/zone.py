"""Pilot-zone selection from SUFOSAT clear-cut density.

A square window slides over the regional park on a coarse cell grid. Windows
with enough clear-cuts in every year (calibration and test) and lying mostly
inside the park are kept, then ranked by how closely they resemble the whole
park (annual cut rate, conifer share). Ranking on resemblance rather than on
density avoids picking a hotspot where precision would look better than
anywhere else.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio import features
from rasterio.warp import transform_bounds
from scipy import ndimage
from shapely.geometry import box, shape

from sylvigie.aoi import Grid
from sylvigie.config import ZoneConfig
from sylvigie.ingest.ancillary import download_file, fetch_bdforet_v2, fetch_pnr
from sylvigie.ingest.stac import s2_footprints
from sylvigie.raster import rasterize_on_grid, read_on_grid, write_geotiff

log = logging.getLogger(__name__)

# Forest classes rasterised from the BD Forêt V2 `tfv_g11` attribute.
NON_FOREST, BROADLEAF, CONIFER, MIXED, FOREST_NO_COVER = 0, 1, 2, 3, 4
FOREST_CLASSES = (BROADLEAF, CONIFER, MIXED, FOREST_NO_COVER)
TFV_G11_TO_CLASS = {
    "Forêt fermée feuillus": BROADLEAF,
    "Forêt ouverte feuillus": BROADLEAF,
    "Peupleraie": BROADLEAF,
    "Forêt fermée conifères": CONIFER,
    "Forêt ouverte conifères": CONIFER,
    "Forêt fermée mixte": MIXED,
    "Forêt ouverte mixte": MIXED,
    # Forest land without trees at mapping time: usually a recent cut.
    "Forêt fermée sans couvert arboré": FOREST_NO_COVER,
}

# 8-connectivity: pixels touching by a corner belong to the same clear-cut.
_EIGHT_CONNECTED = np.ones((3, 3), dtype=bool)

# Period used to list Sentinel-2 tiles and orbits over each candidate.
S2_SAMPLE_PERIOD = "2023-06-01/2023-06-30"


def sufosat_years(dates: np.ndarray) -> np.ndarray:
    """Clear-cut year from SUFOSAT YYDDD values; 0 where there is no cut."""
    return np.where(dates > 0, 2000 + dates // 1000, 0).astype(np.int16)


@dataclass(frozen=True)
class CutObjects:
    """Connected clear-cut patches: centroid (row, col) and size in pixels."""

    rows: np.ndarray
    cols: np.ndarray
    sizes: np.ndarray


def find_cut_objects(mask: np.ndarray, min_pixels: int) -> CutObjects:
    """Connected patches of `mask` with at least `min_pixels` pixels."""
    labels, n = ndimage.label(mask, structure=_EIGHT_CONNECTED)
    if n == 0:
        empty = np.empty(0)
        return CutObjects(empty, empty, empty.astype(np.int64))
    rows, cols = np.nonzero(labels)
    ids = labels[rows, cols]
    sizes = np.bincount(ids, minlength=n + 1)[1:]
    row_sums = np.bincount(ids, weights=rows, minlength=n + 1)[1:]
    col_sums = np.bincount(ids, weights=cols, minlength=n + 1)[1:]
    keep = sizes >= min_pixels
    return CutObjects(
        rows=row_sums[keep] / sizes[keep],
        cols=col_sums[keep] / sizes[keep],
        sizes=sizes[keep],
    )


def block_sum(values: np.ndarray, block: int) -> np.ndarray:
    """Sum over non-overlapping `block` x `block` cells."""
    ny, nx = values.shape
    if ny % block or nx % block:
        raise ValueError(f"shape {values.shape} is not a multiple of {block}")
    cells = values.reshape(ny // block, block, nx // block, block)
    return cells.sum(axis=(1, 3), dtype=np.int64)


def count_per_cell(
    rows: np.ndarray, cols: np.ndarray, block: int, cells_shape: tuple[int, int]
) -> np.ndarray:
    """Number of points (pixel coordinates) falling in each cell."""
    counts = np.zeros(cells_shape, dtype=np.int64)
    np.add.at(counts, (rows.astype(int) // block, cols.astype(int) // block), 1)
    return counts


def window_sums(cells: np.ndarray, size: int) -> np.ndarray:
    """Sum over every `size` x `size` window; result[i, j] has top-left cell (i, j).

    Uses a summed-area table, so the cost does not depend on the window size.
    """
    table = np.pad(cells, ((1, 0), (1, 0))).cumsum(axis=0).cumsum(axis=1)
    return (
        table[size:, size:]
        - table[:-size, size:]
        - table[size:, :-size]
        + table[:-size, :-size]
    )


@dataclass(frozen=True)
class CellStats:
    """Pixel counts aggregated on the coarse cell grid."""

    cuts: dict[int, np.ndarray]  # clear-cut objects per year (by centroid)
    cut_px: dict[int, np.ndarray]  # clear-cut pixels per year, any patch size
    forest_px: np.ndarray
    broadleaf_px: np.ndarray
    conifer_px: np.ndarray
    mixed_px: np.ndarray
    pnr_px: np.ndarray
    cell_px: int  # cell side, in pixels


def compute_cell_stats(
    years_px: np.ndarray,
    forest_class: np.ndarray,
    pnr_mask: np.ndarray,
    years: list[int],
    cell_px: int,
    min_object_px: int,
) -> CellStats:
    cells_shape = (years_px.shape[0] // cell_px, years_px.shape[1] // cell_px)
    cuts, cut_px = {}, {}
    for year in years:
        mask = years_px == year
        objects = find_cut_objects(mask, min_object_px)
        cuts[year] = count_per_cell(objects.rows, objects.cols, cell_px, cells_shape)
        cut_px[year] = block_sum(mask, cell_px)
    return CellStats(
        cuts=cuts,
        cut_px=cut_px,
        forest_px=block_sum(np.isin(forest_class, FOREST_CLASSES), cell_px),
        broadleaf_px=block_sum(forest_class == BROADLEAF, cell_px),
        conifer_px=block_sum(forest_class == CONIFER, cell_px),
        mixed_px=block_sum(forest_class == MIXED, cell_px),
        pnr_px=block_sum(pnr_mask, cell_px),
        cell_px=cell_px,
    )


def _conifer_share(broadleaf, conifer, mixed):
    # Mixed stands count as half conifer.
    return (conifer + 0.5 * mixed) / (broadleaf + conifer + mixed)


@dataclass(frozen=True)
class Profile:
    """What a representative window should look like."""

    cut_rate: float  # mean annual clear-cut area / forest area
    conifer_share: float


def area_profile(
    years_px: np.ndarray, forest_class: np.ndarray, mask: np.ndarray, years: list[int]
) -> Profile:
    """Cut rate and conifer share over the pixels where `mask` is true."""
    classes = forest_class[mask]
    forest = np.isin(classes, FOREST_CLASSES).sum()
    cut = np.mean([(years_px[mask] == year).sum() for year in years])
    share = _conifer_share(
        (classes == BROADLEAF).sum(),
        (classes == CONIFER).sum(),
        (classes == MIXED).sum(),
    )
    return Profile(cut_rate=float(cut / forest), conifer_share=float(share))


def rank_windows(
    stats: CellStats,
    years: list[int],
    window_cells: int,
    profile: Profile,
    min_cuts_per_year: int,
    min_pnr_fraction: float,
    pixel_ha: float,
) -> pd.DataFrame:
    """Valid windows, best first. Lower score = closer to `profile`."""
    w = window_cells
    cuts = {y: window_sums(stats.cuts[y], w) for y in years}
    cut_px = {y: window_sums(stats.cut_px[y], w) for y in years}
    forest = window_sums(stats.forest_px, w)
    broadleaf = window_sums(stats.broadleaf_px, w)
    conifer = window_sums(stats.conifer_px, w)
    mixed = window_sums(stats.mixed_px, w)
    pnr_fraction = window_sums(stats.pnr_px, w) / (stats.cell_px * w) ** 2

    with np.errstate(divide="ignore", invalid="ignore"):
        cut_rate = np.mean([cut_px[y] for y in years], axis=0) / forest
        conifer_share = _conifer_share(broadleaf, conifer, mixed)
    score = np.abs(cut_rate / profile.cut_rate - 1) + np.abs(
        conifer_share / profile.conifer_share - 1
    )

    valid = (pnr_fraction >= min_pnr_fraction) & np.isfinite(score)
    for y in years:
        valid &= cuts[y] >= min_cuts_per_year
    rows, cols = np.nonzero(valid)

    table = {"row": rows, "col": cols}
    for y in years:
        table[f"cuts_{y}"] = cuts[y][rows, cols]
        table[f"cut_ha_{y}"] = cut_px[y][rows, cols] * pixel_ha
    table |= {
        "cut_rate_pct": 100 * cut_rate[rows, cols],
        "conifer_share": conifer_share[rows, cols],
        "forest_ha": forest[rows, cols] * pixel_ha,
        "pnr_fraction": pnr_fraction[rows, cols],
        "score": score[rows, cols],
    }
    table["total_cuts"] = sum(table[f"cuts_{y}"] for y in years)
    return pd.DataFrame(table).sort_values("score", ignore_index=True)


def window_overlap(a: tuple[int, int], b: tuple[int, int], size: int) -> float:
    """Shared area fraction of two `size` x `size` windows at cells `a` and `b`."""
    dy = max(0, size - abs(a[0] - b[0]))
    dx = max(0, size - abs(a[1] - b[1]))
    return dy * dx / size**2


def pick_distinct(
    ranked: pd.DataFrame, window_cells: int, max_overlap: float, n: int
) -> pd.DataFrame:
    """Best windows in rank order, skipping those overlapping a chosen one."""
    chosen: list[int] = []
    for i, rc in enumerate(zip(ranked["row"], ranked["col"], strict=True)):
        others = ranked.loc[chosen, ["row", "col"]].itertuples(index=False)
        if all(
            window_overlap(rc, tuple(o), window_cells) <= max_overlap for o in others
        ):
            chosen.append(i)
            if len(chosen) == n:
                break
    return ranked.loc[chosen].reset_index(drop=True)


# Two shortlists: (id prefix, sort column, ascending).
#  - representative: closest to the park profile, where headline metrics are
#    measured because they transfer to the rest of the territory;
#  - densest: most clear-cuts, where thresholds and models are developed
#    because examples are plentiful.
SHORTLISTS = {
    "representative": ("R", "score", True),
    "densest": ("D", "total_cuts", False),
}


def shortlist(
    ranked: pd.DataFrame, window_cells: int, max_overlap: float, n: int
) -> pd.DataFrame:
    """Top `n` distinct windows of each shortlist, ids R1.. and D1.."""
    parts = []
    for kind, (prefix, column, ascending) in SHORTLISTS.items():
        order = ranked.sort_values(
            column, ascending=ascending, kind="stable", ignore_index=True
        )
        picked = pick_distinct(order, window_cells, max_overlap, n)
        picked.insert(
            0, "candidate", [f"{prefix}{i}" for i in range(1, len(picked) + 1)]
        )
        picked.insert(1, "kind", kind)
        parts.append(picked)
    return pd.concat(parts, ignore_index=True)


def window_bounds(
    grid: Grid, row: int, col: int, window_cells: int, cell_px: int
) -> tuple[float, float, float, float]:
    cell_m = cell_px * grid.resolution
    xmin, ymax = grid.bounds[0] + col * cell_m, grid.bounds[3] - row * cell_m
    side = window_cells * cell_m
    return xmin, ymax - side, xmin + side, ymax


def s2_coverage(candidates: gpd.GeoDataFrame, period: str) -> pd.DataFrame:
    """Sentinel-2 tiles over each window and the fraction seen by each orbit.

    Where two relative orbits overlap, a pixel is imaged twice as often, which
    doubles the chances of a cloud-free observation.
    """
    lonlat = candidates.to_crs("EPSG:4326").total_bounds
    footprints = s2_footprints(tuple(lonlat), period).to_crs(candidates.crs)
    orbits = footprints.dissolve(by="orbit").geometry
    rows = []
    for window in candidates.geometry:
        tiles = sorted(footprints.loc[footprints.intersects(window), "tile"].unique())
        cover = {
            orbit: geom.intersection(window).area / window.area
            for orbit, geom in orbits.items()
        }
        rows.append(
            {
                "s2_tiles": " ".join(tiles),
                "orbit_cover": ", ".join(
                    f"R{orbit:03d} {100 * frac:.0f}%"
                    for orbit, frac in cover.items()
                    if frac > 0
                ),
            }
        )
    return pd.DataFrame(rows, index=candidates.index)


def cut_polygons(
    years_px: np.ndarray,
    grid: Grid,
    years: list[int],
    min_area_ha: float,
    bounds: tuple[float, float, float, float],
) -> gpd.GeoDataFrame:
    """Clear-cut patches inside `bounds` as polygons, one row per patch and year.

    Same 8-connectivity and size threshold as the object counts, so the map
    shows exactly the clear-cuts that were counted.
    """
    (rows, cols), sub = grid.crop(bounds)
    records = []
    for year in years:
        mask = years_px[rows, cols] == year
        for geom, _ in features.shapes(
            mask.astype(np.uint8), mask=mask, transform=sub.transform, connectivity=8
        ):
            polygon = shape(geom)
            area_ha = polygon.area / 10_000
            if area_ha >= min_area_ha:
                records.append({"year": year, "area_ha": area_ha, "geometry": polygon})
    if not records:
        return gpd.GeoDataFrame({"year": [], "area_ha": []}, geometry=[], crs=grid.crs)
    return gpd.GeoDataFrame(records, geometry="geometry", crs=grid.crs)


def _cached_gpkg(path: Path, fetch: Callable[[], gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
    if path.exists():
        log.info("Using cached %s", path)
        return gpd.read_file(path)
    gdf = fetch()
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(path, driver="GPKG")
    return gdf


@dataclass(frozen=True)
class ZoneSuggestion:
    candidates: gpd.GeoDataFrame
    n_valid_windows: int
    profile: Profile
    outputs: dict[str, Path]


def suggest_zone(cfg: ZoneConfig, data_dir: Path) -> ZoneSuggestion:
    """Rank candidate pilot windows inside the park and write review outputs."""
    from sylvigie.viz.maps import plot_zone_candidates  # matplotlib is slow to import
    from sylvigie.viz.webmap import write_zone_webmap

    sel = cfg.selection
    raw = data_dir / "raw"
    interim = data_dir / "interim" / cfg.name / "zone"
    processed = data_dir / "processed" / cfg.name / "zone"
    step_m = sel.step_km * 1000
    cell_px = round(step_m / cfg.resolution)
    window_cells = sel.window_km // sel.step_km
    pixel_ha = cfg.resolution**2 / 10_000

    pnr = _cached_gpkg(
        raw / "ign" / f"pnr_{cfg.pnr.id_mnhn}.gpkg", lambda: fetch_pnr(cfg.pnr.id_mnhn)
    ).to_crs(cfg.crs)
    # Cell corners on multiples of step_m, hence on the Sentinel-2 10 m and 20 m grids.
    grid = Grid.from_bounds(tuple(pnr.total_bounds), cfg.crs, cfg.resolution, step_m)
    log.info("Analysis grid: %d x %d pixels", grid.width, grid.height)

    bbox_l93 = transform_bounds(grid.crs, "EPSG:2154", *grid.bounds, densify_pts=21)
    bdforet = _cached_gpkg(
        raw / "ign" / f"bdforet_v2_{cfg.name}_park.gpkg",
        lambda: fetch_bdforet_v2(bbox_l93),
    )
    sufosat = download_file(
        cfg.sufosat.url, raw / "sufosat" / cfg.sufosat.filename, cfg.sufosat.md5
    )

    # Nearest neighbour: dates and classes must never be interpolated.
    dates = read_on_grid(sufosat, grid)
    unknown = set(bdforet["tfv_g11"].dropna()) - set(TFV_G11_TO_CLASS)
    if unknown:
        log.warning("BD Forêt classes treated as non-forest: %s", sorted(unknown))
    bdforet["forest_class"] = (
        bdforet["tfv_g11"].map(TFV_G11_TO_CLASS).fillna(NON_FOREST)
    )
    forest_class = rasterize_on_grid(bdforet, grid, values="forest_class")
    pnr_mask = rasterize_on_grid(pnr, grid).astype(bool)

    years_px = sufosat_years(dates)
    min_object_px = round(sel.min_object_ha / pixel_ha)
    stats = compute_cell_stats(
        years_px, forest_class, pnr_mask, sel.years, cell_px, min_object_px
    )
    profile = area_profile(years_px, forest_class, pnr_mask, sel.years)
    log.info(
        "Park profile: cut rate %.2f %%/yr, conifer share %.0f %%",
        100 * profile.cut_rate,
        100 * profile.conifer_share,
    )
    ranked = rank_windows(
        stats,
        sel.years,
        window_cells,
        profile,
        sel.min_cuts_per_year,
        sel.min_pnr_fraction,
        pixel_ha,
    )
    picked = shortlist(ranked, window_cells, sel.max_overlap, sel.n_candidates)
    geometries = [
        box(*window_bounds(grid, r, c, window_cells, cell_px))
        for r, c in zip(picked["row"], picked["col"], strict=True)
    ]
    candidates = gpd.GeoDataFrame(
        picked.drop(columns=["row", "col"]), geometry=geometries, crs=grid.crs
    )
    if len(candidates):
        candidates = candidates.join(s2_coverage(candidates, S2_SAMPLE_PERIOD))

    # Mean annual clear-cut area per cell, in ha per km2 (cells are step_km wide).
    density = np.mean([stats.cut_px[y] for y in sel.years], axis=0) * pixel_ha
    density /= sel.step_km**2
    density_grid = grid.coarsen(cell_px)
    outputs = {
        "sufosat_dates": write_geotiff(dates, grid, interim / "sufosat_dates.tif", 0),
        "forest_class": write_geotiff(forest_class, grid, interim / "forest_class.tif"),
        "cut_density": write_geotiff(
            density.astype(np.float32), density_grid, processed / "cut_density.tif"
        ),
        "candidates": processed / "candidates.gpkg",
        "cuts": processed / "candidate_cuts.gpkg",
        "map": processed / "candidates.png",
        "webmap": processed / "candidates_map.html",
    }
    candidates.to_file(outputs["candidates"], driver="GPKG")
    if len(candidates):
        cuts = cut_polygons(
            years_px, grid, sel.years, sel.min_object_ha, tuple(candidates.total_bounds)
        )
        cuts.to_file(outputs["cuts"], driver="GPKG")
        write_zone_webmap(
            pnr, candidates, cuts, outputs["webmap"], cfg.name, sel.min_object_ha
        )
    plot_zone_candidates(
        density,
        density_grid,
        pnr,
        candidates,
        outputs["map"],
        title=f"{cfg.name.capitalize()}: candidate pilot windows ({sel.window_km} km)",
        years=sel.years,
    )
    return ZoneSuggestion(candidates, len(ranked), profile, outputs)
