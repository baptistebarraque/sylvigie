"""Zone configuration: pydantic models loaded from a YAML file."""

import datetime as dt
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Frozen(BaseModel):
    # Unknown keys are errors: a typo in the YAML must not be silently ignored.
    model_config = ConfigDict(extra="forbid", frozen=True)


class FileSource(_Frozen):
    url: str
    filename: str
    md5: str = Field(pattern=r"^[0-9a-f]{32}$")


class SufosatSources(_Frozen):
    dates: FileSource  # clear-cut date per pixel, YYDDD
    prob: FileSource  # detection probability per pixel, %


class PnrSource(_Frozen):
    id_mnhn: str


class Period(_Frozen):
    reference_year: int  # stable year the changes are measured against
    detection_years: list[int] = Field(min_length=1)  # first is the calibration year
    # Imagery is loaded past the last detection year: a cut made late in the
    # year is often only visible after the winter clouds.
    load_until: dt.date

    @model_validator(mode="after")
    def _detection_after_reference(self) -> "Period":
        if min(self.detection_years) <= self.reference_year:
            raise ValueError("detection_years must come after reference_year")
        if self.load_until < dt.date(max(self.detection_years), 12, 31):
            raise ValueError("load_until must cover the last detection year")
        return self

    @property
    def years(self) -> list[int]:
        return [self.reference_year, *self.detection_years]

    @property
    def load_from(self) -> dt.date:
        return dt.date(self.reference_year, 1, 1)


class S2Params(_Frozen):
    """Sentinel-2 L2A ingestion (`sylvigie ingest`)."""

    collection: str
    # Reflectance bands kept in the cube; indices are computed from them.
    bands: list[str] = Field(min_length=1)
    # Tile-level cloud cover is only a loose pre-filter: the clear fraction
    # that matters is measured on the window's forest pixels.
    max_tile_cloud: float = Field(ge=0, le=100)
    min_clear_fraction: float = Field(ge=0, le=1)
    # Scene classification (SCL) classes kept as clear: 4 vegetation,
    # 5 bare soil, 6 water. Everything else (clouds, shadows, snow,
    # saturated, no data) is masked, then the mask is grown by
    # `mask_dilation_px` pixels to catch cloud edges.
    scl_valid: list[int] = Field(min_length=1)
    mask_dilation_px: int = Field(ge=0)


class SelectionParams(_Frozen):
    window_km: int = Field(gt=0)
    step_km: int = Field(gt=0)
    min_cuts_per_year: int = Field(ge=0)
    min_object_ha: float = Field(ge=0)
    min_pnr_fraction: float = Field(ge=0, le=1)
    max_overlap: float = Field(ge=0, le=1)
    n_candidates: int = Field(gt=0)

    @model_validator(mode="after")
    def _window_is_whole_number_of_steps(self) -> "SelectionParams":
        if self.window_km % self.step_km:
            raise ValueError("window_km must be a multiple of step_km")
        return self


class GridParams(_Frozen):
    # Processed around each window but never evaluated, so that clear-cuts
    # crossing the edge stay whole and filters behave the same at the edge.
    margin_m: int = Field(ge=0)


Bbox = tuple[float, float, float, float]  # xmin, ymin, xmax, ymax


def _check_bbox(name: str, bbox: Bbox) -> None:
    xmin, ymin, xmax, ymax = bbox
    if xmin >= xmax or ymin >= ymax:
        raise ValueError(f"windows.{name} must be [xmin, ymin, xmax, ymax]")


class Windows(_Frozen):
    """Pilot windows, in the zone CRS.

    `dev` is where thresholds and models are developed and tuned; `eval` is
    where headline metrics are measured. They must not overlap, otherwise
    tuning would leak into the evaluation area.
    """

    dev: Bbox
    eval: Bbox

    @model_validator(mode="after")
    def _ordered_and_disjoint(self) -> "Windows":
        _check_bbox("dev", self.dev)
        _check_bbox("eval", self.eval)
        a, b = self.dev, self.eval
        if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
            raise ValueError("windows.dev and windows.eval must not overlap")
        return self


class ZoneConfig(_Frozen):
    name: str
    crs: str
    resolution: float = Field(gt=0)
    period: Period
    sufosat: SufosatSources
    pnr: PnrSource
    selection: SelectionParams
    grid: GridParams
    s2: S2Params
    windows: Windows | None = None  # filled in once the windows are chosen

    @model_validator(mode="after")
    def _windows_on_pixel_grid(self) -> "ZoneConfig":
        if self.windows is not None:
            for name, bbox in self.windows:
                if any(v % self.resolution for v in bbox):
                    raise ValueError(
                        f"windows.{name} corners must be multiples of resolution"
                    )
        return self

    @model_validator(mode="after")
    def _step_is_whole_number_of_pixels(self) -> "ZoneConfig":
        step_m = self.selection.step_km * 1000
        if step_m % self.resolution:
            raise ValueError("selection.step_km must be a multiple of resolution")
        return self

    @model_validator(mode="after")
    def _margin_keeps_20m_alignment(self) -> "ZoneConfig":
        # Sentinel-2 20 m bands: grid edges must stay on multiples of 2 pixels.
        if self.grid.margin_m % (2 * self.resolution):
            raise ValueError("grid.margin_m must be a multiple of 2 x resolution")
        return self


def load_config(path: Path) -> ZoneConfig:
    with open(path, encoding="utf-8") as f:
        return ZoneConfig.model_validate(yaml.safe_load(f))
