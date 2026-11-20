"""Zone configuration: pydantic models loaded from a YAML file."""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Frozen(BaseModel):
    # Unknown keys are errors: a typo in the YAML must not be silently ignored.
    model_config = ConfigDict(extra="forbid", frozen=True)


class SufosatSource(_Frozen):
    url: str
    filename: str
    md5: str = Field(pattern=r"^[0-9a-f]{32}$")


class PnrSource(_Frozen):
    id_mnhn: str


class SelectionParams(_Frozen):
    years: list[int] = Field(min_length=1)
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
    sufosat: SufosatSource
    pnr: PnrSource
    selection: SelectionParams
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


def load_config(path: Path) -> ZoneConfig:
    with open(path, encoding="utf-8") as f:
        return ZoneConfig.model_validate(yaml.safe_load(f))
