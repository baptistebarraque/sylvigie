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


class ZoneConfig(_Frozen):
    name: str
    crs: str
    resolution: float = Field(gt=0)
    sufosat: SufosatSource
    pnr: PnrSource
    selection: SelectionParams
    bbox: tuple[float, float, float, float] | None = None

    @model_validator(mode="after")
    def _bbox_is_ordered(self) -> "ZoneConfig":
        if self.bbox is not None:
            xmin, ymin, xmax, ymax = self.bbox
            if xmin >= xmax or ymin >= ymax:
                raise ValueError("bbox must be [xmin, ymin, xmax, ymax]")
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
