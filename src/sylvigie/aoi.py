"""Area of interest geometry and pixel grids."""

import math
from dataclasses import dataclass

from affine import Affine
from odc.geo.geobox import GeoBox
from rasterio.crs import CRS


@dataclass(frozen=True)
class Grid:
    """A north-up pixel grid: CRS, affine transform and shape."""

    crs: CRS
    transform: Affine
    width: int
    height: int

    @classmethod
    def from_bounds(
        cls,
        bounds: tuple[float, float, float, float],
        crs: str | CRS,
        resolution: float,
        anchor: float | None = None,
    ) -> "Grid":
        """Smallest grid covering `bounds` whose edges are multiples of `anchor`.

        Sentinel-2 tiles start on multiples of 20 m in UTM, so an anchor that is
        a multiple of 20 m puts every pixel edge on the 10 m and 20 m band grids.
        """
        anchor = resolution if anchor is None else anchor
        if not math.isclose(anchor / resolution, round(anchor / resolution)):
            raise ValueError("anchor must be a multiple of resolution")
        xmin = math.floor(bounds[0] / anchor) * anchor
        ymin = math.floor(bounds[1] / anchor) * anchor
        xmax = math.ceil(bounds[2] / anchor) * anchor
        ymax = math.ceil(bounds[3] / anchor) * anchor
        return cls(
            crs=CRS.from_user_input(crs),
            transform=Affine(resolution, 0.0, xmin, 0.0, -resolution, ymax),
            width=round((xmax - xmin) / resolution),
            height=round((ymax - ymin) / resolution),
        )

    @property
    def resolution(self) -> float:
        return self.transform.a

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        xmin, ymax = self.transform.c, self.transform.f
        return (
            xmin,
            ymax - self.height * self.resolution,
            xmin + self.width * self.resolution,
            ymax,
        )

    def crop(
        self, bounds: tuple[float, float, float, float]
    ) -> tuple[tuple[slice, slice], "Grid"]:
        """Array slices and sub-grid for the pixels covering `bounds`."""
        xmin, _, _, ymax = self.bounds
        res = self.resolution
        col0 = max(0, math.floor((bounds[0] - xmin) / res))
        col1 = min(self.width, math.ceil((bounds[2] - xmin) / res))
        row0 = max(0, math.floor((ymax - bounds[3]) / res))
        row1 = min(self.height, math.ceil((ymax - bounds[1]) / res))
        if col0 >= col1 or row0 >= row1:
            raise ValueError("bounds do not overlap the grid")
        sub = Grid(
            crs=self.crs,
            transform=self.transform @ Affine.translation(col0, row0),
            width=col1 - col0,
            height=row1 - row0,
        )
        return (slice(row0, row1), slice(col0, col1)), sub

    def to_geobox(self) -> GeoBox:
        """The same grid as an odc-geo GeoBox, the form odc-stac loads onto."""
        return GeoBox(self.shape, self.transform, self.crs)

    def coarsen(self, factor: int) -> "Grid":
        """Grid of `factor` x `factor` pixel blocks; the shape must divide evenly."""
        if self.width % factor or self.height % factor:
            raise ValueError("grid shape is not a multiple of factor")
        return Grid(
            crs=self.crs,
            transform=self.transform @ Affine.scale(factor),
            width=self.width // factor,
            height=self.height // factor,
        )
