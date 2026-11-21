"""Satellite imagery catalogues (STAC)."""

import datetime as dt

import geopandas as gpd
import pystac
from pystac_client import Client
from pystac_client.stac_api_io import StacApiIO
from shapely.geometry import shape
from urllib3.util.retry import Retry

PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
S2_COLLECTION = "sentinel-2-l2a"

# Retry transient server errors with exponential backoff (1, 2, 4, ... s).
_RETRY = Retry(
    total=6,
    backoff_factor=1.0,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET", "POST"],
)


def open_catalog(url: str = PC_STAC_URL) -> Client:
    return Client.open(url, stac_io=StacApiIO(max_retries=_RETRY, timeout=120))


def search_s2(
    bbox_lonlat: tuple[float, float, float, float],
    start: dt.date,
    end: dt.date,
    max_tile_cloud: float,
    collection: str = S2_COLLECTION,
) -> list[pystac.Item]:
    """Sentinel-2 L2A items over `bbox_lonlat` between `start` and `end` (inclusive).

    `eo:cloud_cover` is measured over the whole 110 km tile, so it is only a
    loose pre-filter; the clear fraction over the window is measured later.
    """
    search = open_catalog().search(
        collections=[collection],
        bbox=bbox_lonlat,
        datetime=f"{start.isoformat()}/{end.isoformat()}",
        query={"eo:cloud_cover": {"lt": max_tile_cloud}},
        limit=500,
    )
    return list(search.items())


def s2_footprints(
    bbox_lonlat: tuple[float, float, float, float], datetime: str
) -> gpd.GeoDataFrame:
    """Data footprints of Sentinel-2 L2A items: one row per item.

    Columns: `tile` (MGRS), `orbit` (relative orbit number), `geometry`
    (EPSG:4326). Searching needs no URL signing; reading pixels does.
    """
    search = open_catalog().search(
        collections=[S2_COLLECTION], bbox=bbox_lonlat, datetime=datetime
    )
    rows = [
        {
            "tile": item.properties["s2:mgrs_tile"],
            "orbit": item.properties["sat:relative_orbit"],
            "geometry": shape(item.geometry),
        }
        for item in search.items()
    ]
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
