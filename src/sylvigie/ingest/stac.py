"""Satellite imagery catalogues (STAC)."""

import geopandas as gpd
from pystac_client import Client
from shapely.geometry import shape

PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
S2_COLLECTION = "sentinel-2-l2a"


def s2_footprints(
    bbox_lonlat: tuple[float, float, float, float], datetime: str
) -> gpd.GeoDataFrame:
    """Data footprints of Sentinel-2 L2A items: one row per item.

    Columns: `tile` (MGRS), `orbit` (relative orbit number), `geometry`
    (EPSG:4326). Searching needs no URL signing; reading pixels does.
    """
    search = Client.open(PC_STAC_URL).search(
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
