"""Ancillary data: SUFOSAT labels and IGN Géoplateforme vector layers (WFS)."""

import hashlib
import logging
import time
from collections.abc import Callable
from pathlib import Path

import geopandas as gpd
import requests
from pyproj import CRS

from sylvigie.config import FileSource

log = logging.getLogger(__name__)

WFS_URL = "https://data.geopf.fr/wfs/ows"
PNR_LAYER = "patrinat_pnr:pnr"
BDFORET_V2_LAYER = "LANDCOVER.FORESTINVENTORY.V2:formation_vegetale"
# U-Net forest mask on the departmental BD ORTHO (2023 vintage over the Morvan).
MASQUE_FORET_LAYER = "IGNF_MASQUE-FORET.2021-2023:masque_foret"
WFS_PAGE_SIZE = 5000  # server maximum ("CountDefault" in GetCapabilities)
TIMEOUT_S = 300
WFS_ATTEMPTS = 6
WFS_FIRST_WAIT_S = 5.0  # doubled after each failed attempt
# When overloaded, the Géoplateforme answers HTTP 400 with a database pool
# error ("Unable to obtain connection"): worth retrying, unlike a real 400.
_TRANSIENT_WFS_ERRORS = ("Unable to obtain connection", "Timeout waiting")

_sleep = time.sleep  # replaced in tests


def _is_transient(response: requests.Response) -> bool:
    if response.status_code >= 500:
        return True
    return any(marker in response.text for marker in _TRANSIENT_WFS_ERRORS)


def _get_wfs_page(session: requests.Session, params: dict) -> dict:
    wait = WFS_FIRST_WAIT_S
    for attempt in range(1, WFS_ATTEMPTS + 1):
        try:
            response = session.get(WFS_URL, params=params, timeout=TIMEOUT_S)
        except (requests.ConnectionError, requests.Timeout) as error:
            reason = repr(error)
        else:
            if response.ok:
                try:
                    return response.json()
                except ValueError:
                    # Seen under load: HTTP 200 with an empty or truncated body.
                    reason = "HTTP 200 with a body that is not JSON"
            elif _is_transient(response):
                reason = f"HTTP {response.status_code}"
            else:
                response.raise_for_status()
        if attempt == WFS_ATTEMPTS:
            raise RuntimeError(f"WFS still failing after {attempt} attempts: {reason}")
        log.warning(
            "WFS attempt %d failed (%s), retrying in %.0f s", attempt, reason, wait
        )
        _sleep(wait)
        wait *= 2
    raise AssertionError("unreachable")


def md5sum(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(
    url: str, dest: Path, md5: str, session: requests.Session | None = None
) -> Path:
    """Download `url` to `dest` unless a file with the expected md5 is there.

    Writes to a `.part` file first so an interrupted download never looks
    complete, and refuses a file whose checksum does not match.
    """
    if dest.exists() and md5sum(dest) == md5:
        log.info("Using cached %s", dest)
        return dest
    session = session or requests.Session()
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    log.info("Downloading %s", url)
    with session.get(url, stream=True, timeout=TIMEOUT_S) as response:
        response.raise_for_status()
        with open(part, "wb") as f:
            for chunk in response.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    if (got := md5sum(part)) != md5:
        part.unlink()
        raise ValueError(f"md5 mismatch for {url}: expected {md5}, got {got}")
    part.replace(dest)
    return dest


def download_sufosat(source: FileSource, raw_dir: Path) -> Path:
    return download_file(source.url, raw_dir / "sufosat" / source.filename, source.md5)


def cached_gpkg(path: Path, fetch: Callable[[], gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
    """Read `path` if it exists, else call `fetch` and save its result there.

    Delete the file to refresh it: WFS layers can change on the server.
    """
    if path.exists():
        log.info("Using cached %s", path)
        return gpd.read_file(path)
    gdf = fetch()
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(path, driver="GPKG")
    return gdf


def fetch_wfs(
    layer: str,
    crs: str,
    sort_by: str,
    *,
    bbox: tuple[float, float, float, float] | None = None,
    cql_filter: str | None = None,
    properties: list[str] | None = None,
    session: requests.Session | None = None,
    page_size: int = WFS_PAGE_SIZE,
) -> gpd.GeoDataFrame:
    """Fetch every feature of a WFS 2.0 layer, page by page, as GeoJSON.

    `bbox` is expressed in `crs`, which is also the output CRS. Paging needs a
    stable order, hence the mandatory `sort_by` attribute.
    """
    if bbox is not None and cql_filter is not None:
        raise ValueError("the server rejects BBOX and CQL_FILTER together")
    session = session or requests.Session()
    params: dict[str, str | int] = {
        "SERVICE": "WFS",
        "VERSION": "2.0.0",
        "REQUEST": "GetFeature",
        "TYPENAMES": layer,
        "OUTPUTFORMAT": "application/json",
        "SRSNAME": crs,
        "SORTBY": sort_by,
        "COUNT": page_size,
    }
    if bbox is not None:
        params["BBOX"] = ",".join(map(str, bbox)) + f",{crs}"
    if cql_filter is not None:
        params["CQL_FILTER"] = cql_filter
    if properties is not None:
        params["PROPERTYNAME"] = ",".join([*properties, "geom"])

    features: list[dict] = []
    while True:
        params["STARTINDEX"] = len(features)
        page = _get_wfs_page(session, params)
        features.extend(page["features"])
        log.info("%s: %d features fetched", layer, len(features))
        if len(page["features"]) < page_size:
            break

    expected = page.get("numberMatched")
    if isinstance(expected, int) and expected != len(features):
        raise RuntimeError(
            f"{layer}: got {len(features)} features, expected {expected}"
        )
    returned_crs = page.get("crs", {}).get("properties", {}).get("name")
    if returned_crs is None or CRS.from_user_input(returned_crs) != CRS.from_user_input(
        crs
    ):
        raise RuntimeError(f"{layer}: asked for {crs}, server returned {returned_crs}")
    return gpd.GeoDataFrame.from_features(features, crs=crs)


def fetch_pnr(id_mnhn: str, crs: str = "EPSG:2154") -> gpd.GeoDataFrame:
    """Boundary of one Parc naturel régional, by its MNHN identifier."""
    pnr = fetch_wfs(
        PNR_LAYER, crs, sort_by="id_mnhn", cql_filter=f"id_mnhn='{id_mnhn}'"
    )
    if len(pnr) != 1:
        raise ValueError(f"expected one PNR with id_mnhn={id_mnhn}, got {len(pnr)}")
    return pnr


def fetch_bdforet_v2(
    bbox: tuple[float, float, float, float], crs: str = "EPSG:2154"
) -> gpd.GeoDataFrame:
    """BD Forêt V2 polygons intersecting `bbox` (expressed in `crs`)."""
    return fetch_wfs(
        BDFORET_V2_LAYER,
        crs,
        sort_by="id",
        bbox=bbox,
        properties=["id", "code_tfv", "tfv_g11", "essence"],
    )


def fetch_masque_foret(
    bbox: tuple[float, float, float, float], crs: str = "EPSG:2154"
) -> gpd.GeoDataFrame:
    """IGN Masque Forêt 2021-2023 polygons intersecting `bbox` (in `crs`).

    `millesime` is the year of the orthophotos it was derived from.
    """
    return fetch_wfs(
        MASQUE_FORET_LAYER,
        crs,
        sort_by="id",
        bbox=bbox,
        properties=["id", "nature", "usage", "code_dpt", "millesime"],
    )
