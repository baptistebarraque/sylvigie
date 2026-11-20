"""Self-contained interactive web maps (Leaflet) over IGN basemaps.

The page embeds its data (GeoJSON, PNG overlays) and loads Leaflet from a CDN
and basemap tiles from the IGN Géoplateforme WMTS, so it opens from disk in
any browser without a server.
"""

import base64
import io
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
from PIL import Image
from rasterio.warp import transform_bounds

from sylvigie.aoi import Grid
from sylvigie.raster import reproject_array

LEAFLET_VERSION = "1.9.4"
WMTS_URL = (
    "https://data.geopf.fr/wmts?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0"
    "&LAYER={layer}&STYLE=normal&TILEMATRIXSET=PM"
    "&TILEMATRIX={{z}}&TILEROW={{y}}&TILECOL={{x}}&FORMAT={fmt}"
)
# (label, WMTS layer, format). Yearly "Express" orthos may not cover every area.
BASEMAPS = [
    ("Orthophoto IGN (latest)", "ORTHOIMAGERY.ORTHOPHOTOS", "image/jpeg"),
    ("Ortho Express 2025", "ORTHOIMAGERY.ORTHOPHOTOS.ORTHO-EXPRESS.2025", "image/jpeg"),
    ("Ortho Express 2024", "ORTHOIMAGERY.ORTHOPHOTOS.ORTHO-EXPRESS.2024", "image/jpeg"),
    ("Ortho Express 2023", "ORTHOIMAGERY.ORTHOPHOTOS.ORTHO-EXPRESS.2023", "image/jpeg"),
    ("Plan IGN", "GEOGRAPHICALGRIDSYSTEMS.PLANIGNV2", "image/png"),
]
# One colour per calendar year, identical on every map; bright enough to stay
# readable over dark forest on orthophotos.
YEAR_COLORS = {2022: "#3fd0ff", 2023: "#ffb000", 2024: "#ff3fa4", 2025: "#9cff57"}
OTHER_YEAR_COLOR = "#c3c2b7"

_HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@__LEAFLET__/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@__LEAFLET__/dist/leaflet.js"></script>
<style>
  html, body, #map { height: 100%; margin: 0; }
  body { font: 14px/1.4 system-ui, sans-serif; }
  .panel { background: rgba(252, 252, 251, 0.92); color: #0b0b0b; padding: 8px 10px;
           border-radius: 6px; box-shadow: 0 1px 4px rgba(0, 0, 0, 0.3); }
  .panel h1 { font-size: 15px; margin: 0 0 4px; }
  .panel .muted { color: #52514e; font-size: 12px; }
  .panel table td { padding: 0 8px 0 0; }
  .swatch { display: inline-block; width: 14px; height: 10px; margin-right: 6px;
            vertical-align: middle; border: 2px solid; }
  .window-label { background: none; border: none; box-shadow: none; color: #fff;
                  font-weight: 700; font-size: 18px; text-shadow: 0 0 3px #000; }
  table.stats td { padding: 0 6px 0 0; }
</style>
</head>
<body>
<div id="map"></div>
<script>
const DATA = __DATA__;
const map = L.map("map");
const basemaps = {};
DATA.basemaps.forEach((b, i) => {
  basemaps[b.label] = L.tileLayer(b.url, {
    maxZoom: 19, attribution: "&copy; IGN Géoplateforme",
  });
  if (i === 0) basemaps[b.label].addTo(map);
});
const fmt = (v, d = 0) => Number(v).toLocaleString("en", { maximumFractionDigits: d });
const swatch = (color, extra = "") =>
  `<span class="swatch" style="border-color:${color};background:${color}66;${extra}">` +
  `</span>`;
function addPanel(html) {
  const info = L.control({ position: "bottomleft" });
  info.onAdd = () => {
    const div = L.DomUtil.create("div", "panel");
    div.innerHTML = `<h1>${DATA.title}</h1>` + html +
      `<div class="muted">Yearly Express orthos may be blank ` +
      `where not yet flown.</div>`;
    return div;
  };
  info.addTo(map);
}
"""

_TAIL = """
</script>
</body>
</html>
"""

_ZONE_SCRIPT = """
const park = L.geoJSON(DATA.park, {
  style: { color: "#ffffff", weight: 2, dashArray: "6 4", fill: false },
}).addTo(map);

const windows = L.geoJSON(DATA.candidates, {
  style: f => ({
    color: "#ffffff", weight: 3, fill: false,
    dashArray: f.properties.kind === "densest" ? "10 6" : null,
  }),
  onEachFeature: (f, layer) => {
    const p = f.properties;
    const rows = DATA.years.map(y =>
      `<tr><td>Clear-cuts ${y}</td>` +
      `<td>${p["cuts_" + y]} (${fmt(p["cut_ha_" + y])} ha)</td></tr>`
    ).join("");
    layer.bindPopup(
      `<b>Candidate ${p.candidate}</b> (${p.kind})<table class="stats">${rows}` +
      `<tr><td>Cut rate</td><td>${fmt(p.cut_rate_pct, 2)} %/yr</td></tr>` +
      `<tr><td>Conifer share</td><td>${fmt(100 * p.conifer_share)} %</td></tr>` +
      `<tr><td>In park</td><td>${fmt(100 * p.pnr_fraction)} %</td></tr>` +
      `<tr><td>Sentinel-2</td><td>${p.s2_tiles}, ${p.orbit_cover}</td></tr>` +
      `<tr><td>Score</td><td>${fmt(p.score, 3)}</td></tr></table>`
    );
    layer.bindTooltip(String(p.candidate), {
      permanent: true, direction: "center", className: "window-label",
    });
  },
}).addTo(map);

const overlays = { "Park boundary": park, "Candidate windows": windows };
let legend = "";
DATA.years.forEach(year => {
  const color = DATA.year_colors[year];
  const layer = L.geoJSON(DATA.cuts, {
    filter: f => f.properties.year === year,
    style: { color: color, weight: 1.5, fillColor: color, fillOpacity: 0.25 },
    onEachFeature: (f, l) => l.bindPopup(
      `SUFOSAT clear-cut ${year}<br>${fmt(f.properties.area_ha, 2)} ha`
    ),
  }).addTo(map);
  overlays[`SUFOSAT clear-cuts ${year}`] = layer;
  legend += `<div>${swatch(color)}Clear-cuts ${year} (&ge; ${DATA.min_ha} ha)</div>`;
});
L.control.layers(basemaps, overlays, { collapsed: false }).addTo(map);
addPanel(
  legend +
  `<div>${swatch("#ffffff")}R: representative window (click for stats)</div>` +
  `<div>${swatch("#ffffff", "border-style:dashed")}D: densest window</div>`
);
map.fitBounds(windows.getBounds().pad(0.05));
"""

_REFERENCE_SCRIPT = """
const overlays = {};
let legend = "";
DATA.rasters.forEach(r => {
  const layer = L.imageOverlay(r.png, r.bounds, {
    opacity: r.opacity, interactive: false, className: "pixelated",
  });
  if (r.visible) layer.addTo(map);
  overlays[r.label] = layer;
  legend += `<div><b>${r.label}</b></div>` + r.legend.map(
    ([name, color]) => `<div>${swatch(color)}${name}</div>`
  ).join("");
});
const extent = L.geoJSON(DATA.extent, {
  style: { color: "#ffffff", weight: 2, dashArray: "6 4", fill: false },
}).addTo(map);
const evalArea = L.geoJSON(DATA.eval_area, {
  style: { color: "#ffffff", weight: 3, fill: false },
}).addTo(map);
overlays["Window (evaluated)"] = evalArea;
overlays["Margin (processed only)"] = extent;
L.control.layers(basemaps, overlays, { collapsed: false }).addTo(map);
const rows = DATA.coverage.map(
  ([year, share]) => `<tr><td>${year}</td><td>${fmt(100 * share, 1)} %</td></tr>`
).join("");
addPanel(
  legend +
  `<div><b>Clear-cut pixels inside the forest mask</b></div><table>${rows}</table>`
);
map.fitBounds(extent.getBounds());
"""

_PIXELATED_CSS = "  .pixelated { image-rendering: pixelated; }\n</style>"


def _geojson(gdf: gpd.GeoDataFrame) -> dict:
    return json.loads(gdf.to_crs("EPSG:4326").to_json(drop_id=True))


def _basemaps() -> list[dict]:
    return [
        {"label": label, "url": WMTS_URL.format(layer=layer, fmt=fmt)}
        for label, layer, fmt in BASEMAPS
    ]


def year_color(year: int) -> str:
    return YEAR_COLORS.get(year, OTHER_YEAR_COLOR)


def _render(data: dict, script: str, path: Path) -> Path:
    # "</" inside the inline script would close the <script> element early.
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    html = (
        _HEAD.replace("__DATA__", payload)
        .replace("__LEAFLET__", LEAFLET_VERSION)
        .replace("__TITLE__", data["title"])
        .replace("</style>", _PIXELATED_CSS, 1)
        + script
        + _TAIL
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path


def write_zone_webmap(
    park: gpd.GeoDataFrame,
    candidates: gpd.GeoDataFrame,
    cuts: gpd.GeoDataFrame,
    path: Path,
    title: str,
    min_area_ha: float,
) -> Path:
    """Candidate windows and their clear-cuts over IGN orthophotos."""
    years = sorted(int(y) for y in cuts["year"].unique())
    data = {
        "title": f"{title.capitalize()}: candidate pilot windows",
        "basemaps": _basemaps(),
        "park": _geojson(park[["geometry"]]),
        "candidates": _geojson(candidates),
        "cuts": _geojson(cuts.assign(area_ha=cuts["area_ha"].round(2))),
        "years": years,
        "year_colors": {year: year_color(year) for year in years},
        "min_ha": min_area_ha,
    }
    return _render(data, _ZONE_SCRIPT, path)


def categorical_overlay(
    codes: np.ndarray, grid: Grid, colors: dict[int, str]
) -> tuple[str, list[list[float]]]:
    """PNG data URL and Leaflet bounds of a categorical raster.

    The raster is reprojected to Web Mercator first: an image overlay is
    stretched linearly in the map's projection, so a UTM image would drift
    by hundreds of metres across a 25 km window. Codes absent from `colors`
    are transparent.
    """
    mercator, merc_grid = reproject_array(codes.astype(np.uint8), grid, "EPSG:3857")
    palette_index = np.zeros(256, dtype=np.uint8)  # index 0 = transparent
    palette = [0, 0, 0]
    for i, (code, color) in enumerate(colors.items(), start=1):
        palette_index[code] = i
        palette += [int(color[k : k + 2], 16) for k in (1, 3, 5)]
    indexed = np.ascontiguousarray(palette_index[mercator])
    image = Image.frombytes("P", (merc_grid.width, merc_grid.height), indexed.tobytes())
    image.putpalette(palette)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True, transparency=0)
    png = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    west, south, east, north = transform_bounds(
        merc_grid.crs, "EPSG:4326", *merc_grid.bounds
    )
    return png, [[south, west], [north, east]]


def write_reference_webmap(
    title: str,
    rasters: list[dict],
    extent: gpd.GeoDataFrame,
    eval_area: gpd.GeoDataFrame,
    coverage: dict[int, float],
    path: Path,
) -> Path:
    """Reference layers of a window over IGN orthophotos, for visual checks.

    Each raster is a dict with `label`, `codes` (array), `grid`, `colors`
    (code -> hex), `names` (code -> legend label), `opacity`, `visible`.
    """
    overlays = []
    for raster in rasters:
        png, bounds = categorical_overlay(
            raster["codes"], raster["grid"], raster["colors"]
        )
        overlays.append(
            {
                "label": raster["label"],
                "png": png,
                "bounds": bounds,
                "opacity": raster["opacity"],
                "visible": raster["visible"],
                "legend": [
                    [raster["names"][code], color]
                    for code, color in raster["colors"].items()
                ],
            }
        )
    data = {
        "title": title,
        "basemaps": _basemaps(),
        "rasters": overlays,
        "extent": _geojson(extent),
        "eval_area": _geojson(eval_area),
        "coverage": [[year, share] for year, share in coverage.items()],
    }
    return _render(data, _REFERENCE_SCRIPT, path)
