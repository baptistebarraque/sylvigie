"""Self-contained interactive web maps (Leaflet) over IGN basemaps.

The page embeds its vector data as GeoJSON and loads Leaflet from a CDN and
basemap tiles from the IGN Géoplateforme WMTS, so it opens from disk in any
browser without a server.
"""

import json
from pathlib import Path

import geopandas as gpd

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
# Bright, distinct hues that stay readable over dark forest on orthophotos.
YEAR_COLORS = ["#ffb000", "#ff3fa4", "#3fd0ff", "#9cff57"]

_TEMPLATE = """<!doctype html>
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

const park = L.geoJSON(DATA.park, {
  style: { color: "#ffffff", weight: 2, dashArray: "6 4", fill: false },
}).addTo(map);

const fmt = (v, d = 0) => Number(v).toLocaleString("en", { maximumFractionDigits: d });
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
      offset: [0, 0],
    });
  },
}).addTo(map);

const overlays = { "Park boundary": park, "Candidate windows": windows };
let legend = "";
DATA.years.forEach((year, i) => {
  const color = DATA.colors[i];
  const layer = L.geoJSON(DATA.cuts, {
    filter: f => f.properties.year === year,
    style: { color: color, weight: 1.5, fillColor: color, fillOpacity: 0.25 },
    onEachFeature: (f, l) => l.bindPopup(
      `SUFOSAT clear-cut ${year}<br>${fmt(f.properties.area_ha, 2)} ha`
    ),
  }).addTo(map);
  overlays[`SUFOSAT clear-cuts ${year}`] = layer;
  legend += `<div><span class="swatch" style="border-color:${color};` +
            `background:${color}40"></span>` +
            `Clear-cuts ${year} (&ge; ${DATA.min_ha} ha)</div>`;
});
L.control.layers(basemaps, overlays, { collapsed: false }).addTo(map);

const info = L.control({ position: "bottomleft" });
info.onAdd = () => {
  const div = L.DomUtil.create("div", "panel");
  div.innerHTML = `<h1>${DATA.title}</h1>` + legend +
    `<div><span class="swatch" style="border-color:#fff;background:#0004"></span>` +
    `R: representative window (click for stats)</div>` +
    `<div><span class="swatch" style="border-color:#fff;border-style:dashed;` +
    `background:#0004"></span>D: densest window</div>` +
    `<div class="muted">Yearly Express orthos may be blank where not yet flown.</div>`;
  return div;
};
info.addTo(map);

map.fitBounds(windows.getBounds().pad(0.05));
</script>
</body>
</html>
"""


def _geojson(gdf: gpd.GeoDataFrame) -> dict:
    return json.loads(gdf.to_crs("EPSG:4326").to_json(drop_id=True))


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
        "basemaps": [
            {"label": label, "url": WMTS_URL.format(layer=layer, fmt=fmt)}
            for label, layer, fmt in BASEMAPS
        ],
        "park": _geojson(park[["geometry"]]),
        "candidates": _geojson(candidates),
        "cuts": _geojson(cuts.assign(area_ha=cuts["area_ha"].round(2))),
        "years": years,
        "colors": YEAR_COLORS[: len(years)],
        "min_ha": min_area_ha,
    }
    # "</" inside the inline script would close the <script> element early.
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    html = (
        _TEMPLATE.replace("__DATA__", payload)
        .replace("__LEAFLET__", LEAFLET_VERSION)
        .replace("__TITLE__", data["title"])
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path
