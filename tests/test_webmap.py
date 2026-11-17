import json
import re

import geopandas as gpd
from shapely.geometry import box

from sylvigie.viz.webmap import write_zone_webmap

UTM = "EPSG:32631"


def test_write_zone_webmap_embeds_layers(tmp_path):
    park = gpd.GeoDataFrame(geometry=[box(600000, 5200000, 650000, 5260000)], crs=UTM)
    candidates = gpd.GeoDataFrame(
        {"candidate": [1], "cuts_2023": [120], "note": ["</script>"]},
        geometry=[box(610000, 5210000, 635000, 5235000)],
        crs=UTM,
    )
    cuts = gpd.GeoDataFrame(
        {"year": [2023, 2024], "area_ha": [1.234, 0.6]},
        geometry=[box(611000, 5211000, 611100, 5211100)] * 2,
        crs=UTM,
    )

    path = write_zone_webmap(
        park, candidates, cuts, tmp_path / "map.html", "morvan", 0.5
    )

    html = path.read_text(encoding="utf-8")
    payload = re.search(r"const DATA = (.*);\n", html).group(1)
    data = json.loads(payload)
    assert data["years"] == [2023, 2024]
    assert data["min_ha"] == 0.5
    assert len(data["cuts"]["features"]) == 2
    lon, lat = data["candidates"]["features"][0]["geometry"]["coordinates"][0][0]
    assert 3 < lon < 5 and 46 < lat < 48  # reprojected to lon/lat for Leaflet
    assert "</script>" not in payload  # cannot break out of the inline script
    assert "ORTHOIMAGERY.ORTHOPHOTOS" in html
