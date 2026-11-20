import geopandas as gpd
import numpy as np
from shapely.geometry import box

from sylvigie.aoi import Grid
from sylvigie.forest import (
    BEECH,
    DOUGLAS_FIR,
    MIXED,
    NO_COVER,
    NON_FOREST,
    OAK,
    OTHER_CONIFER,
    UNKNOWN,
    bdforet_types,
    forest_layers,
)

UTM = "EPSG:32631"
GRID = Grid.from_bounds((0, 0, 400, 100), UTM, 10)  # 10 x 40 pixels


def _bdforet(rows):
    essences, tfvs, geoms = zip(*rows, strict=True)
    return gpd.GeoDataFrame(
        {"essence": essences, "tfv_g11": tfvs}, geometry=list(geoms), crs=UTM
    )


def test_bdforet_types_maps_essences():
    bdforet = _bdforet(
        [
            ("Chênes décidus", "Forêt fermée feuillus", box(0, 0, 1, 1)),
            ("Hêtre", "Forêt fermée feuillus", box(0, 0, 1, 1)),
            ("Douglas", "Forêt fermée conifères", box(0, 0, 1, 1)),
            ("Mixte", "Forêt ouverte mixte", box(0, 0, 1, 1)),
            ("NR", "Forêt fermée sans couvert arboré", box(0, 0, 1, 1)),
            ("NC", "Lande", box(0, 0, 1, 1)),
        ]
    )

    types = bdforet_types(bdforet)

    assert types.tolist() == [OAK, BEECH, DOUGLAS_FIR, MIXED, NO_COVER, NON_FOREST]


def test_bdforet_types_falls_back_on_tfv_g11():
    bdforet = _bdforet([("Cèdre", "Forêt fermée conifères", box(0, 0, 1, 1))])

    assert bdforet_types(bdforet).tolist() == [OTHER_CONIFER]


def test_forest_layers_is_the_union_with_unknown_species():
    bdforet = _bdforet(
        [
            ("Douglas", "Forêt fermée conifères", box(0, 0, 200, 100)),
            ("NC", "Lande", box(200, 0, 300, 100)),
        ]
    )
    # Masque Forêt overlaps the Douglas stand and covers the heath and beyond.
    masque = gpd.GeoDataFrame(geometry=[box(100, 0, 400, 100)], crs=UTM)

    mask, types = forest_layers(bdforet, masque, GRID)

    assert mask.all()
    assert (types[:, :20] == DOUGLAS_FIR).all()  # species kept where known
    assert (types[:, 20:] == UNKNOWN).all()  # masque only: forest, species unknown


def test_forest_layers_without_masque_polygons():
    bdforet = _bdforet([("Hêtre", "Forêt fermée feuillus", box(0, 0, 100, 100))])
    masque = gpd.GeoDataFrame(geometry=[], crs=UTM)

    mask, types = forest_layers(bdforet, masque, GRID)

    assert mask.sum() == 100
    assert (types[mask] == BEECH).all()
    assert (types[~mask] == NON_FOREST).all()
    assert types.dtype == np.uint8
