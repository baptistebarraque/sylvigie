"""Forest mask and forest type from IGN maps.

The mask is the union of BD Forêt V2 (mapped 2006-2019, carries species) and
the IGN Masque Forêt (2023 orthophotos, recent but no species): each one
recovers clear-cuts the other misses. These functions never see clear-cut
labels, so the mask cannot leak the answers into the evaluation.
"""

import logging

import geopandas as gpd
import numpy as np
import pandas as pd

from sylvigie.aoi import Grid
from sylvigie.raster import rasterize_on_grid

log = logging.getLogger(__name__)

# Forest type codes (uint8). Keep the detail; group when computing metrics.
NON_FOREST = 0
OAK = 1
BEECH = 2
OTHER_BROADLEAF = 3
DOUGLAS_FIR = 4
FIR_SPRUCE = 5
OTHER_CONIFER = 6
MIXED = 7
NO_COVER = 8  # forest land without trees when mapped, usually a recent cut
UNKNOWN = 9  # forest in the Masque Forêt only: species not known

FOREST_TYPE_NAMES = {
    NON_FOREST: "non-forest",
    OAK: "oak",
    BEECH: "beech",
    OTHER_BROADLEAF: "other broadleaf",
    DOUGLAS_FIR: "Douglas fir",
    FIR_SPRUCE: "fir and spruce",
    OTHER_CONIFER: "other conifer",
    MIXED: "mixed",
    NO_COVER: "no tree cover",
    UNKNOWN: "unknown species",
}
FOREST_GROUPS = {
    "broadleaf": (OAK, BEECH, OTHER_BROADLEAF),
    "conifer": (DOUGLAS_FIR, FIR_SPRUCE, OTHER_CONIFER),
    "mixed": (MIXED,),
    "no tree cover": (NO_COVER,),
    "unknown": (UNKNOWN,),
}

# BD Forêt V2 `essence` values; NR goes with "Forêt fermée sans couvert
# arboré", NC with heath and grassland.
ESSENCE_TO_TYPE = {
    "Chênes décidus": OAK,
    "Chênes sempervirents": OAK,
    "Hêtre": BEECH,
    "Feuillus": OTHER_BROADLEAF,
    "Châtaignier": OTHER_BROADLEAF,
    "Robinier": OTHER_BROADLEAF,
    "Peuplier": OTHER_BROADLEAF,
    "Douglas": DOUGLAS_FIR,
    "Sapin, épicéa": FIR_SPRUCE,
    "Conifères": OTHER_CONIFER,
    "Mélèze": OTHER_CONIFER,
    "Pin sylvestre": OTHER_CONIFER,
    "Pin laricio, pin noir": OTHER_CONIFER,
    "Pin maritime": OTHER_CONIFER,
    "Pin d'Alep": OTHER_CONIFER,
    "Pin à crochets, pin cembro": OTHER_CONIFER,
    "Pin autre": OTHER_CONIFER,
    "Pins mélangés": OTHER_CONIFER,
    "Mixte": MIXED,
    "NR": NO_COVER,
    "NC": NON_FOREST,
}
# Fallback on the coarser `tfv_g11` attribute for an unexpected essence.
TFV_G11_TO_TYPE = {
    "Forêt fermée feuillus": OTHER_BROADLEAF,
    "Forêt ouverte feuillus": OTHER_BROADLEAF,
    "Peupleraie": OTHER_BROADLEAF,
    "Forêt fermée conifères": OTHER_CONIFER,
    "Forêt ouverte conifères": OTHER_CONIFER,
    "Forêt fermée mixte": MIXED,
    "Forêt ouverte mixte": MIXED,
    "Forêt fermée sans couvert arboré": NO_COVER,
}


def bdforet_types(bdforet: gpd.GeoDataFrame) -> pd.Series:
    """Forest type code of each BD Forêt V2 polygon."""
    by_essence = bdforet["essence"].map(ESSENCE_TO_TYPE)
    unknown = sorted(set(bdforet.loc[by_essence.isna(), "essence"].dropna()))
    if unknown:
        log.warning("Unexpected BD Forêt essences, typed from tfv_g11: %s", unknown)
    by_tfv = bdforet["tfv_g11"].map(TFV_G11_TO_TYPE)
    return by_essence.fillna(by_tfv).fillna(NON_FOREST).astype(np.uint8)


def forest_layers(
    bdforet: gpd.GeoDataFrame, masque: gpd.GeoDataFrame, grid: Grid
) -> tuple[np.ndarray, np.ndarray]:
    """Forest mask (bool) and forest type (uint8) on `grid`."""
    types = rasterize_on_grid(
        bdforet.assign(forest_type=bdforet_types(bdforet)), grid, values="forest_type"
    )
    in_masque = rasterize_on_grid(masque, grid).astype(bool)
    mask = (types != NON_FOREST) | in_masque
    types[in_masque & (types == NON_FOREST)] = UNKNOWN
    return mask, types
