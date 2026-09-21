# =============================================================================
# services/inference_api/dependencies.py — Qué modelos y qué histórico sirve la API.
#
# El histórico se lee una sola vez por proceso (lru_cache): misma idea que ya
# usa app.py con @st.cache_resource para no releer Delta Lake en cada
# interacción. Los tests reemplazan estas dos funciones con
# `app.dependency_overrides` para no tocar Delta Lake ni el dataset real.
# =============================================================================

from functools import lru_cache

import pandas as pd

from config import DIR_MODELOS, DIR_PARTIDOS_FOOTBALLDATA_BRONZE
from utils.delta import leer_tabla_delta


def get_dir_modelos() -> str:
    return DIR_MODELOS


@lru_cache(maxsize=1)
def _historico_cacheado() -> pd.DataFrame:
    return leer_tabla_delta(DIR_PARTIDOS_FOOTBALLDATA_BRONZE)


def get_historico() -> pd.DataFrame:
    return _historico_cacheado()
