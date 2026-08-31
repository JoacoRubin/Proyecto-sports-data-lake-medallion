# =============================================================================
# tests/test_ml_leakage.py — UNA SOLA definicion de que es leakage.
#
# EL PROBLEMA QUE RESUELVE
#
# La regla "estas columnas describen el partido ya jugado" estaba escrita dos
# veces: en ml/features.py (para descartarlas al construir features) y en
# quality/contracts.py (para rechazarlas al validar). Dos listas de la misma
# regla de negocio, y ya habian divergido.
#
# El riesgo es concreto: se agrega una estadistica nueva a bronze, se actualiza
# una lista y no la otra, y queda un agujero por donde entra leakage sin que
# ningun test lo note. Una regla de dominio se define UNA vez.
# =============================================================================

import pandas as pd
import pytest

from ml.leakage import COLUMNAS_POST_PARTIDO, detectar_leakage


def test_la_regla_cubre_todas_las_estadisticas_del_partido():
    """Si football-data trae una estadistica nueva, tiene que estar acá."""
    esperadas = {
        "goles_local", "goles_visitante",
        "goles_local_entretiempo", "goles_visitante_entretiempo",
        "rojas_local", "rojas_visitante",
        "amarillas_local", "amarillas_visitante",
        "faltas_local", "faltas_visitante",
        "tiros_local", "tiros_visitante",
        "tiros_arco_local", "tiros_arco_visitante",
        "corners_local", "corners_visitante",
    }
    faltantes = esperadas - set(COLUMNAS_POST_PARTIDO)
    assert not faltantes, f"estadisticas del partido sin cubrir: {faltantes}"


def test_incluye_los_derivados_del_target():
    """rojas_total y goles_totales salen del marcador: tambien son leakage."""
    assert "rojas_total" in COLUMNAS_POST_PARTIDO
    assert "goles_totales" in COLUMNAS_POST_PARTIDO


def test_no_mezcla_decisiones_de_modelado_con_la_regla():
    """`arbitro` no es leakage: se descarta por no tener senal, que es otra cosa.

    Mezclarlas hacia que la lista no se pudiera reutilizar para validar, porque
    el contrato habria rechazado una columna perfectamente legitima.
    """
    assert "arbitro" not in COLUMNAS_POST_PARTIDO


def test_no_mezcla_columnas_internas_de_trabajo():
    """`_i` y `_par` son andamiaje del calculo; se borran por prefijo."""
    assert not [c for c in COLUMNAS_POST_PARTIDO if c.startswith("_")]


# --- El detector -----------------------------------------------------------
def test_detecta_las_columnas_prohibidas_presentes():
    df = pd.DataFrame([{"liga": "X", "goles_local": 2, "paridad": 0.8}])
    assert detectar_leakage(df) == ["goles_local"]


def test_no_reporta_nada_en_una_matriz_limpia():
    df = pd.DataFrame([{"liga": "X", "paridad": 0.8, "prob_local": 0.4}])
    assert detectar_leakage(df) == []


def test_devuelve_todas_las_encontradas_no_solo_la_primera():
    df = pd.DataFrame([{"goles_local": 1, "rojas_visitante": 0, "liga": "X"}])
    assert set(detectar_leakage(df)) == {"goles_local", "rojas_visitante"}


# --- LA garantia: una sola fuente de verdad --------------------------------
def test_el_contrato_y_el_feature_engineering_usan_la_MISMA_regla():
    """Si estas dos divergen, vuelve el agujero que motivo este modulo."""
    from ml.features import _COLUMNAS_CON_LEAKAGE
    from quality.contracts import _COLUMNAS_PROHIBIDAS_ML

    canonicas = set(COLUMNAS_POST_PARTIDO)
    assert canonicas <= set(_COLUMNAS_CON_LEAKAGE), \
        "ml/features.py no descarta todas las columnas post-partido"
    assert canonicas == set(_COLUMNAS_PROHIBIDAS_ML), \
        "quality/contracts.py no rechaza exactamente las columnas post-partido"


def test_los_dos_modelos_comparten_la_regla():
    """features.py y features_goles.py no pueden tener criterios distintos."""
    from ml.features import _COLUMNAS_CON_LEAKAGE
    from ml.features_goles import _COLUMNAS_A_DESCARTAR

    canonicas = set(COLUMNAS_POST_PARTIDO)
    assert canonicas <= set(_COLUMNAS_CON_LEAKAGE)
    assert canonicas <= set(_COLUMNAS_A_DESCARTAR)
