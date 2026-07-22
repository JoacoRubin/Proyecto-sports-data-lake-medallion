# =============================================================================
# tests/test_contracts.py — Verifica que los contratos ATRAPAN datos corruptos.
#
# Un contrato que nunca falla no protege de nada. Estos tests alimentan datos
# inválidos a propósito y exigen que la validación los rechace.
# =============================================================================

import pandas as pd
import pandera.pandas as pa
import pytest

from quality.contracts import (
    validar_gold_posiciones,
    validar_silver_partidos,
    verificar_silver_no_vacio_si_habia_finalizados,
)


def _fila_gold(liga, id_equipo, equipo, PJ, PG, PE, PP, GF, GC):
    return {
        "liga": liga, "id_equipo": id_equipo, "equipo": equipo,
        "PJ": PJ, "PG": PG, "PE": PE, "PP": PP, "GF": GF, "GC": GC,
        "DG": GF - GC, "Pts": PG * 3 + PE,
    }


def _fila_silver(id_evento, resultado="Local", gl=2, gv=0):
    return {
        "id_evento": id_evento, "liga": "Liga", "goles_local": gl,
        "goles_visitante": gv, "resultado": resultado,
        "diferencia_goles": abs(gl - gv), "es_goleada": abs(gl - gv) > 2,
    }


# --- Datos válidos pasan ---------------------------------------------------
def test_gold_valido_pasa():
    df = pd.DataFrame([_fila_gold("Liga", "1", "River", 2, 1, 1, 0, 3, 1)])
    validar_gold_posiciones(df)  # no debe lanzar


def test_silver_valido_pasa():
    df = pd.DataFrame([_fila_silver("1"), _fila_silver("2", "Empate", 1, 1)])
    validar_silver_partidos(df)  # no debe lanzar


# --- Datos corruptos son rechazados ----------------------------------------
def test_gold_rechaza_pts_inconsistente():
    """Pts que no cumple PG*3 + PE debe reventar el contrato."""
    fila = _fila_gold("Liga", "1", "River", 2, 1, 1, 0, 3, 1)
    fila["Pts"] = 99  # invariante roto
    with pytest.raises(pa.errors.SchemaErrors):
        validar_gold_posiciones(pd.DataFrame([fila]))


def test_gold_rechaza_equipo_duplicado_en_liga():
    """El mismo id_equipo dos veces en la misma liga es inválido."""
    df = pd.DataFrame([
        _fila_gold("Liga", "1", "River", 1, 0, 1, 0, 1, 1),
        _fila_gold("Liga", "1", "River", 1, 1, 0, 0, 2, 0),
    ])
    with pytest.raises(pa.errors.SchemaErrors):
        validar_gold_posiciones(df)


def test_gold_rechaza_goles_negativos():
    fila = _fila_gold("Liga", "1", "River", 1, 1, 0, 0, -3, 1)
    with pytest.raises(pa.errors.SchemaErrors):
        validar_gold_posiciones(pd.DataFrame([fila]))


def test_silver_rechaza_resultado_invalido():
    df = pd.DataFrame([_fila_silver("1", resultado="Ganó")])  # no está en el enum
    with pytest.raises(pa.errors.SchemaErrors):
        validar_silver_partidos(df)


def test_silver_rechaza_id_evento_duplicado():
    df = pd.DataFrame([_fila_silver("1"), _fila_silver("1")])
    with pytest.raises(pa.errors.SchemaErrors):
        validar_silver_partidos(df)


# --- Check cross-layer: el caso 'FT' que motivó todo -----------------------
def test_cross_layer_detecta_silver_vacio_con_finalizados():
    """Si bronze tenía finalizados y silver quedó vacío, debe gritar."""
    df_bronze = pd.DataFrame([
        {"estado": "FT"}, {"estado": "Match Finished"}, {"estado": "Not Started"},
    ])
    df_silver_vacio = pd.DataFrame()
    with pytest.raises(ValueError, match="silver quedó vacío"):
        verificar_silver_no_vacio_si_habia_finalizados(df_bronze, df_silver_vacio)


def test_cross_layer_ok_cuando_silver_tiene_datos():
    df_bronze = pd.DataFrame([{"estado": "FT"}])
    df_silver = pd.DataFrame([_fila_silver("1")])
    verificar_silver_no_vacio_si_habia_finalizados(df_bronze, df_silver)  # no lanza
