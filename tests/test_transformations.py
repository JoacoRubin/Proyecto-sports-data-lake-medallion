# =============================================================================
# tests/test_transformations.py — Tests de las transformaciones silver (T1-T4).
#
# Las transformaciones son funciones puras (DataFrame -> DataFrame), sin efectos
# secundarios ni I/O: el caso ideal para tests unitarios rápidos y deterministas.
# =============================================================================

import pandas as pd

from silver.transformations import (
    agregar_columnas_derivadas,
    convertir_tipos_y_fechas,
    deduplicar_partidos,
    manejar_nulos_partidos,
)


def test_deduplicar_conserva_la_extraccion_mas_reciente():
    """T1 — ante el mismo id_evento, gana la fecha_extraccion más nueva."""
    df = pd.DataFrame([
        {"id_evento": "10", "goles_local": 1, "fecha_extraccion": "2022-01-01"},
        {"id_evento": "10", "goles_local": 3, "fecha_extraccion": "2022-01-02"},
        {"id_evento": "11", "goles_local": 0, "fecha_extraccion": "2022-01-01"},
    ])
    out = deduplicar_partidos(df)
    assert len(out) == 2
    fila_10 = out[out["id_evento"] == "10"].iloc[0]
    assert fila_10["goles_local"] == 3  # el registro más reciente


def test_manejar_nulos_filtra_no_finalizados_y_sin_resultado():
    """T2 — descarta partidos sin goles y los que no están finalizados.

    La API de TheSportsDB marca los partidos jugados como 'FT' (Full Time),
    no como 'Match Finished'. El filtro debe reconocer ambos formatos.
    """
    df = pd.DataFrame([
        {"id_evento": "1", "goles_local": 1, "goles_visitante": 0,
         "estado": "FT", "estadio": "Monumental", "temporada": "2022"},
        {"id_evento": "2", "goles_local": 2, "goles_visitante": 2,
         "estado": "Match Finished", "estadio": "Bombonera", "temporada": "2022"},
        {"id_evento": "3", "goles_local": None, "goles_visitante": None,
         "estado": "Not Started", "estadio": None, "temporada": None},
    ])
    out = manejar_nulos_partidos(df)
    assert len(out) == 2  # FT y Match Finished se conservan; Not Started se descarta
    assert set(out["id_evento"]) == {"1", "2"}


def test_convierte_fecha_a_datetime_y_goles_a_int():
    """T3 — fecha_partido pasa a datetime64 y los goles a int."""
    df = pd.DataFrame([
        {"fecha_partido": "2022-05-15", "goles_local": 2, "goles_visitante": 1},
    ])
    out = convertir_tipos_y_fechas(df)
    assert pd.api.types.is_datetime64_any_dtype(out["fecha_partido"])
    assert out.iloc[0]["goles_local"] == 2
    assert out["goles_local"].dtype == "int64"


def test_columnas_derivadas_resultado_diferencia_y_goleada():
    """T4 — resultado, diferencia_goles y es_goleada (dif > 2)."""
    df = pd.DataFrame([
        {"fecha_partido": pd.Timestamp("2022-05-15"), "goles_local": 4, "goles_visitante": 0},
        {"fecha_partido": pd.Timestamp("2022-05-16"), "goles_local": 1, "goles_visitante": 1},
        {"fecha_partido": pd.Timestamp("2022-05-17"), "goles_local": 0, "goles_visitante": 2},
    ])
    out = agregar_columnas_derivadas(df)
    assert list(out["resultado"]) == ["Local", "Empate", "Visitante"]
    assert list(out["diferencia_goles"]) == [4, 0, 2]
    assert list(out["es_goleada"]) == [True, False, False]
    assert list(out["anio_partido"]) == [2022, 2022, 2022]
