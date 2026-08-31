# =============================================================================
# tests/test_silver_multifuente.py — Silver une las dos fuentes SIN matching
# de nombres.
#
# LA DECISION
#
# TheSportsDB dice "Manchester United"; football-data dice "Man United". La
# reaccion instintiva es escribir un matcher difuso. Es un pozo: ~160 equipos,
# fusiones y ascensos, y cada falso positivo corrompe una tabla de posiciones
# en silencio.
#
# No hace falta. `fuente` pasa a ser una dimension mas y las posiciones se
# calculan por (fuente, liga). Cada tabla es internamente consistente, y el
# lector elige cual mirar. Mas honesto que un merge que finge una union que no
# fue verificada.
# =============================================================================

import pandas as pd
import pytest

from silver.transformations import procesar_partidos_footballdata, FUENTE_FOOTBALLDATA
from gold.aggregations import calcular_tabla_posiciones


def _partido_fd(id_evento, local, visitante, gl, gv, fecha="2024-05-15"):
    return {
        "id_evento": id_evento, "nombre_evento": f"{local} vs {visitante}",
        "temporada": "2024-2025", "liga": "English Premier League", "id_liga": "FD-E0",
        "equipo_local": local, "id_equipo_local": f"FD-{local.lower()}",
        "equipo_visitante": visitante, "id_equipo_visitante": f"FD-{visitante.lower()}",
        "goles_local": gl, "goles_visitante": gv,
        "fecha_partido": fecha, "hora_partido": "20:00",
        "estadio": None, "estado": "FT",
        "timestamp_extraccion": "2026-01-01T00:00:00+00:00",
        "fecha_extraccion": "2026-01-01",
    }


def _df(filas):
    return pd.DataFrame(filas)


# --- Procesamiento de football-data ----------------------------------------
def test_marca_la_fuente_en_cada_fila():
    """Sin esta columna, las dos fuentes se mezclarian sin poder separarlas."""
    out = procesar_partidos_footballdata(_df([_partido_fd("1", "Arsenal", "Chelsea", 2, 1)]))
    assert out["fuente"].iloc[0] == FUENTE_FOOTBALLDATA


def test_aplica_las_mismas_transformaciones_que_la_otra_fuente():
    out = procesar_partidos_footballdata(_df([_partido_fd("1", "Arsenal", "Chelsea", 3, 0)]))
    fila = out.iloc[0]
    assert fila["resultado"] == "Local"
    assert fila["diferencia_goles"] == 3
    assert bool(fila["es_goleada"]) is True
    assert fila["anio_partido"] == 2024


def test_no_necesita_tabla_de_equipos():
    """football-data no publica estadios ni ciudades: silver no puede exigirlos."""
    out = procesar_partidos_footballdata(_df([_partido_fd("1", "Arsenal", "Chelsea", 1, 1)]))
    assert len(out) == 1


def test_deduplica_igual_que_la_otra_fuente():
    df = _df([
        _partido_fd("1", "Arsenal", "Chelsea", 2, 1),
        _partido_fd("1", "Arsenal", "Chelsea", 2, 1),
    ])
    assert len(procesar_partidos_footballdata(df)) == 1


def test_descarta_partidos_sin_marcador():
    df = _df([_partido_fd("1", "Arsenal", "Chelsea", None, None)])
    assert procesar_partidos_footballdata(df).empty


# --- Gold separa por fuente ------------------------------------------------
def _silver(fuente, local, visitante, gl, gv, id_evento):
    return {
        "fuente": fuente, "id_evento": id_evento, "liga": "English Premier League",
        "id_equipo_local": f"{fuente}-{local}", "equipo_local": local,
        "id_equipo_visitante": f"{fuente}-{visitante}", "equipo_visitante": visitante,
        "goles_local": gl, "goles_visitante": gv,
        "resultado": "Local" if gl > gv else ("Visitante" if gv > gl else "Empate"),
    }


def test_las_posiciones_se_calculan_por_fuente():
    """El mismo club en dos fuentes no se suma: son dos filas distintas."""
    df = _df([
        _silver("thesportsdb", "Manchester United", "Chelsea", 3, 0, "a"),
        _silver("footballdata", "Man United", "Chelsea", 1, 0, "b"),
    ])
    tabla = calcular_tabla_posiciones(df)
    assert "fuente" in tabla.columns
    assert set(tabla["fuente"]) == {"thesportsdb", "footballdata"}
    assert len(tabla[tabla["equipo"] == "Manchester United"]) == 1
    assert len(tabla[tabla["equipo"] == "Man United"]) == 1


def test_cada_tabla_es_internamente_consistente():
    """Los puntos de una fuente no pueden contaminarse con los de la otra."""
    df = _df([
        _silver("footballdata", "Arsenal", "Chelsea", 2, 0, "a"),
        _silver("footballdata", "Arsenal", "Everton", 1, 0, "b"),
        _silver("thesportsdb", "Arsenal", "Chelsea", 0, 3, "c"),
    ])
    tabla = calcular_tabla_posiciones(df)
    fd = tabla[(tabla["fuente"] == "footballdata") & (tabla["equipo"] == "Arsenal")].iloc[0]
    ts = tabla[(tabla["fuente"] == "thesportsdb") & (tabla["equipo"] == "Arsenal")].iloc[0]
    assert fd["PJ"] == 2 and fd["Pts"] == 6
    assert ts["PJ"] == 1 and ts["Pts"] == 0


def test_sin_columna_fuente_sigue_funcionando():
    """Retrocompatibilidad: silver viejo no tiene la columna."""
    df = _df([_silver("x", "Arsenal", "Chelsea", 2, 0, "a")]).drop(columns=["fuente"])
    tabla = calcular_tabla_posiciones(df)
    assert len(tabla) == 2
