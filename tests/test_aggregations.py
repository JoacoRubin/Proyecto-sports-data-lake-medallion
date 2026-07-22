# =============================================================================
# tests/test_aggregations.py — Tests de la agregación gold (T7).
#
# Verifica el cálculo de la tabla de posiciones y, sobre todo, que la
# agrupación use el id del equipo (clave estable) y no su nombre (mutable).
# =============================================================================

import pandas as pd

from gold.aggregations import calcular_tabla_posiciones


def _partido(liga, id_local, nom_local, goles_local, id_vis, nom_vis, goles_vis):
    """Construye una fila silver mínima con el `resultado` ya derivado."""
    if goles_local > goles_vis:
        resultado = "Local"
    elif goles_local < goles_vis:
        resultado = "Visitante"
    else:
        resultado = "Empate"
    return {
        "liga": liga,
        "id_equipo_local": id_local, "equipo_local": nom_local, "goles_local": goles_local,
        "id_equipo_visitante": id_vis, "equipo_visitante": nom_vis, "goles_visitante": goles_vis,
        "resultado": resultado,
    }


def test_puntos_y_estadisticas_basicas():
    """PJ, PG/PE/PP, GF/GC/DG y Pts (3*PG + PE) combinando local y visitante."""
    df = pd.DataFrame([
        _partido("Liga", "1", "River", 2, "2", "Boca", 0),  # River gana de local
        _partido("Liga", "2", "Boca", 1, "1", "River", 1),  # empate
    ])
    tabla = calcular_tabla_posiciones(df)

    river = tabla[tabla["id_equipo"] == "1"].iloc[0]
    assert river["PJ"] == 2
    assert river["PG"] == 1
    assert river["PE"] == 1
    assert river["PP"] == 0
    assert river["GF"] == 3   # 2 de local + 1 de visitante
    assert river["GC"] == 1
    assert river["DG"] == 2
    assert river["Pts"] == 4  # 3*1 + 1

    boca = tabla[tabla["id_equipo"] == "2"].iloc[0]
    assert boca["PP"] == 1
    assert boca["Pts"] == 1


def test_agrupa_por_id_no_por_nombre():
    """El mismo id con nombres distintos debe consolidarse en UNA fila.

    Antes se agrupaba por nombre: 'River' y 'River Plate' se contaban como dos
    equipos. Agrupando por id_equipo (clave estable) el problema desaparece.
    """
    df = pd.DataFrame([
        _partido("Liga", "1", "River", 1, "2", "Boca", 0),
        _partido("Liga", "2", "Boca", 0, "1", "River Plate", 3),
    ])
    tabla = calcular_tabla_posiciones(df)

    filas_equipo_1 = tabla[tabla["id_equipo"] == "1"]
    assert len(filas_equipo_1) == 1
    assert filas_equipo_1.iloc[0]["PJ"] == 2


def test_multiliga_no_mezcla_equipos_entre_ligas():
    """Con varias ligas, cada equipo queda en la suya; las tablas no se cruzan."""
    df = pd.DataFrame([
        _partido("Liga A", "1", "River", 2, "2", "Boca", 0),
        _partido("Liga B", "3", "Real Madrid", 1, "4", "Barcelona", 1),
    ])
    tabla = calcular_tabla_posiciones(df)
    assert set(tabla["liga"]) == {"Liga A", "Liga B"}
    assert len(tabla) == 4  # 2 equipos por liga, sin cruces

    river = tabla[(tabla["liga"] == "Liga A") & (tabla["id_equipo"] == "1")].iloc[0]
    assert river["Pts"] == 3  # su puntaje no se contamina con la otra liga


def test_ordena_por_puntos_descendente():
    """La tabla queda ordenada por Pts dentro de cada liga."""
    df = pd.DataFrame([
        _partido("Liga", "1", "River", 3, "2", "Boca", 0),
        _partido("Liga", "1", "River", 2, "3", "San Lorenzo", 0),
    ])
    tabla = calcular_tabla_posiciones(df)
    assert tabla.iloc[0]["id_equipo"] == "1"  # River, puntero
    assert tabla.iloc[0]["Pts"] >= tabla.iloc[-1]["Pts"]
