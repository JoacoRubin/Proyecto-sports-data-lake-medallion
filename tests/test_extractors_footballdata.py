# =============================================================================
# tests/test_extractors_footballdata.py — Tests de la extracción desde
# football-data.co.uk.
#
# El descargador se inyecta: los tests corren SIN red y son reproducibles.
# =============================================================================

import pandas as pd
import pytest

from bronze.extractors_footballdata import (
    extraer_partidos_footballdata,
    url_csv,
)

_CSV = (
    "Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HTHG,HTAG,HTR,Referee,"
    "HS,AS,HST,AST,HF,AF,HC,AC,HY,AY,HR,AR,B365H,B365D,B365A\n"
    "E0,16/08/2024,20:00,Man United,Fulham,1,0,H,0,0,D,R Jones,"
    "14,10,5,2,12,10,7,8,2,3,0,0,1.6,4.2,5.25\n"
    "E0,17/08/2024,12:30,Ipswich,Liverpool,0,2,A,0,0,D,T Robinson,"
    "7,18,2,5,9,18,2,10,3,1,0,0,8.5,5.5,1.33\n"
)

# Filas basura reales del CSV: el archivo termina con líneas vacías/parciales.
_CSV_CON_BASURA = _CSV + ",,,,,,,,,,,,,,,,,,,,,,,,,,\n"


def _descargador(_url: str) -> str:
    return _CSV


def _extraer(descargador=_descargador, div="E0", temporada="2425"):
    return extraer_partidos_footballdata(div, temporada, descargador=descargador)


# --- URL -------------------------------------------------------------------
def test_url_csv_arma_la_ruta_de_football_data():
    url = url_csv("E0", "2425")
    assert url == "https://www.football-data.co.uk/mmz4281/2425/E0.csv"


# --- Extracción feliz ------------------------------------------------------
def test_extraccion_devuelve_una_fila_por_partido():
    assert len(_extraer()) == 2


def test_extraccion_produce_el_esquema_canonico_bronze():
    df = _extraer()
    for col in ("id_evento", "liga", "equipo_local", "equipo_visitante",
                "goles_local", "goles_visitante", "fecha_partido",
                "estado", "fecha_extraccion", "timestamp_extraccion"):
        assert col in df.columns, f"falta la columna canónica '{col}'"


def test_extraccion_tipa_los_goles():
    df = _extraer()
    assert df["goles_local"].dtype == "Int64"
    assert df.loc[0, "goles_local"] == 1
    assert df.loc[1, "goles_visitante"] == 2


def test_extraccion_es_idempotente_en_los_ids():
    """Dos corridas sobre el mismo CSV deben dar los mismos id_evento."""
    assert list(_extraer()["id_evento"]) == list(_extraer()["id_evento"])


def test_extraccion_no_genera_ids_duplicados():
    df = _extraer()
    assert df["id_evento"].is_unique


def test_extraccion_descarta_filas_basura():
    """El CSV real trae líneas vacías al final. No deben llegar a bronze."""
    df = extraer_partidos_footballdata(
        "E0", "2425", descargador=lambda _u: _CSV_CON_BASURA
    )
    assert len(df) == 2
    assert df["equipo_local"].notna().all()


# --- Fallos ruidosos -------------------------------------------------------
def test_liga_desconocida_falla_temprano():
    """Un código de división inventado debe reventar ANTES de bajar nada."""
    with pytest.raises(ValueError, match="[Dd]ivisi"):
        extraer_partidos_footballdata("XX", "2425", descargador=_descargador)


def test_csv_vacio_falla_ruidosamente():
    """Cabecera bien formada pero sin ninguna fila: no hay nada que ingerir."""
    cabecera = "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG\n"
    with pytest.raises(ValueError, match="[Ss]in datos|vac"):
        extraer_partidos_footballdata("E0", "2425", descargador=lambda _u: cabecera)


def test_csv_sin_columnas_obligatorias_falla():
    """Si football-data cambia el formato, queremos enterarnos en el acto."""
    with pytest.raises(ValueError, match="[Cc]olumnas"):
        extraer_partidos_footballdata(
            "E0", "2425", descargador=lambda _u: "Div,Date\nE0,16/08/2024\n"
        )


def test_multiples_temporadas_no_colisionan():
    """Mismo partido nominal en temporadas distintas = ids distintos."""
    a = _extraer(temporada="2425")
    b = _extraer(temporada="2324")
    assert set(a["id_evento"]).isdisjoint(set(b["id_evento"]))


def test_la_temporada_queda_registrada_en_la_fila():
    df = _extraer(temporada="2425")
    assert (df["temporada"] == "2024-2025").all()
