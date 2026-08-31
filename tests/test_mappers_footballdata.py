# =============================================================================
# tests/test_mappers_footballdata.py — Tests del mapeo de football-data.co.uk
# al esquema canónico bronze.
#
# Esta fuente NO trae ids: hay que sintetizarlos. El requisito crítico es que
# sean DETERMINISTAS — si no, cada re-descarga genera ids nuevos, la
# deduplicación por id_evento deja de funcionar y el data lake se llena de
# duplicados silenciosos.
# =============================================================================

import pandas as pd
import pytest

from bronze.mappers_footballdata import (
    LIGAS_FOOTBALLDATA,
    codigo_temporada_a_nombre,
    generar_id_equipo,
    generar_id_evento,
    mapear_partido_footballdata,
    parsear_fecha,
    tipar_partidos_footballdata,
)

_RAW = {
    "Div": "E0", "Date": "16/08/2024", "Time": "20:00",
    "HomeTeam": "Man United", "AwayTeam": "Fulham",
    "FTHG": "1", "FTAG": "0", "FTR": "H",
    "HTHG": "0", "HTAG": "0", "HTR": "D",
    "Referee": "R Jones",
    "HS": "14", "AS": "10", "HST": "5", "AST": "2",
    "HF": "12", "AF": "10", "HC": "7", "AC": "8",
    "HY": "2", "AY": "3", "HR": "0", "AR": "0",
    "B365H": "1.6", "B365D": "4.2", "B365A": "5.25",
}

_TS    = "2024-09-01T00:00:00+00:00"
_FECHA = "2024-09-01"


def _mapear(raw=None, div="E0", temporada="2425"):
    return mapear_partido_footballdata(raw or _RAW, div, temporada, _TS, _FECHA)


# --- Fechas: el CSV usa DD/MM/YYYY y las temporadas viejas DD/MM/YY --------
def test_parsear_fecha_formato_de_cuatro_digitos():
    assert parsear_fecha("16/08/2024") == "2024-08-16"


def test_parsear_fecha_formato_de_dos_digitos():
    """Las temporadas anteriores a ~2003 usan año de 2 dígitos."""
    assert parsear_fecha("14/08/93") == "1993-08-14"


def test_parsear_fecha_invalida_devuelve_none():
    assert parsear_fecha("") is None
    assert parsear_fecha("no-es-fecha") is None


# --- Temporadas: el código "2425" tiene que volverse legible ---------------
@pytest.mark.parametrize("codigo,esperado", [
    ("2425", "2024-2025"),
    ("2021", "2020-2021"),
    ("0001", "2000-2001"),
    ("9394", "1993-1994"),
])
def test_codigo_temporada_a_nombre(codigo, esperado):
    assert codigo_temporada_a_nombre(codigo) == esperado


# --- Ids sintéticos: DETERMINISMO ------------------------------------------
def test_id_evento_es_determinista():
    """Misma entrada, mismo id. Sin esto la deduplicación no sirve."""
    a = generar_id_evento("E0", "2425", "2024-08-16", "Man United", "Fulham")
    b = generar_id_evento("E0", "2425", "2024-08-16", "Man United", "Fulham")
    assert a == b


def test_id_evento_ignora_espacios_accidentales():
    """'Man United ' y 'Man United' son el mismo equipo: mismo partido."""
    a = generar_id_evento("E0", "2425", "2024-08-16", "Man United", "Fulham")
    b = generar_id_evento("E0", "2425", "2024-08-16", " Man United", "Fulham ")
    assert a == b


def test_id_evento_distingue_partidos_distintos():
    ida    = generar_id_evento("E0", "2425", "2024-08-16", "Man United", "Fulham")
    vuelta = generar_id_evento("E0", "2425", "2025-01-20", "Fulham", "Man United")
    assert ida != vuelta


def test_id_evento_lleva_prefijo_de_fuente():
    """El prefijo evita colisionar con los ids numéricos de TheSportsDB."""
    assert generar_id_evento("E0", "2425", "2024-08-16", "A", "B").startswith("FD-")


def test_id_equipo_es_determinista_y_legible():
    assert generar_id_equipo("Man United") == "FD-man-united"
    assert generar_id_equipo("Ath Bilbao") == "FD-ath-bilbao"


def test_id_equipo_normaliza_acentos_y_puntos():
    assert generar_id_equipo("Atlético Madrid") == "FD-atletico-madrid"
    assert generar_id_equipo("St. Etienne")     == "FD-st-etienne"


# --- Mapeo al esquema canónico ---------------------------------------------
def test_mapeo_produce_las_17_columnas_canonicas_de_bronze():
    """Debe encajar en el MISMO esquema que TheSportsDB, o silver no lo lee."""
    canonicas = {
        "id_evento", "nombre_evento", "temporada", "liga", "id_liga",
        "equipo_local", "id_equipo_local", "goles_local",
        "equipo_visitante", "id_equipo_visitante", "goles_visitante",
        "fecha_partido", "hora_partido", "estadio", "estado",
        "timestamp_extraccion", "fecha_extraccion",
    }
    assert canonicas.issubset(set(_mapear().keys()))


def test_mapeo_traduce_el_codigo_de_liga_a_nombre():
    assert _mapear()["liga"] == "English Premier League"
    assert LIGAS_FOOTBALLDATA["SP1"] == "Spanish La Liga"


def test_mapeo_llena_los_campos_canonicos():
    d = _mapear()
    assert d["equipo_local"]     == "Man United"
    assert d["equipo_visitante"] == "Fulham"
    assert d["goles_local"]      == "1"
    assert d["goles_visitante"]  == "0"
    assert d["fecha_partido"]    == "2024-08-16"
    assert d["temporada"]        == "2024-2025"
    assert d["nombre_evento"]    == "Man United vs Fulham"
    assert d["fecha_extraccion"] == _FECHA


def test_mapeo_marca_estado_finalizado():
    """El CSV solo publica partidos jugados: silver los tiene que aceptar."""
    from bronze.mappers import ESTADOS_FINALIZADOS
    assert _mapear()["estado"].upper() in ESTADOS_FINALIZADOS


def test_mapeo_conserva_las_estadisticas_para_ml():
    """Las columnas extra son el motivo de traer esta fuente. No se pierden."""
    d = _mapear()
    assert d["tiros_local"]                 == "14"
    assert d["tiros_arco_local"]            == "5"
    assert d["corners_visitante"]           == "8"
    assert d["amarillas_visitante"]         == "3"
    assert d["goles_local_entretiempo"]     == "0"
    assert d["cuota_local"]                 == "1.6"
    assert d["arbitro"]                     == "R Jones"


def test_mapeo_tolera_columnas_faltantes():
    """Las temporadas viejas no traen tiros ni cuotas. No debe explotar."""
    minimo = {
        "Div": "E0", "Date": "14/08/93",
        "HomeTeam": "Arsenal", "AwayTeam": "Coventry",
        "FTHG": "0", "FTAG": "3",
    }
    d = mapear_partido_footballdata(minimo, "E0", "9394", _TS, _FECHA)
    assert d["tiros_local"] is None
    assert d["cuota_local"] is None
    assert d["goles_visitante"] == "3"


# --- Tipado ----------------------------------------------------------------
def test_tipado_convierte_goles_a_int64_nullable():
    df = pd.DataFrame([_mapear()])
    tipado = tipar_partidos_footballdata(df)
    assert tipado["goles_local"].dtype     == "Int64"
    assert tipado["goles_visitante"].dtype == "Int64"
    assert tipado["id_evento"].dtype       == "string"


def test_tipado_convierte_estadisticas_y_cuotas():
    df = pd.DataFrame([_mapear()])
    tipado = tipar_partidos_footballdata(df)
    assert tipado["tiros_local"].dtype == "Int64"
    assert tipado["cuota_local"].dtype == "Float64"
    assert tipado["tiros_local"].iloc[0] == 14


def test_tipado_no_revienta_con_estadisticas_nulas():
    df = pd.DataFrame([mapear_partido_footballdata(
        {"Div": "E0", "Date": "14/08/93", "HomeTeam": "A", "AwayTeam": "B",
         "FTHG": "0", "FTAG": "3"},
        "E0", "9394", _TS, _FECHA,
    )])
    tipado = tipar_partidos_footballdata(df)
    assert pd.isna(tipado["tiros_local"].iloc[0])
    assert tipado["goles_visitante"].iloc[0] == 3


# =============================================================================
# Cuotas del MERCADO (promedio de todas las casas), no de una sola.
#
# Habilitan el baseline mas duro que existe en prediccion deportiva: la
# prediccion agregada de gente que se juega plata. Le ganamos a "predecir la
# prevalencia" — eso es una vara pasiva. El mercado es una vara adversaria.
#
# GOTCHA: football-data renombro estas columnas en 2019.
#   2015-16 a 2018-19 : BbAv>2.5, BbAv<2.5, BbAvH/D/A   (Betbrain average)
#   2019-20 en adelante: Avg>2.5,  Avg<2.5,  AvgH/D/A
# Son la misma magnitud. Sin el fallback se pierden 4 temporadas.
# =============================================================================

def test_mapea_las_cuotas_de_over_under_del_mercado():
    raw = {**_RAW, "Avg>2.5": "1.85", "Avg<2.5": "1.95"}
    d = _mapear(raw)
    assert d["cuota_over25"]  == "1.85"
    assert d["cuota_under25"] == "1.95"


def test_usa_el_nombre_viejo_cuando_falta_el_nuevo():
    """Temporadas anteriores a 2019-20: BbAv en lugar de Avg."""
    raw = {**_RAW, "BbAv>2.5": "1.80", "BbAv<2.5": "2.00"}
    d = _mapear(raw)
    assert d["cuota_over25"]  == "1.80"
    assert d["cuota_under25"] == "2.00"


def test_el_nombre_nuevo_tiene_prioridad_sobre_el_viejo():
    raw = {**_RAW, "Avg>2.5": "1.85", "BbAv>2.5": "9.99"}
    assert _mapear(raw)["cuota_over25"] == "1.85"


def test_mapea_el_1x2_del_mercado_ademas_del_de_bet365():
    """El consenso de todas las casas es mejor estimador que una sola."""
    raw = {**_RAW, "AvgH": "1.70", "AvgD": "4.00", "AvgA": "5.00"}
    d = _mapear(raw)
    assert d["cuota_local_mercado"]     == "1.70"
    assert d["cuota_empate_mercado"]    == "4.00"
    assert d["cuota_visitante_mercado"] == "5.00"
    assert d["cuota_local"] == "1.6", "la de Bet365 se conserva"


def test_el_1x2_del_mercado_tambien_tiene_fallback():
    raw = {**_RAW, "BbAvH": "1.75", "BbAvD": "3.90", "BbAvA": "4.80"}
    d = _mapear(raw)
    assert d["cuota_local_mercado"]  == "1.75"
    assert d["cuota_empate_mercado"] == "3.90"


def test_sin_cuotas_de_mercado_queda_nulo():
    d = _mapear(_RAW)
    assert d["cuota_over25"] is None
    assert d["cuota_local_mercado"] is None


def test_las_cuotas_de_mercado_se_tipan_como_decimales():
    raw = {**_RAW, "Avg>2.5": "1.85", "AvgH": "1.70"}
    df = tipar_partidos_footballdata(pd.DataFrame([_mapear(raw)]))
    assert df["cuota_over25"].dtype        == "Float64"
    assert df["cuota_local_mercado"].dtype == "Float64"
    assert df["cuota_over25"].iloc[0] == pytest.approx(1.85)
