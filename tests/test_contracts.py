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


# =============================================================================
# Contrato bronze — la frontera que hace posible el MULTI-FUENTE.
#
# TheSportsDB y football-data.co.uk entran por caminos distintos, pero ambas
# deben cumplir el MISMO contrato antes de tocar silver. Si una fuente cambia
# su formato, revienta acá y no dos capas más abajo.
# =============================================================================

from quality.contracts import validar_bronze_partidos


def _fila_bronze(id_evento="1", local="River", visitante="Boca", gl=2, gv=1,
                 fecha="2024-05-15", estado="FT"):
    return {
        "id_evento": id_evento, "liga": "Liga", "temporada": "2024",
        "equipo_local": local, "id_equipo_local": f"id-{local}",
        "equipo_visitante": visitante, "id_equipo_visitante": f"id-{visitante}",
        "goles_local": gl, "goles_visitante": gv,
        "fecha_partido": fecha, "estado": estado,
    }


def test_bronze_valido_pasa():
    df = pd.DataFrame([_fila_bronze("1"), _fila_bronze("2", "Racing", "Independiente")])
    validar_bronze_partidos(df)  # no debe lanzar


def test_bronze_acepta_ambas_fuentes_en_la_misma_tabla():
    """Ids numéricos (TheSportsDB) y sintéticos (football-data) conviven."""
    df = pd.DataFrame([
        _fila_bronze("1687432"),                    # TheSportsDB
        _fila_bronze("FD-a1b2c3d4e5"),              # football-data
    ])
    validar_bronze_partidos(df)


def test_bronze_acepta_partidos_pendientes_sin_goles():
    """TheSportsDB trae partidos no jugados: goles nulos son válidos en bronze."""
    fila = _fila_bronze("1", gl=None, gv=None, estado="Not Started")
    validar_bronze_partidos(pd.DataFrame([fila]))


def test_bronze_rechaza_id_evento_duplicado():
    """Es la garantía de idempotencia: dos veces el mismo partido es un bug."""
    df = pd.DataFrame([_fila_bronze("1"), _fila_bronze("1")])
    with pytest.raises(pa.errors.SchemaErrors):
        validar_bronze_partidos(df)


def test_bronze_rechaza_equipo_contra_si_mismo():
    """Invariante de negocio: nadie juega contra sí mismo."""
    df = pd.DataFrame([_fila_bronze("1", local="River", visitante="River")])
    with pytest.raises(pa.errors.SchemaErrors):
        validar_bronze_partidos(df)


def test_bronze_rechaza_fecha_no_iso():
    """football-data usa DD/MM/YYYY: si el mapeo no la convierte, revienta acá."""
    df = pd.DataFrame([_fila_bronze("1", fecha="16/08/2024")])
    with pytest.raises(pa.errors.SchemaErrors):
        validar_bronze_partidos(df)


def test_bronze_rechaza_goles_negativos():
    df = pd.DataFrame([_fila_bronze("1", gl=-1)])
    with pytest.raises(pa.errors.SchemaErrors):
        validar_bronze_partidos(df)


def test_bronze_rechaza_equipo_nulo():
    df = pd.DataFrame([_fila_bronze("1", local=None)])
    with pytest.raises(pa.errors.SchemaErrors):
        validar_bronze_partidos(df)


# =============================================================================
# Contrato de la matriz de FEATURES (frontera de ML).
#
# El proyecto valida bronze, silver y gold. La frontera de ML no tenia
# contrato, y es donde mas barato sale un error silencioso: una columna con
# leakage, una probabilidad fuera de rango o una tasa historica corrupta no
# rompen nada — entrenan un modelo malo y nadie se entera.
# =============================================================================

from quality.contracts import validar_matriz_ml
from ml.features import TARGET


def _fila_ml(**extra):
    base = {
        TARGET: 1,
        "liga": "Spanish La Liga",
        "prob_local": 0.45, "prob_empate": 0.30, "prob_visitante": 0.25,
        "paridad": 0.80,
        "equipo_tasa_exp_local": 0.17, "equipo_tasa_exp_visitante": 0.19,
        "h2h_tasa_exp": 0.17,
        "faltas_prom5_local": 12.0, "amarillas_prom5_local": 2.1,
        "mes": 5, "fecha_num": 12,
    }
    base.update(extra)
    return base


def test_matriz_ml_valida_pasa():
    validar_matriz_ml(pd.DataFrame([_fila_ml(), _fila_ml(**{TARGET: 0})]))


def test_matriz_ml_rechaza_columnas_con_leakage():
    """LA razon de existir de este contrato: que no se cuele el propio partido."""
    df = pd.DataFrame([_fila_ml()])
    df["rojas_local"] = 1          # estadistica del partido que se quiere predecir
    with pytest.raises(ValueError, match="[Ll]eakage"):
        validar_matriz_ml(df)


def test_matriz_ml_rechaza_goles_del_propio_partido():
    df = pd.DataFrame([_fila_ml()])
    df["goles_local"] = 2
    with pytest.raises(ValueError, match="[Ll]eakage"):
        validar_matriz_ml(df)


def test_matriz_ml_rechaza_target_no_binario():
    with pytest.raises(pa.errors.SchemaErrors):
        validar_matriz_ml(pd.DataFrame([_fila_ml(**{TARGET: 3})]))


def test_matriz_ml_rechaza_probabilidad_fuera_de_rango():
    with pytest.raises(pa.errors.SchemaErrors):
        validar_matriz_ml(pd.DataFrame([_fila_ml(prob_local=1.4)]))


def test_matriz_ml_rechaza_tasa_historica_imposible():
    """Una tasa suavizada siempre vive en [0, 1]. Fuera de ahi hay un bug."""
    with pytest.raises(pa.errors.SchemaErrors):
        validar_matriz_ml(pd.DataFrame([_fila_ml(equipo_tasa_exp_local=1.8)]))


def test_matriz_ml_rechaza_probabilidades_que_no_suman_uno():
    """Se normalizan al construirlas: si no suman 1, el mapeo se rompio."""
    fila = _fila_ml(prob_local=0.9, prob_empate=0.9, prob_visitante=0.9)
    with pytest.raises(pa.errors.SchemaErrors):
        validar_matriz_ml(pd.DataFrame([fila]))


def test_matriz_ml_acepta_nulos_en_las_medias_moviles():
    """Los primeros partidos de cada equipo no tienen historia. Es esperado."""
    validar_matriz_ml(pd.DataFrame([_fila_ml(faltas_prom5_local=None,
                                             amarillas_prom5_local=None)]))


def test_matriz_ml_acepta_partidos_sin_cuotas():
    """6 partidos del dataset real no tienen cuotas publicadas (suspendidos).

    Con el sum() por defecto de pandas esas filas suman 0.0 en vez de NaN y el
    invariante las marcaria como violacion. Es un caso valido, no un bug.
    """
    fila = _fila_ml(prob_local=None, prob_empate=None, prob_visitante=None)
    validar_matriz_ml(pd.DataFrame([fila]))


# =============================================================================
# El contrato de ML sirve a los DOS modelos.
#
# Se construyo para expulsiones y despues se agrego un segundo modelo sin
# ninguno: la misma incoherencia que el contrato venia a resolver. Se
# parametriza por target en vez de copiarse, y los nombres de las tasas se
# validan por patron porque cambian entre modelos (tasa_exp / tasa_over).
# =============================================================================

from ml.features_goles import TARGET_GOLES


def _fila_goles(**extra):
    base = {
        TARGET_GOLES: 1,
        "liga": "Italian Serie A",
        "prob_local": 0.45, "prob_empate": 0.30, "prob_visitante": 0.25,
        "paridad": 0.80,
        "equipo_tasa_over_local": 0.55, "equipo_tasa_over_visitante": 0.51,
        "h2h_tasa_over": 0.53,
        "goles_favor_prom5_local": 1.6, "goles_contra_prom5_local": 1.2,
        "prob_mercado_over25": 0.58,
        "mes": 5, "fecha_num": 12,
    }
    base.update(extra)
    return base


def test_matriz_de_goles_valida_pasa():
    validar_matriz_ml(pd.DataFrame([_fila_goles()]), target=TARGET_GOLES)


def test_matriz_de_goles_rechaza_leakage():
    df = pd.DataFrame([_fila_goles()])
    df["goles_totales"] = 3          # derivado del marcador del propio partido
    with pytest.raises(ValueError, match="[Ll]eakage"):
        validar_matriz_ml(df, target=TARGET_GOLES)


def test_matriz_de_goles_rechaza_target_no_binario():
    with pytest.raises(pa.errors.SchemaErrors):
        validar_matriz_ml(pd.DataFrame([_fila_goles(**{TARGET_GOLES: 7})]), target=TARGET_GOLES)


def test_matriz_de_goles_valida_la_tasa_over_por_patron():
    """Los nombres cambian entre modelos: se validan por patron, no por lista."""
    fila = _fila_goles(equipo_tasa_over_local=1.7)
    with pytest.raises(pa.errors.SchemaErrors):
        validar_matriz_ml(pd.DataFrame([fila]), target=TARGET_GOLES)


def test_matriz_de_goles_valida_la_probabilidad_del_mercado():
    """El mercado viaja en la matriz: tambien tiene que estar en rango."""
    fila = _fila_goles(prob_mercado_over25=1.3)
    with pytest.raises(pa.errors.SchemaErrors):
        validar_matriz_ml(pd.DataFrame([fila]), target=TARGET_GOLES)


def test_el_contrato_de_expulsiones_sigue_funcionando_igual():
    """La parametrizacion no puede romper al modelo que ya estaba."""
    validar_matriz_ml(pd.DataFrame([_fila_ml()]))
    with pytest.raises(pa.errors.SchemaErrors):
        validar_matriz_ml(pd.DataFrame([_fila_ml(equipo_tasa_exp_local=1.8)]))
