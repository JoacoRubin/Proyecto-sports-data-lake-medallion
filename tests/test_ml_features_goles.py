# =============================================================================
# tests/test_ml_features_goles.py — Features del modelo de goles (over 2.5).
#
# DOS COSAS NUEVAS RESPECTO DEL MODELO DE EXPULSIONES
#
# 1. El target esta BALANCEADO (52,8%) y el evento es frecuente (2,78 goles por
#    partido contra 0,20 rojas). Hay mucha mas senal para extraer.
#
# 2. Existe un BASELINE ADVERSARIO: la probabilidad implicita del mercado de
#    apuestas. Le ganamos a "predecir la prevalencia" es una vara pasiva; el
#    mercado es el consenso de gente que se juega plata.
#
# Por eso el mercado se trata con cuidado: es baseline, y solo entra como
# feature en el experimento que explicitamente pregunta si nuestro modelo
# aporta informacion POR ENCIMA del mercado.
# =============================================================================

import numpy as np
import pandas as pd
import pytest

from ml.features_goles import (
    TARGET_GOLES,
    agregar_target_goles,
    construir_features_goles,
    probabilidad_mercado_over25,
)


def _partido(fecha, local, visitante, gl=1, gv=1, tiros_l=12, tiros_v=10,
             tiros_arco_l=4, tiros_arco_v=3, corners_l=5, corners_v=4,
             c_over=1.90, c_under=1.90, temporada="2020-2021"):
    return {
        "fecha_partido": fecha, "temporada": temporada, "liga": "Spanish La Liga",
        "equipo_local": local, "equipo_visitante": visitante,
        "goles_local": gl, "goles_visitante": gv,
        "tiros_local": tiros_l, "tiros_visitante": tiros_v,
        "tiros_arco_local": tiros_arco_l, "tiros_arco_visitante": tiros_arco_v,
        "corners_local": corners_l, "corners_visitante": corners_v,
        "cuota_over25": c_over, "cuota_under25": c_under,
        "cuota_local_mercado": 2.0, "cuota_empate_mercado": 3.4,
        "cuota_visitante_mercado": 3.8,
    }


def _df(filas):
    return pd.DataFrame(filas)


# --- Target ----------------------------------------------------------------
def test_target_es_mas_de_dos_goles_y_medio():
    df = agregar_target_goles(_df([
        _partido("2020-01-01", "A", "B", gl=1, gv=1),   # 2 goles -> under
        _partido("2020-01-02", "A", "B", gl=2, gv=1),   # 3 goles -> over
        _partido("2020-01-03", "A", "B", gl=0, gv=0),   # 0 goles -> under
        _partido("2020-01-04", "A", "B", gl=3, gv=4),   # 7 goles -> over
    ]))
    assert list(df[TARGET_GOLES]) == [0, 1, 0, 1]


def test_el_target_esta_balanceado_a_diferencia_del_de_expulsiones():
    """52,8% en el dataset real: se pueden usar metricas que con 17% mentian."""
    df = agregar_target_goles(_df([
        _partido(f"2020-01-{i+1:02d}", "A", "B", gl=i % 3, gv=1) for i in range(30)
    ]))
    assert 0.2 < df[TARGET_GOLES].mean() < 0.8


# --- Probabilidad del mercado ----------------------------------------------
def test_la_probabilidad_del_mercado_saca_el_margen_de_la_casa():
    """1/cuota suma mas de 1: ese exceso es el overround y hay que normalizarlo."""
    df = _df([_partido("2020-01-01", "A", "B", c_over=1.90, c_under=1.90)])
    p = probabilidad_mercado_over25(df)
    assert p.iloc[0] == pytest.approx(0.5), "cuotas simetricas -> 50%"


def test_la_probabilidad_del_mercado_refleja_la_asimetria():
    df = _df([_partido("2020-01-01", "A", "B", c_over=1.40, c_under=3.00)])
    p = probabilidad_mercado_over25(df)
    assert p.iloc[0] > 0.6, "cuota baja en over -> el mercado lo ve probable"


def test_sin_cuotas_la_probabilidad_del_mercado_es_nula():
    df = _df([_partido("2020-01-01", "A", "B", c_over=None, c_under=None)])
    assert pd.isna(probabilidad_mercado_over25(df).iloc[0])


# --- Historia de ataque y defensa ------------------------------------------
def test_usa_goles_a_favor_y_en_contra_no_tarjetas():
    """Para goles importan ataque y defensa, no la agresividad."""
    out = construir_features_goles(_dataset(), prior=0.53)
    for esperada in ("goles_favor_prom5_local", "goles_contra_prom5_local",
                     "tiros_arco_prom5_local", "corners_prom5_visitante"):
        assert esperada in out.columns, f"falta '{esperada}'"
    assert not [c for c in out.columns if "amarillas" in c or "faltas" in c]


def test_los_goles_en_contra_toman_la_columna_del_rival():
    """El 'goles_contra' del local son los goles del VISITANTE. Es el bug clasico."""
    df = _df([
        _partido("2020-01-01", "A", "B", gl=0, gv=3),   # A recibio 3
        _partido("2020-01-02", "A", "C", gl=1, gv=1),
    ])
    out = construir_features_goles(df, prior=0.53, ventanas=(1,))
    assert out["goles_contra_prom1_local"].iloc[1] == pytest.approx(3.0)
    assert out["goles_favor_prom1_local"].iloc[1] == pytest.approx(0.0)


def test_la_historia_del_equipo_cruza_local_y_visitante():
    df = _df([
        _partido("2020-01-01", "A", "B", gl=4, gv=0),   # A de local, marco 4
        _partido("2020-01-02", "C", "A", gl=0, gv=2),   # A de visitante, marco 2
        _partido("2020-01-03", "A", "D"),
    ])
    out = construir_features_goles(df, prior=0.53, ventanas=(2,))
    assert out["goles_favor_prom2_local"].iloc[2] == pytest.approx(3.0)


# --- El mercado: baseline, no feature por defecto --------------------------
def test_el_mercado_NO_entra_como_feature_por_defecto():
    """Si entrara, el modelo copiaria al mercado y no se podria decir que le gana."""
    out = construir_features_goles(_dataset(), prior=0.53)
    assert "prob_mercado_over25" not in out.columns


def test_el_mercado_se_puede_pedir_explicitamente():
    """Para el experimento que pregunta si aportamos POR ENCIMA del mercado."""
    out = construir_features_goles(_dataset(), prior=0.53, incluir_mercado=True)
    assert "prob_mercado_over25" in out.columns


# --- Leakage ---------------------------------------------------------------
def _dataset(ultimo_gl=1):
    filas, equipos = [], ["A", "B", "C", "D"]
    for i in range(12):
        gl = ultimo_gl if i == 11 else (i % 4)
        filas.append(_partido(
            f"2020-01-{i+1:02d}", equipos[i % 4], equipos[(i + 1) % 4],
            gl=gl, gv=i % 3, tiros_l=10 + i,
        ))
    return _df(filas)


def test_ninguna_feature_depende_del_resultado_del_propio_partido():
    """El mismo test mecanico que protege al modelo de expulsiones."""
    a = construir_features_goles(_dataset(ultimo_gl=0), prior=0.53)
    b = construir_features_goles(_dataset(ultimo_gl=9), prior=0.53)
    columnas = [c for c in a.columns if c != TARGET_GOLES]
    pd.testing.assert_frame_equal(
        a[columnas], b[columnas], check_dtype=False,
        obj="cambiar los goles del ultimo partido altero una feature -> LEAKAGE",
    )


def test_no_sobreviven_los_goles_del_propio_partido():
    out = construir_features_goles(_dataset(), prior=0.53)
    prohibidas = {"goles_local", "goles_visitante", "tiros_local", "corners_local"}
    assert not (prohibidas & set(out.columns))


def test_conserva_una_fila_por_partido():
    df = _dataset()
    assert len(construir_features_goles(df, prior=0.53)) == len(df)


# =============================================================================
# LA GARANTIA DEL EXPERIMENTO: el modelo compite contra el mercado, no lo copia.
#
# El mercado viaja en la matriz para poder evaluarlo como baseline sobre los
# MISMOS folds. Pero si llegara a la matriz de features, el modelo aprenderia
# a reproducirlo y la comparacion "le ganamos al mercado?" no significaria
# nada. Es la misma clase de error que el leakage, con otro disfraz.
# =============================================================================

def test_el_mercado_nunca_llega_a_las_features_aunque_este_en_la_matriz():
    from ml.dataset import columnas_features, preparar_matriz

    matriz = construir_features_goles(_dataset(), prior=0.53, incluir_mercado=True)
    assert "prob_mercado_over25" in matriz.columns, "debe viajar para poder evaluarlo"

    features = columnas_features(matriz, TARGET_GOLES)
    assert "prob_mercado_over25" not in features, \
        "el modelo NO puede ver la probabilidad del mercado"

    X, _ = preparar_matriz(matriz, TARGET_GOLES)
    assert "prob_mercado_over25" not in X.columns


def test_ninguna_cuota_cruda_llega_a_las_features():
    """Las cuotas entran normalizadas como probabilidad, nunca crudas."""
    from ml.dataset import columnas_features

    matriz = construir_features_goles(_dataset(), prior=0.53, incluir_mercado=True)
    features = set(columnas_features(matriz, TARGET_GOLES))
    assert not {c for c in features if c.startswith("cuota_")}
