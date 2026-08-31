# =============================================================================
# tests/test_ml_inspection.py — Tests de la importancia por permutación.
#
# POR QUÉ NO ALCANZA CON LEER LOS COEFICIENTES
#
# Un coeficiente dice cuánto pesa una variable DENTRO del modelo. No dice si
# esa variable sirve FUERA de muestra — que es la única pregunta que importa.
# De hecho, los coeficientes más grandes que tuvimos correspondían a equipos
# con 30 partidos: peso enorme, valor predictivo nulo.
#
# La importancia por permutación mide otra cosa: cuánto EMPEORA la métrica en
# datos no vistos si rompés esa columna. Es la definición operativa de "esta
# feature aporta".
# =============================================================================

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from ml.inspection import importancia_por_permutacion


@pytest.fixture
def escenario():
    """Tres features: una que predice, una copia ruidosa, y una inútil."""
    rng = np.random.default_rng(0)
    n = 1200
    senal = rng.normal(size=n)
    y = rng.binomial(1, 1 / (1 + np.exp(-(senal - 1.0))))
    X = pd.DataFrame({
        "senal": senal,
        "senal_ruidosa": senal + rng.normal(scale=2.0, size=n),
        "basura": rng.normal(size=n),
    })
    y = pd.Series(y)
    modelo = LogisticRegression(max_iter=1000).fit(X[:800], y[:800])
    return modelo, X[800:], y[800:]


def test_devuelve_una_fila_por_feature(escenario):
    modelo, X, y = escenario
    imp = importancia_por_permutacion(modelo, X, y, n_repeticiones=8, semilla=0)
    assert set(imp["feature"]) == {"senal", "senal_ruidosa", "basura"}


def test_viene_ordenada_de_mayor_a_menor(escenario):
    modelo, X, y = escenario
    imp = importancia_por_permutacion(modelo, X, y, n_repeticiones=8, semilla=0)
    assert list(imp["importancia"]) == sorted(imp["importancia"], reverse=True)


def test_la_feature_predictiva_queda_arriba(escenario):
    modelo, X, y = escenario
    imp = importancia_por_permutacion(modelo, X, y, n_repeticiones=10, semilla=0)
    assert imp.iloc[0]["feature"] == "senal"


def test_la_feature_basura_tiene_importancia_despreciable(escenario):
    """Romper una columna que no aporta no puede empeorar la métrica."""
    modelo, X, y = escenario
    imp = importancia_por_permutacion(modelo, X, y, n_repeticiones=10, semilla=0)
    basura = imp[imp["feature"] == "basura"]["importancia"].iloc[0]
    senal  = imp[imp["feature"] == "senal"]["importancia"].iloc[0]
    assert abs(basura) < senal / 4


def test_reporta_la_dispersion_para_no_leer_ruido(escenario):
    """Sin desvío no se sabe si una diferencia entre features es real."""
    modelo, X, y = escenario
    imp = importancia_por_permutacion(modelo, X, y, n_repeticiones=8, semilla=0)
    assert "desvio" in imp.columns
    assert (imp["desvio"] >= 0).all()


def test_es_reproducible(escenario):
    modelo, X, y = escenario
    a = importancia_por_permutacion(modelo, X, y, n_repeticiones=6, semilla=42)
    b = importancia_por_permutacion(modelo, X, y, n_repeticiones=6, semilla=42)
    pd.testing.assert_frame_equal(a, b)


def test_mide_sobre_pr_auc_no_sobre_accuracy(escenario):
    """En un target desbalanceado, permutar contra accuracy no detecta nada."""
    modelo, X, y = escenario
    imp = importancia_por_permutacion(modelo, X, y, n_repeticiones=6, semilla=0)
    assert imp.attrs.get("metrica") == "average_precision"
