# =============================================================================
# tests/test_ml_training.py — Tests del entrenamiento.
#
# Los tres escalones existen por una razón: sin la tasa base no sabés si tu
# modelo aprendió algo, y sin la logística no sabés si el gradient boosting
# está aportando o solo memorizando.
# =============================================================================

import numpy as np
import pandas as pd
import pytest

from ml.features import TARGET
from ml.training import (
    MODELOS,
    evaluar_por_folds,
    modelo_hgb,
    modelo_logistico,
    modelo_tasa_base,
)
from tests.test_ml_dataset import _df_features


# --- Los modelos entrenan y devuelven probabilidades -----------------------
@pytest.mark.parametrize("fabrica", [modelo_tasa_base, modelo_logistico, modelo_hgb])
def test_el_modelo_entrena_y_devuelve_probabilidades(fabrica):
    from ml.dataset import preparar_matriz
    X, y = preparar_matriz(_df_features(por_temporada=40))

    modelo = fabrica()
    modelo.fit(X, y)
    prob = modelo.predict_proba(X)[:, 1]

    assert len(prob) == len(y)
    assert prob.min() >= 0.0 and prob.max() <= 1.0


def test_la_tasa_base_predice_siempre_la_prevalencia():
    """El escalón 0: la vara contra la que se mide todo lo demás."""
    from ml.dataset import preparar_matriz
    X, y = preparar_matriz(_df_features(por_temporada=40))

    modelo = modelo_tasa_base().fit(X, y)
    prob = modelo.predict_proba(X)[:, 1]

    assert np.allclose(prob, y.mean()), "debe predecir la prevalencia, constante"


def test_los_modelos_toleran_nulos():
    """Las medias móviles son NaN en los primeros partidos de cada equipo."""
    from ml.dataset import preparar_matriz
    df = _df_features(por_temporada=40)
    df.loc[df.index[:20], "faltas_prom5_local"] = np.nan
    X, y = preparar_matriz(df)

    for fabrica in (modelo_logistico, modelo_hgb):
        prob = fabrica().fit(X, y).predict_proba(X)[:, 1]
        assert not np.isnan(prob).any(), f"{fabrica.__name__} devolvió NaN"


def test_el_registro_de_modelos_tiene_los_tres_escalones():
    assert set(MODELOS) == {"tasa_base", "logistico", "hgb"}


# --- Evaluación por folds --------------------------------------------------
def test_evaluar_por_folds_devuelve_una_fila_por_fold():
    df = _df_features(por_temporada=60)
    res = evaluar_por_folds(df, modelo_tasa_base, min_temporadas_train=2)
    assert len(res) == 2
    assert set(res["fold"]) == {0, 1}


def test_evaluar_por_folds_reporta_la_temporada_evaluada():
    df = _df_features(por_temporada=60)
    res = evaluar_por_folds(df, modelo_tasa_base, min_temporadas_train=2)
    assert set(res["temporada_test"]) == {"2020-2021", "2021-2022"}


def test_evaluar_por_folds_trae_las_metricas():
    df = _df_features(por_temporada=60)
    res = evaluar_por_folds(df, modelo_hgb, min_temporadas_train=2)
    for col in ("pr_auc", "lift_pr_auc", "log_loss", "brier", "tasa_base", "n_test", "n_positivos"):
        assert col in res.columns, f"falta '{col}'"


def test_la_tasa_base_tiene_lift_uno_por_definicion():
    """Predecir la prevalencia no puede ganarle a la prevalencia."""
    df = _df_features(por_temporada=60)
    res = evaluar_por_folds(df, modelo_tasa_base, min_temporadas_train=2)
    assert res["lift_pr_auc"].max() == pytest.approx(1.0, abs=0.25)
