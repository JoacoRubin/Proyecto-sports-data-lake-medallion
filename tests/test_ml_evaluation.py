# =============================================================================
# tests/test_ml_evaluation.py — Tests de las métricas.
#
# POR QUÉ NO USAMOS ACCURACY.
#
# La tasa base de expulsión es 17,6%. Un modelo que prediga siempre "no hay
# expulsión" acierta el 82,4% y no aprendió absolutamente nada. Accuracy en
# un problema desbalanceado no mide capacidad predictiva: mide la tasa base.
#
# Usamos PR-AUC (cuyo baseline ES la tasa base, así que el lift es
# interpretable), log-loss y Brier para la calidad de la probabilidad, y
# bootstrap para el intervalo — porque con ~350 positivos por fold, un punto
# solo no dice nada.
# =============================================================================

import numpy as np
import pytest

from ml.evaluation import bootstrap_ic, metricas, pr_auc


def _y(n_pos=20, n_neg=80, seed=0):
    y = np.array([1] * n_pos + [0] * n_neg)
    return y


# --- PR-AUC ----------------------------------------------------------------
def test_predictor_perfecto_da_pr_auc_uno():
    y = _y()
    prob = y.astype(float)
    assert pr_auc(y, prob) == pytest.approx(1.0)


def test_predictor_aleatorio_da_pr_auc_cercano_a_la_tasa_base():
    """El baseline de PR-AUC ES la prevalencia. Por eso el lift es legible."""
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.176, 20000)
    prob = rng.random(20000)
    assert pr_auc(y, prob) == pytest.approx(0.176, abs=0.02)


def test_predictor_invertido_es_peor_que_la_tasa_base():
    """Un modelo anti-predictivo debe puntuar POR DEBAJO de la tasa base.

    Los scores tienen que ser distintos entre sí: con un predictor binario
    todos los positivos empatan y el average precision colapsa exactamente a
    la prevalencia, sin poder quedar por debajo.
    """
    y = _y()
    prob = np.linspace(0.0, 1.0, len(y))   # los positivos quedan al fondo del ranking
    assert pr_auc(y, prob) < y.mean()


# --- Paquete de métricas ---------------------------------------------------
def test_metricas_trae_todo_lo_necesario_y_nada_de_accuracy():
    rng = np.random.default_rng(1)
    y = rng.binomial(1, 0.2, 500)
    prob = rng.random(500)
    m = metricas(y, prob)
    for clave in ("pr_auc", "roc_auc", "log_loss", "brier", "tasa_base", "lift_pr_auc"):
        assert clave in m, f"falta la métrica '{clave}'"
    assert "accuracy" not in m, "accuracy no debe reportarse en un target desbalanceado"


def test_lift_compara_contra_la_tasa_base():
    """lift = 1.0 significa 'no le ganás a predecir la prevalencia'."""
    rng = np.random.default_rng(2)
    y = rng.binomial(1, 0.176, 20000)
    m = metricas(y, rng.random(20000))
    assert m["lift_pr_auc"] == pytest.approx(1.0, abs=0.15)


def test_lift_de_un_predictor_perfecto_es_mucho_mayor_que_uno():
    y = _y()
    m = metricas(y, y.astype(float))
    assert m["lift_pr_auc"] > 4.0


def test_brier_de_una_prediccion_perfecta_es_cero():
    y = _y()
    assert metricas(y, y.astype(float))["brier"] == pytest.approx(0.0)


def test_la_tasa_base_reportada_es_la_prevalencia_real():
    y = _y(n_pos=25, n_neg=75)
    assert metricas(y, np.full(100, 0.25))["tasa_base"] == pytest.approx(0.25)


# --- Bootstrap -------------------------------------------------------------
def test_el_intervalo_contiene_la_estimacion_puntual():
    rng = np.random.default_rng(3)
    y = rng.binomial(1, 0.2, 800)
    prob = np.clip(y * 0.4 + rng.random(800) * 0.6, 0, 1)
    punto = pr_auc(y, prob)
    lo, hi = bootstrap_ic(y, prob, pr_auc, n_muestras=200, semilla=0)
    assert lo <= punto <= hi


def test_el_intervalo_se_angosta_con_mas_datos():
    """Es el punto: con pocos positivos el intervalo es enorme y hay que mostrarlo."""
    rng = np.random.default_rng(4)

    def ancho(n):
        y = rng.binomial(1, 0.2, n)
        prob = np.clip(y * 0.4 + rng.random(n) * 0.6, 0, 1)
        lo, hi = bootstrap_ic(y, prob, pr_auc, n_muestras=200, semilla=0)
        return hi - lo

    assert ancho(4000) < ancho(300)


def test_el_bootstrap_es_reproducible():
    rng = np.random.default_rng(5)
    y = rng.binomial(1, 0.2, 400)
    prob = rng.random(400)
    a = bootstrap_ic(y, prob, pr_auc, n_muestras=100, semilla=7)
    b = bootstrap_ic(y, prob, pr_auc, n_muestras=100, semilla=7)
    assert a == b
