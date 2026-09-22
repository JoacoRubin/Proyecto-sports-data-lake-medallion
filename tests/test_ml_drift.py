# =============================================================================
# tests/test_ml_drift.py — ¿el modelo YA en producción sigue sirviendo?
#
# LA REGLA QUE SE PRUEBA (ver ml/drift.py)
# -----------------------------------------
# Marca drift si la métrica principal cae más de la tolerancia O si el Brier
# empeora más de la tolerancia -- cualquiera de las dos, a diferencia del
# quality gate de promoción que exige ambas condiciones para rechazar.
# =============================================================================

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from ml.drift import evaluar_drift_metricas, medir_drift


def _metricas(roc_auc=0.60, brier=0.20):
    return {"roc_auc": roc_auc, "pr_auc": 0.30, "log_loss": 0.65, "brier": brier}


# --- evaluar_drift_metricas: la regla pura -----------------------------------
def test_sin_drift_si_las_metricas_se_mantienen():
    v = evaluar_drift_metricas(
        _metricas(roc_auc=0.60, brier=0.20), _metricas(roc_auc=0.60, brier=0.20),
        metrica_principal="roc_auc", n_partidos_frescos=30,
    )
    assert v.evaluado
    assert not v.drift_detectado


def test_sin_drift_si_las_metricas_mejoran():
    v = evaluar_drift_metricas(
        _metricas(roc_auc=0.65, brier=0.18), _metricas(roc_auc=0.60, brier=0.20),
        metrica_principal="roc_auc", n_partidos_frescos=30,
    )
    assert not v.drift_detectado


def test_drift_si_la_metrica_principal_cae_mas_de_la_tolerancia():
    v = evaluar_drift_metricas(
        _metricas(roc_auc=0.50, brier=0.20), _metricas(roc_auc=0.60, brier=0.20),
        metrica_principal="roc_auc", n_partidos_frescos=30, tolerancia_ranking=0.05,
    )
    assert v.drift_detectado
    assert "roc_auc" in v.razon


def test_drift_si_la_calibracion_empeora_mas_de_la_tolerancia():
    v = evaluar_drift_metricas(
        _metricas(roc_auc=0.60, brier=0.35), _metricas(roc_auc=0.60, brier=0.20),
        metrica_principal="roc_auc", n_partidos_frescos=30, tolerancia_calibracion=0.01,
    )
    assert v.drift_detectado
    assert "brier" in v.razon


def test_drift_alcanza_con_una_sola_de_las_dos_condiciones():
    """A diferencia de promotion.py: acá no se compite contra un candidato,
    cualquier degradacion real es señal, no hace falta que fallen las dos."""
    v = evaluar_drift_metricas(
        _metricas(roc_auc=0.50, brier=0.20), _metricas(roc_auc=0.60, brier=0.20),
        metrica_principal="roc_auc", n_partidos_frescos=30,
        tolerancia_ranking=0.05, tolerancia_calibracion=0.01,
    )
    assert v.drift_detectado


# --- El borde de la tolerancia ------------------------------------------------
def test_sin_drift_en_el_limite_exacto_de_la_tolerancia():
    v = evaluar_drift_metricas(
        _metricas(roc_auc=0.55, brier=0.20), _metricas(roc_auc=0.60, brier=0.20),
        metrica_principal="roc_auc", n_partidos_frescos=30, tolerancia_ranking=0.05,
    )
    assert not v.drift_detectado


def test_drift_apenas_pasado_el_limite_de_la_tolerancia():
    v = evaluar_drift_metricas(
        _metricas(roc_auc=0.5499, brier=0.20), _metricas(roc_auc=0.60, brier=0.20),
        metrica_principal="roc_auc", n_partidos_frescos=30, tolerancia_ranking=0.05,
    )
    assert v.drift_detectado


# --- Trazabilidad -------------------------------------------------------------
def test_la_razon_es_un_string_no_vacio():
    v = evaluar_drift_metricas(
        _metricas(), _metricas(), metrica_principal="roc_auc", n_partidos_frescos=30,
    )
    assert isinstance(v.razon, str) and v.razon


def test_evaluado_es_true_cuando_hay_veredicto():
    v = evaluar_drift_metricas(
        _metricas(), _metricas(), metrica_principal="roc_auc", n_partidos_frescos=30,
    )
    assert v.evaluado


# --- medir_drift: el contrato sobre el DataFrame -----------------------------
def _matriz_sintetica(n, fecha, target_desde=0):
    """Filas con dos features linealmente separables por 'x1' y un target
    que 'x1' predice perfecto -- controla si el modelo acierta o no."""
    rng = np.random.default_rng(0)
    x1 = rng.normal(size=n)
    return pd.DataFrame({
        "x1": x1,
        "x2": rng.normal(size=n),
        "target": (x1 > target_desde).astype(int),
        "fecha_partido": [fecha] * n,
    })


@pytest.fixture
def modelo_y_metadata():
    """Un modelo real, entrenado sobre datos donde x1>0 predice el target."""
    train = _matriz_sintetica(200, fecha="2020-01-01")
    modelo = LogisticRegression().fit(train[["x1", "x2"]], train["target"])
    probs = modelo.predict_proba(train[["x1", "x2"]])[:, 1]
    from ml.evaluation import metricas as calc_metricas
    metadata = {
        "guardado_en": "2024-01-01T00:00:00+00:00",
        "features": ["x1", "x2"],
        "target": "target",
        "prior": 0.5,
        "metricas": calc_metricas(train["target"].to_numpy(), probs),
    }
    return modelo, metadata


def test_no_evaluado_si_no_hay_partidos_posteriores_a_la_promocion(modelo_y_metadata):
    modelo, metadata = modelo_y_metadata
    matriz = _matriz_sintetica(30, fecha="2020-06-01")  # todo ANTES de guardado_en
    v = medir_drift(modelo, metadata, matriz, metrica_principal="roc_auc", min_partidos=20)
    assert not v.evaluado
    assert v.n_partidos_frescos == 0


def test_no_evaluado_si_hay_pocos_partidos_frescos(modelo_y_metadata):
    modelo, metadata = modelo_y_metadata
    matriz = _matriz_sintetica(10, fecha="2025-01-01")  # despues, pero pocos
    v = medir_drift(modelo, metadata, matriz, metrica_principal="roc_auc", min_partidos=20)
    assert not v.evaluado
    assert v.n_partidos_frescos == 10


def test_sin_drift_si_los_partidos_frescos_siguen_el_mismo_patron(modelo_y_metadata):
    modelo, metadata = modelo_y_metadata
    matriz = _matriz_sintetica(50, fecha="2025-01-01")  # mismo patron x1>0
    v = medir_drift(modelo, metadata, matriz, metrica_principal="roc_auc", min_partidos=20)
    assert v.evaluado
    assert not v.drift_detectado


def test_drift_si_el_patron_se_invirtio(modelo_y_metadata):
    """El modelo aprendio 'x1>0 -> target=1'. Si en los partidos frescos el
    patron se invierte, el modelo predice mal sistematicamente: eso ES drift."""
    modelo, metadata = modelo_y_metadata
    matriz = _matriz_sintetica(50, fecha="2025-01-01")
    matriz["target"] = 1 - matriz["target"]  # invertido a proposito
    v = medir_drift(modelo, metadata, matriz, metrica_principal="roc_auc",
                     min_partidos=20, tolerancia_ranking=0.05)
    assert v.evaluado
    assert v.drift_detectado
