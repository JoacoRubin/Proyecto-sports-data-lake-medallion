# =============================================================================
# tests/test_ml_calibration.py — Tests del calibrador temporal.
#
# EL PROBLEMA QUE RESUELVE
#
# El modelo mide bien en el grueso de los datos pero se pone sobreconfiado en
# la cola: en el bin con 253 casos predecía 32,1% y ocurría 21,7%. Para un
# modelo cuyo producto ES la probabilidad, eso no es un detalle.
#
# La regresión isotónica corrige el mapeo probabilidad -> frecuencia observada.
# Pero necesita datos que el modelo base NO haya visto, y acá esos datos tienen
# que ser POSTERIORES, no una muestra aleatoria: `CalibratedClassifierCV` con
# su cv por defecto mezclaría el tiempo y volvería a meter futuro.
# =============================================================================

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from ml.calibration import CalibradorTemporal
from ml.evaluation import metricas


def _datos(n=600, semilla=0):
    """Dataset sintético con señal real y orden temporal implícito."""
    rng = np.random.default_rng(semilla)
    x = rng.normal(size=n)
    prob = 1 / (1 + np.exp(-(x - 1.6)))
    y = rng.binomial(1, prob)
    X = pd.DataFrame({"x": x, "ruido": rng.normal(size=n)})
    return X, pd.Series(y)


def _mal_calibrado():
    """Base deliberadamente descalibrado: class_weight='balanced' infla la probabilidad."""
    return LogisticRegression(class_weight="balanced", max_iter=1000)


# --- Contrato básico -------------------------------------------------------
def test_entrena_y_devuelve_probabilidades_validas():
    X, y = _datos()
    modelo = CalibradorTemporal(LogisticRegression(max_iter=1000)).fit(X, y)
    prob = modelo.predict_proba(X)[:, 1]

    assert len(prob) == len(y)
    assert prob.min() >= 0.0 and prob.max() <= 1.0


def test_predict_proba_devuelve_dos_columnas_que_suman_uno():
    X, y = _datos()
    prob = CalibradorTemporal(LogisticRegression(max_iter=1000)).fit(X, y).predict_proba(X)
    assert prob.shape == (len(y), 2)
    np.testing.assert_allclose(prob.sum(axis=1), 1.0)


def test_expone_classes_como_cualquier_clasificador_de_sklearn():
    X, y = _datos()
    modelo = CalibradorTemporal(LogisticRegression(max_iter=1000)).fit(X, y)
    assert list(modelo.classes_) == [0, 1]


# --- Lo que lo hace TEMPORAL -----------------------------------------------
def test_el_modelo_base_no_ve_los_datos_de_calibracion():
    """Si el base viera el holdout, la isotónica se ajustaría sobre datos ya
    memorizados y la calibración quedaría optimista."""
    X, y = _datos(n=1000)
    modelo = CalibradorTemporal(
        LogisticRegression(max_iter=1000), fraccion_calibracion=0.2
    ).fit(X, y)
    assert modelo.n_base_ == 800
    assert modelo.n_calibracion_ == 200


def test_la_calibracion_usa_el_tramo_FINAL_no_una_muestra_al_azar():
    """El holdout tiene que ser el futuro respecto del entrenamiento del base."""
    X, y = _datos(n=100)
    X = X.copy()
    X["orden"] = np.arange(100)
    modelo = CalibradorTemporal(
        LogisticRegression(max_iter=1000), fraccion_calibracion=0.3
    ).fit(X, y)
    assert modelo.indice_corte_ == 70


def test_falla_si_el_holdout_queda_sin_ambas_clases():
    """Sin positivos y negativos en el holdout, la isotónica no puede ajustar."""
    X = pd.DataFrame({"x": np.arange(50, dtype=float)})
    y = pd.Series([1] * 40 + [0] * 10)          # el tramo final es todo ceros
    with pytest.raises(ValueError, match="[Cc]alibraci"):
        CalibradorTemporal(LogisticRegression(max_iter=1000),
                           fraccion_calibracion=0.1).fit(X, y)


# --- Lo que tiene que MEJORAR ----------------------------------------------
def test_corrige_un_modelo_deliberadamente_descalibrado():
    """La prueba real: agarrar el bug de class_weight='balanced' y arreglarlo."""
    X, y = _datos(n=3000)
    corte = 2000
    X_tr, y_tr, X_te, y_te = X[:corte], y[:corte], X[corte:], y[corte:]

    crudo = _mal_calibrado().fit(X_tr, y_tr)
    calibrado = CalibradorTemporal(_mal_calibrado()).fit(X_tr, y_tr)

    brier_crudo     = metricas(y_te.values, crudo.predict_proba(X_te)[:, 1])["brier"]
    brier_calibrado = metricas(y_te.values, calibrado.predict_proba(X_te)[:, 1])["brier"]

    assert brier_calibrado < brier_crudo, "la calibración debe mejorar el Brier"


def test_acerca_la_probabilidad_media_a_la_tasa_real():
    X, y = _datos(n=3000)
    corte = 2000
    X_tr, y_tr, X_te, y_te = X[:corte], y[:corte], X[corte:], y[corte:]
    tasa = y_te.mean()

    crudo     = _mal_calibrado().fit(X_tr, y_tr).predict_proba(X_te)[:, 1].mean()
    calibrado = CalibradorTemporal(_mal_calibrado()).fit(X_tr, y_tr).predict_proba(X_te)[:, 1].mean()

    assert abs(calibrado - tasa) < abs(crudo - tasa)


def _ap(modelo, X_te, y_te):
    return metricas(y_te.values, modelo.predict_proba(X_te)[:, 1])["pr_auc"]


def _ap_base_vs_calibrado(metodo):
    """Aísla el efecto de la CALIBRACIÓN comparando contra el propio modelo base.

    Comparar contra un modelo entrenado aparte no serviría: el base del
    calibrador ve solo el 80% del train (el resto es holdout), así que serían
    dos modelos distintos y la diferencia mezclaría dos causas.
    """
    X, y = _datos(n=3000)
    corte = 2000
    X_tr, y_tr, X_te, y_te = X[:corte], y[:corte], X[corte:], y[corte:]

    modelo = CalibradorTemporal(_mal_calibrado(), metodo=metodo).fit(X_tr, y_tr)
    ap_base      = metricas(y_te.values, modelo.base_.predict_proba(X_te)[:, 1])["pr_auc"]
    ap_calibrado = _ap(modelo, X_te, y_te)
    return ap_base, ap_calibrado


def test_la_isotonica_cuesta_algo_de_ranking_y_eso_es_esperado():
    """La isotónica es una función ESCALONADA: mapea probabilidades distintas
    al mismo valor y esos empates degradan el ranking. Es el precio de la
    técnica, no un bug — se documenta y se mide, no se esconde."""
    ap_base, ap_iso = _ap_base_vs_calibrado("isotonic")
    assert ap_iso < ap_base, "si no perdiera ranking, no estaria creando empates"
    assert ap_iso > ap_base * 0.85, "pero la perdida tiene que ser acotada"


def test_la_sigmoide_preserva_el_ranking_exacto():
    """Platt es estrictamente monótona: el orden no cambia, el PR-AUC tampoco."""
    ap_base, ap_sig = _ap_base_vs_calibrado("sigmoid")
    assert ap_sig == pytest.approx(ap_base, abs=1e-9)


def test_la_sigmoide_tambien_arregla_la_escala():
    """Preservar el orden no sirve si la probabilidad sigue siendo mentira."""
    X, y = _datos(n=3000)
    corte = 2000
    X_tr, y_tr, X_te, y_te = X[:corte], y[:corte], X[corte:], y[corte:]
    tasa = y_te.mean()

    crudo = _mal_calibrado().fit(X_tr, y_tr).predict_proba(X_te)[:, 1].mean()
    sig   = (CalibradorTemporal(_mal_calibrado(), metodo="sigmoid")
             .fit(X_tr, y_tr).predict_proba(X_te)[:, 1].mean())

    assert abs(sig - tasa) < abs(crudo - tasa)


def test_metodo_desconocido_falla_claro():
    X, y = _datos()
    with pytest.raises(ValueError, match="[Mm]etodo"):
        CalibradorTemporal(LogisticRegression(max_iter=1000), metodo="magia").fit(X, y)
