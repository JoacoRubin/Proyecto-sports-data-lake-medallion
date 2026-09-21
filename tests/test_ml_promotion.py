# =============================================================================
# tests/test_ml_promotion.py — El quality gate: ¿el candidato reemplaza al
# modelo en producción?
#
# LA REGLA QUE SE PRUEBA (ver ml/promotion.py)
# -----------------------------------------------
# Promueve si: (1) la métrica principal mejora Y (2) la calibración (Brier)
# no empeora más de la tolerancia. Cualquier otra combinación rechaza.
# =============================================================================

from ml.promotion import evaluar_promocion


def _metricas(pr_auc=0.20, brier=0.14):
    return {"pr_auc": pr_auc, "roc_auc": 0.58, "log_loss": 0.45, "brier": brier}


# --- Sin modelo actual: primer entrenamiento --------------------------------
def test_sin_modelo_actual_promueve_siempre():
    decision = evaluar_promocion(_metricas(), None, metrica_principal="pr_auc")
    assert decision.promovido
    assert decision.valor_actual is None


# --- Mejora limpia ------------------------------------------------------------
def test_promueve_si_mejora_la_metrica_principal_y_la_calibracion_no_empeora():
    candidato = _metricas(pr_auc=0.22, brier=0.138)
    actual = _metricas(pr_auc=0.21, brier=0.140)
    decision = evaluar_promocion(candidato, actual, metrica_principal="pr_auc")
    assert decision.promovido


def test_promueve_si_la_calibracion_tambien_mejora():
    candidato = _metricas(pr_auc=0.22, brier=0.130)
    actual = _metricas(pr_auc=0.21, brier=0.140)
    decision = evaluar_promocion(candidato, actual, metrica_principal="pr_auc")
    assert decision.promovido


# --- Rechazos -------------------------------------------------------------
def test_rechaza_si_la_metrica_principal_no_mejora():
    candidato = _metricas(pr_auc=0.20, brier=0.130)
    actual = _metricas(pr_auc=0.21, brier=0.140)
    decision = evaluar_promocion(candidato, actual, metrica_principal="pr_auc")
    assert not decision.promovido


def test_rechaza_en_empate_exacto():
    """Un empate no es una mejora: '>' estricto, no '>='."""
    candidato = _metricas(pr_auc=0.21, brier=0.140)
    actual = _metricas(pr_auc=0.21, brier=0.140)
    decision = evaluar_promocion(candidato, actual, metrica_principal="pr_auc")
    assert not decision.promovido


def test_rechaza_si_mejora_el_ranking_pero_la_calibracion_empeora_de_mas():
    """El caso class_weight='balanced' documentado en ml/training.py: gana
    PR-AUC pero rompe la calibracion. Esto es lo que el gate existe para
    evitar."""
    candidato = _metricas(pr_auc=0.22, brier=0.20)
    actual = _metricas(pr_auc=0.21, brier=0.140)
    decision = evaluar_promocion(candidato, actual, metrica_principal="pr_auc", tolerancia_calibracion=0.01)
    assert not decision.promovido


# --- El borde de la tolerancia ----------------------------------------------
def test_promueve_en_el_limite_exacto_de_la_tolerancia():
    candidato = _metricas(pr_auc=0.22, brier=0.150)
    actual = _metricas(pr_auc=0.21, brier=0.140)
    decision = evaluar_promocion(candidato, actual, metrica_principal="pr_auc", tolerancia_calibracion=0.01)
    assert decision.promovido


def test_rechaza_apenas_pasado_el_limite_de_la_tolerancia():
    candidato = _metricas(pr_auc=0.22, brier=0.1501)
    actual = _metricas(pr_auc=0.21, brier=0.140)
    decision = evaluar_promocion(candidato, actual, metrica_principal="pr_auc", tolerancia_calibracion=0.01)
    assert not decision.promovido


# --- Parametrizable por modelo (expulsiones usa pr_auc, goles usa roc_auc) --
def test_la_metrica_principal_es_configurable():
    candidato = {"pr_auc": 0.10, "roc_auc": 0.62, "brier": 0.20}
    actual = {"pr_auc": 0.30, "roc_auc": 0.60, "brier": 0.20}
    # Por pr_auc perdería; por roc_auc gana. El gate debe usar la que se le pide.
    decision = evaluar_promocion(candidato, actual, metrica_principal="roc_auc")
    assert decision.promovido


# --- Trazabilidad: la razon queda legible para auditar despues --------------
def test_la_razon_es_un_string_no_vacio():
    decision = evaluar_promocion(_metricas(0.22, 0.13), _metricas(0.21, 0.14), metrica_principal="pr_auc")
    assert isinstance(decision.razon, str) and decision.razon
