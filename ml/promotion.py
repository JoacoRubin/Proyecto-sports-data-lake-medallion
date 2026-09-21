# =============================================================================
# ml/promotion.py — Quality gate: ¿el candidato reemplaza al modelo en producción?
#
# POR QUÉ ESTO Y NO EL MODEL REGISTRY DE MLflow
# ------------------------------------------------
# Mismo argumento que ml/registry.py (leelo primero): MLflow resuelve "cuál
# versión es la de producción" con un servidor y estados (Staging/Production).
# Ya tenemos esa respuesta sin infraestructura nueva — es la versión más
# reciente en `dir_modelos` — así que promover un modelo es, literalmente,
# decidir si `pipelines/ml_pipeline` llama a `guardar_modelo` o no.
#
# LA REGLA
# --------
# Sin modelo en producción (primer entrenamiento): promueve siempre, no hay
# nada contra qué comparar.
#
# Con modelo en producción, promueve solo si:
#   1. metrica_principal(candidato) > metrica_principal(actual), Y
#   2. brier(candidato) no empeora más de `tolerancia_calibracion`
#
# La condición (2) existe porque ganar ranking perdiendo calibración es un
# intercambio real y medido en este mismo proyecto: `ml/training.py` documenta
# el caso de `class_weight="balanced"`, que mejoraba el PR-AUC (0.1997 vs
# 0.1993) pero rompía la calibración (log-loss 0.6683 vs 0.4563). Sin el
# chequeo de calibración, ese modelo hubiera pasado el gate.
#
# El 0.01 de tolerancia es un default deliberadamente conservador, NO una
# cifra medida con bootstrap como el resto de las decisiones de este
# proyecto — todavía no existe esa medición para el Brier entre folds. Es
# ajustable vía `config.ML_TOLERANCIA_CALIBRACION`; si alguna vez se mide la
# variabilidad real, este número debería salir de ahí en vez de ser un
# supuesto.
# =============================================================================

from dataclasses import dataclass


@dataclass(frozen=True)
class DecisionPromocion:
    """El veredicto del gate, con los números que lo sostienen — para que
    quede auditable por qué un candidato se aceptó o se rechazó."""
    promovido: bool
    razon: str
    metrica_principal: str
    valor_candidato: float
    valor_actual: float | None
    metrica_calibracion: str
    valor_calibracion_candidato: float
    valor_calibracion_actual: float | None


def evaluar_promocion(
    metricas_candidato: dict,
    metricas_actual: dict | None,
    metrica_principal: str,
    metrica_calibracion: str = "brier",
    tolerancia_calibracion: float = 0.01,
) -> DecisionPromocion:
    """Decide si un modelo candidato reemplaza al modelo en producción.

    Args:
        metricas_candidato: `metadata["metricas"]` del modelo recién entrenado.
        metricas_actual: `metadata["metricas"]` del modelo hoy en producción,
            o `None` si todavía no hay ninguno (primer entrenamiento).
        metrica_principal: la métrica de ranking a exigir mejora ("pr_auc"
            para expulsiones, "roc_auc" para goles — mismo criterio que
            `app.py::_seccion_modelo`).
        metrica_calibracion: métrica donde "menor es mejor". Brier por
            defecto: es la que documenta el caso `class_weight="balanced"`.
        tolerancia_calibracion: cuánto puede empeorar la calibración sin
            bloquear la promoción.
    """
    valor_candidato = metricas_candidato[metrica_principal]
    calibracion_candidato = metricas_candidato[metrica_calibracion]

    if metricas_actual is None:
        return DecisionPromocion(
            promovido=True,
            razon="No hay modelo en producción todavía: primer entrenamiento.",
            metrica_principal=metrica_principal,
            valor_candidato=valor_candidato,
            valor_actual=None,
            metrica_calibracion=metrica_calibracion,
            valor_calibracion_candidato=calibracion_candidato,
            valor_calibracion_actual=None,
        )

    valor_actual = metricas_actual[metrica_principal]
    calibracion_actual = metricas_actual[metrica_calibracion]

    mejora_ranking = valor_candidato > valor_actual
    calibracion_aceptable = calibracion_candidato <= calibracion_actual + tolerancia_calibracion

    if mejora_ranking and calibracion_aceptable:
        promovido = True
        razon = (
            f"{metrica_principal} {valor_candidato:.4f} > {valor_actual:.4f} y "
            f"{metrica_calibracion} {calibracion_candidato:.4f} no empeora más de "
            f"{tolerancia_calibracion} (actual {calibracion_actual:.4f})."
        )
    elif not mejora_ranking:
        promovido = False
        razon = f"{metrica_principal} {valor_candidato:.4f} no supera al actual {valor_actual:.4f}."
    else:
        promovido = False
        razon = (
            f"{metrica_principal} mejora ({valor_candidato:.4f} > {valor_actual:.4f}) pero "
            f"{metrica_calibracion} empeora demasiado: {calibracion_candidato:.4f} vs "
            f"{calibracion_actual:.4f} (tolerancia {tolerancia_calibracion})."
        )

    return DecisionPromocion(
        promovido=promovido,
        razon=razon,
        metrica_principal=metrica_principal,
        valor_candidato=valor_candidato,
        valor_actual=valor_actual,
        metrica_calibracion=metrica_calibracion,
        valor_calibracion_candidato=calibracion_candidato,
        valor_calibracion_actual=calibracion_actual,
    )
