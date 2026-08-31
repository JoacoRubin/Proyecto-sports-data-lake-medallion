# =============================================================================
# ml/evaluation.py — Métricas para un target desbalanceado.
#
# ACCURACY NO SE REPORTA. A propósito, y con una función que la omite.
#
# La tasa base de expulsión es 17,6%. Un modelo que prediga siempre "no" saca
# 82,4% de accuracy sin haber aprendido nada. En un problema desbalanceado,
# accuracy no mide capacidad predictiva: mide la prevalencia.
#
# Lo que sí reportamos:
#   - PR-AUC : su baseline ES la prevalencia, así que el lift contra la tasa
#              base es directamente interpretable.
#   - Log-loss y Brier: calidad de la PROBABILIDAD. Acá importa más que la
#     clasificación — un "35% de chances de expulsión" que se cumple 35 veces
#     de cada 100 es más útil que un sí/no con un umbral inventado.
#   - Bootstrap: con ~350 positivos por fold, la estimación puntual sola no
#     alcanza para afirmar que un modelo es mejor que otro.
# =============================================================================

from typing import Callable

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Área bajo la curva precision-recall (average precision)."""
    return float(average_precision_score(y_true, y_prob))


def metricas(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Paquete de métricas del modelo. Sin accuracy, por diseño.

    `lift_pr_auc` es la lectura más importante: cuántas veces mejor que
    predecir la prevalencia. Un lift de 1.0 significa que el modelo no aporta
    absolutamente nada, por más alto que se vea el PR-AUC en términos
    absolutos.
    """
    y_true = np.asarray(y_true)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-15, 1 - 1e-15)
    tasa_base = float(y_true.mean())

    resultado = {
        "tasa_base": tasa_base,
        "pr_auc":    pr_auc(y_true, y_prob),
        "log_loss":  float(log_loss(y_true, y_prob, labels=[0, 1])),
        "brier":     float(brier_score_loss(y_true, y_prob)),
    }
    resultado["lift_pr_auc"] = (
        resultado["pr_auc"] / tasa_base if tasa_base > 0 else float("nan")
    )
    # ROC-AUC no está definido si hay una sola clase en el fold.
    resultado["roc_auc"] = (
        float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")
    )
    return resultado


def bootstrap_ic(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metrica: Callable[[np.ndarray, np.ndarray], float],
    n_muestras: int = 1000,
    alfa: float = 0.05,
    semilla: int = 0,
) -> tuple[float, float]:
    """Intervalo de confianza por bootstrap de una métrica.

    Remuestrea con reemplazo y devuelve los percentiles. Es lo que convierte
    "PR-AUC 0.21" en "PR-AUC 0.21 (IC95 0.17-0.26)" — y esa segunda forma es
    la única que permite decir si un modelo le gana a otro o si la diferencia
    entra dentro del ruido.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    rng = np.random.default_rng(semilla)
    n = len(y_true)

    valores = []
    for _ in range(n_muestras):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue                      # remuestreo degenerado: se descarta
        valores.append(metrica(y_true[idx], y_prob[idx]))

    if not valores:
        return (float("nan"), float("nan"))
    return (
        float(np.percentile(valores, 100 * alfa / 2)),
        float(np.percentile(valores, 100 * (1 - alfa / 2))),
    )


def curva_calibracion(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> list[dict[str, float]]:
    """Compara la probabilidad predicha con la frecuencia observada por bin.

    Es el diagnóstico honesto de un modelo probabilístico: si en el bin del
    30% la tasa real es 12%, el modelo está mintiendo aunque su PR-AUC sea
    bueno.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    bordes = np.linspace(0, 1, n_bins + 1)
    bins = np.clip(np.digitize(y_prob, bordes[1:-1]), 0, n_bins - 1)

    filas = []
    for b in range(n_bins):
        mascara = bins == b
        if not mascara.any():
            continue
        filas.append({
            "bin":            b,
            "n":              int(mascara.sum()),
            "prob_predicha":  float(y_prob[mascara].mean()),
            "tasa_observada": float(y_true[mascara].mean()),
        })
    return filas
