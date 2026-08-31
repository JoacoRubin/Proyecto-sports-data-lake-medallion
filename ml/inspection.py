# =============================================================================
# ml/inspection.py — Importancia de features por permutación.
#
# POR QUÉ NO ALCANZA CON LEER COEFICIENTES
# ----------------------------------------
# Un coeficiente dice cuánto pesa una variable DENTRO del modelo ajustado. No
# dice si esa variable aporta FUERA de muestra, que es la única pregunta que
# importa.
#
# Este proyecto tiene el contraejemplo perfecto: cuando los equipos entraban
# como dummies, los coeficientes más grandes eran de clubes con 30-60 partidos
# (Hannover, Bastia, Nimes). Peso enorme en el modelo, valor predictivo nulo.
# Leer coeficientes lo insinuaba; permutar lo demuestra.
#
# La permutación mide lo único operativo: cuánto EMPEORA la métrica en datos
# no vistos si rompo esa columna. Se mide sobre average precision, no sobre
# accuracy — con una tasa base del 17%, permutar contra accuracy no detecta
# nada porque el modelo sigue acertando el 83% dijera lo que dijera.
# =============================================================================

import pandas as pd
from sklearn.inspection import permutation_importance

_METRICA = "average_precision"


def importancia_por_permutacion(
    modelo,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeticiones: int = 20,
    semilla: int = 0,
) -> pd.DataFrame:
    """Importancia de cada feature medida por caída de PR-AUC al permutarla.

    Se calcula sobre datos NO usados para entrenar: permutar sobre el propio
    train mide memorización, no capacidad predictiva.

    Args:
        modelo: estimador ya entrenado.
        X, y: datos de evaluación (una fold de test).
        n_repeticiones: permutaciones por feature. Más repeticiones, menos ruido.
        semilla: reproducibilidad.

    Returns:
        DataFrame con feature, importancia y desvio, de mayor a menor.
        El desvío es imprescindible: sin él no se sabe si la diferencia entre
        dos features es real o es ruido de la permutación.
    """
    resultado = permutation_importance(
        modelo, X, y,
        scoring=_METRICA,
        n_repeats=n_repeticiones,
        random_state=semilla,
        n_jobs=1,
    )

    tabla = (
        pd.DataFrame({
            "feature":     list(X.columns),
            "importancia": resultado.importances_mean,
            "desvio":      resultado.importances_std,
        })
        .sort_values("importancia", ascending=False)
        .reset_index(drop=True)
    )
    tabla.attrs["metrica"] = _METRICA
    return tabla
