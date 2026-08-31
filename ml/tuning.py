# =============================================================================
# ml/tuning.py — Busqueda ANIDADA y TEMPORAL de hiperparametros.
#
# POR QUE ANIDADA
# ---------------
# El error clasico: buscar hiperparametros mirando el mismo set con el que
# despues se reportan las metricas. Elegis la configuracion que mejor le va a
# ESE set y reportas ese numero como si fuera rendimiento fuera de muestra. Es
# leakage, es sutil, y es de lo mas comun en proyectos que "dieron bien".
#
# El anidamiento separa dos preguntas que no son la misma:
#
#   bucle INTERNO (solo sobre train)  -> que hiperparametros elijo?
#   bucle EXTERNO                     -> cuanto rinde ESE PROCEDIMIENTO en
#                                        datos que nadie miro?
#
# Lo que reporta el bucle externo no es "el rendimiento del mejor modelo":
# es el rendimiento de "buscar y despues predecir", que es lo que realmente
# vas a hacer en produccion.
#
# POR QUE NO GridSearchCV
# -----------------------
# Su cv por defecto mezcla el tiempo. Habria que pasarle splits temporales a
# mano, y a esa altura escribir el bucle explicito es mas claro que configurar
# el de sklearn: se ve exactamente que ve cada nivel.
# =============================================================================

import logging
from itertools import product
from typing import Callable

import numpy as np
import pandas as pd

from ml.dataset import preparar_matriz, splits_origen_movil
from ml.evaluation import metricas, pr_auc

logger = logging.getLogger(__name__)


def _combinaciones(grilla: dict) -> list[dict]:
    """Producto cartesiano de la grilla, como lista de diccionarios."""
    claves = list(grilla)
    return [dict(zip(claves, valores)) for valores in product(*(grilla[k] for k in claves))]


def buscar_hiperparametros(
    matriz: pd.DataFrame,
    fabrica: Callable,
    grilla: dict,
    min_temporadas_train: int = 2,
) -> dict:
    """Bucle INTERNO: elige hiperparametros usando solo los datos recibidos.

    Evalua cada combinacion con validacion de origen movil sobre `matriz` y
    devuelve la que maximiza el PR-AUC promedio.

    Quien llama es responsable de pasar SOLO datos de entrenamiento: esta
    funcion no sabe que hay despues, y es justamente asi como tiene que ser.

    Args:
        matriz: features + target, solo del tramo de entrenamiento.
        fabrica: callable que recibe los hiperparametros y devuelve un estimador.
        grilla: {nombre_parametro: [valores]}.
        min_temporadas_train: temporadas minimas del bucle interno.

    Returns:
        La combinacion ganadora.
    """
    X, y = preparar_matriz(matriz)
    folds = splits_origen_movil(matriz, min_temporadas_train)

    mejor_params, mejor_score = None, -np.inf
    for params in _combinaciones(grilla):
        scores = []
        for idx_train, idx_test in folds:
            modelo = fabrica(**params).fit(X.iloc[idx_train], y.iloc[idx_train])
            prob = modelo.predict_proba(X.iloc[idx_test])[:, 1]
            y_test = y.iloc[idx_test].values
            if len(np.unique(y_test)) < 2:
                continue
            scores.append(pr_auc(y_test, prob))

        promedio = float(np.mean(scores)) if scores else -np.inf
        if promedio > mejor_score:
            mejor_params, mejor_score = params, promedio

    logger.debug("Hiperparametros elegidos: %s (PR-AUC interno %.4f)", mejor_params, mejor_score)
    return mejor_params


def evaluar_con_tuning_anidado(
    matriz: pd.DataFrame,
    fabrica: Callable,
    grilla: dict,
    min_temporadas_train: int = 3,
    min_temporadas_interno: int = 2,
) -> pd.DataFrame:
    """Bucle EXTERNO: evalua el procedimiento completo de busqueda + prediccion.

    Para cada fold externo: busca hiperparametros usando UNICAMENTE las
    temporadas de entrenamiento de ese fold, reajusta con la combinacion
    ganadora y evalua en la temporada siguiente, que la busqueda nunca vio.

    Returns:
        Una fila por fold externo, con los hiperparametros elegidos y las
        metricas obtenidas.
    """
    import ml.tuning as este_modulo   # permite interceptar la busqueda en tests

    X, y = preparar_matriz(matriz)
    filas = []

    for i, (idx_train, idx_test) in enumerate(splits_origen_movil(matriz, min_temporadas_train)):
        matriz_train = matriz.iloc[idx_train]

        params = este_modulo.buscar_hiperparametros(
            matriz_train, fabrica, grilla, min_temporadas_interno
        )

        modelo = fabrica(**params).fit(X.iloc[idx_train], y.iloc[idx_train])
        prob = modelo.predict_proba(X.iloc[idx_test])[:, 1]
        y_test = y.iloc[idx_test].values

        fila = {
            "fold":             i,
            "temporada_test":   matriz.iloc[idx_test]["temporada"].iloc[0],
            "hiperparametros":  str(params),
            "n_train":          len(idx_train),
            "n_test":           len(idx_test),
            "n_positivos":      int(y_test.sum()),
        }
        fila.update(metricas(y_test, prob))
        filas.append(fila)
        logger.info("Fold %d (%s): %s | PR-AUC %.4f", i, fila["temporada_test"], params, fila["pr_auc"])

    return pd.DataFrame(filas)
