# =============================================================================
# ml/dataset.py — Preparación de la matriz y split temporal.
#
# Acá se resuelve la SEGUNDA mitad del problema de leakage.
#
# `features.py` garantiza que ninguna feature mire el futuro. Este módulo
# garantiza que ningún FOLD lo haga: se entrena siempre con el pasado y se
# evalúa siempre en el futuro. Un `train_test_split(shuffle=True)` sobre datos
# de fútbol es entrenar con el resultado de mayo para predecir el de agosto.
#
# Y una decisión más: NO usamos un único test set. Con ~350 positivos por
# temporada, una sola evaluación tiene un intervalo de confianza tan ancho que
# no permite distinguir dos modelos. La validación de ORIGEN MÓVIL da una
# evaluación por temporada — siete en vez de una — sin romper el orden temporal.
# =============================================================================

import numpy as np
import pandas as pd

from ml.features import TARGET

# Columnas que identifican o describen el registro, pero no son información
# predictiva. Un id como feature es una invitación a memorizar.
METADATA = (
    "id_evento", "nombre_evento", "id_liga",
    "id_equipo_local", "id_equipo_visitante",
    "fecha_partido", "hora_partido", "estadio", "estado",
    "timestamp_extraccion", "fecha_extraccion",
    "temporada",           # se usa para el split, no como feature
    "equipo_local", "equipo_visitante",   # entran como tasa historica, no como dummies
    "arbitro",             # sin señal (ver ml/features.py) y solo existe en la Premier
    # La probabilidad del mercado viaja en la matriz para poder evaluarla como
    # BASELINE sobre los mismos folds, pero NO es una feature: si el modelo la
    # viera, aprendería a copiar al mercado y la comparación perdería sentido.
    "prob_mercado_over25",
)

# Las cuotas CRUDAS se excluyen por prefijo, no nombrándolas una por una.
#
# Estaban listadas a mano, y cuando se agregaron las cuotas de mercado a bronze
# para el modelo de goles se colaron como features del modelo de EXPULSIONES:
# nadie se acordó de actualizar la lista. Una lista que hay que mantener a mano
# es una lista que va a quedar desactualizada.
#
# La regla es simple: una cuota entra normalizada como probabilidad (prob_*,
# paridad) o no entra. Da igual qué casa la publique o cómo se llame.
PREFIJOS_EXCLUIDOS = ("cuota_",)

# Categóricas de verdad: solo la liga.
#
# Los nombres de equipo NO entran como categóricas. Con ~160 equipos, muchos
# presentes apenas un par de temporadas (ascensos y descensos), el modelo les
# ajustaba coeficientes enormes sobre 30-60 partidos: memorizaba en vez de
# aprender. Entran como `equipo_tasa_exp_local/visitante`, la tasa histórica
# suavizada que construye ml/features.py.
#
# La liga sí: son 5 valores, con miles de partidos cada uno, y es un rasgo
# estable en el tiempo (fiabilidad split-half +0.88).
CATEGORICAS = ("liga",)


def columnas_features(df: pd.DataFrame, target: str = TARGET) -> list[str]:
    """Devuelve las columnas que sí entran al modelo.

    `target` se parametriza porque el proyecto tiene dos modelos sobre la misma
    maquinaria: expulsiones y over 2.5 goles.
    """
    excluidas = set(METADATA) | {target}
    return [
        c for c in df.columns
        if c not in excluidas and not c.startswith(PREFIJOS_EXCLUIDOS)
    ]


def preparar_matriz(
    df: pd.DataFrame, target: str = TARGET
) -> tuple[pd.DataFrame, pd.Series]:
    """Separa la matriz de features del target y tipa las categóricas.

    Returns:
        (X, y) con el mismo índice, listos para `fit`.
    """
    X = df[columnas_features(df, target)].copy()
    for col in CATEGORICAS:
        if col in X.columns:
            X[col] = X[col].astype("category")
    y = df[target].astype(int)
    return X, y


def splits_origen_movil(
    df: pd.DataFrame, min_temporadas_train: int = 3
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Validación de origen móvil (expanding window) por temporada.

    Fold i: entrena con todas las temporadas anteriores, evalúa en la siguiente.

        fold 0: train [T0, T1, T2]              -> test T3
        fold 1: train [T0, T1, T2, T3]          -> test T4
        ...

    El train CRECE en cada fold, que es como funciona en producción: cada
    temporada nueva se entrena con toda la historia disponible hasta ese
    momento. Nunca al revés.

    Args:
        df: DataFrame con columna 'temporada'.
        min_temporadas_train: temporadas mínimas antes del primer fold.

    Returns:
        Lista de (indices_train, indices_test) posicionales.

    Raises:
        ValueError: si no hay temporadas suficientes para armar un fold.
    """
    temporadas = sorted(df["temporada"].dropna().unique())
    if len(temporadas) <= min_temporadas_train:
        raise ValueError(
            f"Hacen falta más de {min_temporadas_train} temporadas para armar un "
            f"fold; hay {len(temporadas)}: {temporadas}"
        )

    posiciones = np.arange(len(df))
    folds = []
    for corte in range(min_temporadas_train, len(temporadas)):
        es_train = df["temporada"].isin(temporadas[:corte]).values
        es_test  = (df["temporada"] == temporadas[corte]).values
        folds.append((posiciones[es_train], posiciones[es_test]))
    return folds
