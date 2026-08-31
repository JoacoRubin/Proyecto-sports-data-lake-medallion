# =============================================================================
# ml/training.py — Los tres escalones del modelo.
#
# ESCALÓN 0 — Tasa base. No es un modelo: es LA VARA. Predice siempre la
#   prevalencia. Si el modelo copado no le gana, el modelo copado no sirve.
#   Saltearse este paso es cómo se termina festejando un 82% de accuracy.
#
# ESCALÓN 1 — Regresión logística. Interpretable y naturalmente bien
#   calibrada. Es la herramienta de diagnóstico: si acá aparece algo raro, el
#   problema está en los datos, no en el modelo.
#
# ESCALÓN 2 — HistGradientBoostingClassifier. El que va a producción.
#   Elegido sobre XGBoost/LightGBM por razones concretas de ESTE problema:
#   maneja NaN nativamente (las medias móviles son nulas en los primeros
#   partidos de cada equipo), soporta categóricas por dtype (~160 equipos sin
#   one-hot) y no agrega dependencias al build de Streamlit Cloud.
#
#   Con 18.011 filas la diferencia entre implementaciones de boosting entra
#   dentro del ruido. Lo que mueve la aguja son las features, no el algoritmo.
# =============================================================================

import logging
from typing import Callable

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.dataset import CATEGORICAS, preparar_matriz, splits_origen_movil
from ml.features import TARGET
from ml.evaluation import metricas

logger = logging.getLogger(__name__)


def modelo_tasa_base() -> DummyClassifier:
    """Escalón 0 — predice siempre la prevalencia del entrenamiento."""
    return DummyClassifier(strategy="prior")


def modelo_logistico() -> Pipeline:
    """Escalón 1 — logística con imputación, escalado y one-hot.

    A diferencia del boosting, la logística NO tolera NaN ni categóricas
    crudas: por eso necesita todo el preprocesamiento explícito. Que haga
    falta este andamiaje es, en sí, parte de por qué el escalón 2 es otro.
    """
    numericas = Pipeline([
        ("imputar", SimpleImputer(strategy="median")),
        ("escalar", StandardScaler()),
    ])
    categoricas = Pipeline([
        ("imputar", SimpleImputer(strategy="most_frequent")),
        ("onehot",  OneHotEncoder(handle_unknown="ignore", min_frequency=20)),
    ])
    preproceso = ColumnTransformer(
        transformers=[
            ("cat", categoricas, [c for c in CATEGORICAS]),
        ],
        remainder=numericas,
        verbose_feature_names_out=False,
    )
    # SIN class_weight="balanced", y es deliberado.
    #
    # Es el reflejo automático ante un target desbalanceado, y acá era
    # activamente dañino: re-pesa las clases como si fueran 50/50, así que
    # infla las probabilidades. Medido sobre los 7 folds:
    #
    #   con balanced : prob media 0.466 | log-loss 0.6683 | Brier 0.2382
    #   sin balanced : prob media 0.172 | log-loss 0.4563 | Brier 0.1408
    #   (la tasa base real es 0.168)
    #
    # Y el PR-AUC quedó igual (0.1993 vs 0.1997): no compraba nada de ranking
    # a cambio de romper la calibración. Solo tiene sentido si se va a tomar
    # una decisión dura con un umbral, que no es el caso: queremos la
    # probabilidad.
    return Pipeline([
        ("preproceso", preproceso),
        ("modelo", LogisticRegression(max_iter=2000)),
    ])


def modelo_hgb() -> HistGradientBoostingClassifier:
    """Escalón 2 — el modelo de producción.

    `categorical_features="from_dtype"` toma las columnas con dtype category
    directamente; `preparar_matriz` ya las dejó tipadas.

    Los hiperparámetros son deliberadamente conservadores: con 3.122 positivos
    y una señal débil (correlaciones ~0.07), un modelo profundo memoriza en
    lugar de aprender.
    """
    return HistGradientBoostingClassifier(
        categorical_features="from_dtype",
        max_depth=4,
        max_iter=300,
        learning_rate=0.05,
        min_samples_leaf=50,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=42,
    )


MODELOS: dict[str, Callable] = {
    "tasa_base": modelo_tasa_base,
    "logistico": modelo_logistico,
    "hgb":       modelo_hgb,
}


def evaluar_por_folds(
    df: pd.DataFrame,
    fabrica_modelo: Callable,
    min_temporadas_train: int = 3,
    target: str = TARGET,
) -> pd.DataFrame:
    """Entrena y evalúa el modelo con validación de origen móvil.

    Una fila por fold: cada temporada se evalúa con un modelo entrenado
    únicamente con las temporadas anteriores.

    Args:
        df: matriz de features (salida de `construir_features`).
        fabrica_modelo: función sin argumentos que devuelve un estimador nuevo.
        min_temporadas_train: temporadas mínimas antes del primer fold.

    Returns:
        DataFrame con las métricas de cada fold.
    """
    X, y = preparar_matriz(df, target)
    filas = []

    for i, (idx_train, idx_test) in enumerate(
        splits_origen_movil(df, min_temporadas_train)
    ):
        X_tr, y_tr = X.iloc[idx_train], y.iloc[idx_train]
        X_te, y_te = X.iloc[idx_test],  y.iloc[idx_test]

        modelo = fabrica_modelo().fit(X_tr, y_tr)
        prob = modelo.predict_proba(X_te)[:, 1]

        fila = {
            "fold":            i,
            "temporada_test":  df.iloc[idx_test]["temporada"].iloc[0],
            "n_train":         len(idx_train),
            "n_test":          len(idx_test),
            "n_positivos":     int(y_te.sum()),
        }
        fila.update(metricas(y_te.values, prob))
        filas.append(fila)

    return pd.DataFrame(filas)


def comparar_modelos(
    df: pd.DataFrame, min_temporadas_train: int = 3, target: str = TARGET
) -> pd.DataFrame:
    """Corre los tres escalones y devuelve el promedio de métricas por modelo."""
    resultados = []
    for nombre, fabrica in MODELOS.items():
        logger.info("Evaluando '%s'...", nombre)
        folds = evaluar_por_folds(df, fabrica, min_temporadas_train, target)
        resumen = {
            "modelo":      nombre,
            "folds":       len(folds),
            "pr_auc":      folds["pr_auc"].mean(),
            "lift_pr_auc": folds["lift_pr_auc"].mean(),
            "roc_auc":     folds["roc_auc"].mean(),
            "log_loss":    folds["log_loss"].mean(),
            "brier":       folds["brier"].mean(),
        }
        resultados.append(resumen)
    return pd.DataFrame(resultados)
