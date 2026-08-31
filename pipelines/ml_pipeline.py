# =============================================================================
# pipelines/ml_pipeline.py — Entrenamiento y registro de los modelos.
#
# DÓNDE ENCAJA EN LA ARQUITECTURA
# -------------------------------
# ML no es una capa nueva del medallion: es un CONSUMIDOR de los datos ya
# procesados, igual que el dashboard. Lee la tabla bronze de football-data,
# construye features, evalúa y deja tres cosas:
#
#   - el modelo entrenado en data/models/ (joblib + ficha técnica)
#   - una fila por fold en la tabla Delta gold de experimentos
#   - una fila por feature en la tabla Delta gold de importancias
#
# DOS MODELOS, UN SOLO RUNNER
# ---------------------------
# `expulsiones` y `goles` comparten toda la maquinaria: split de origen móvil,
# los tres escalones, métricas, registro. Lo único que cambia es qué se predice
# y con qué features. Por eso hay un runner genérico y dos configuraciones,
# en vez de dos pipelines copiados.
#
# El de goles además compara contra el MERCADO DE APUESTAS, que es una vara
# adversaria y no pasiva. Ver ml/features_goles.py.
# =============================================================================

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from config import (
    DIR_EXPERIMENTOS_GOLD,
    DIR_IMPORTANCIAS_GOLD,
    DIR_MODELOS,
    DIR_PARTIDOS_FOOTBALLDATA_BRONZE,
    ML_MIN_TEMPORADAS_TRAIN,
    ML_MODELO_PRODUCCION,
    NOMBRE_MODELO_EXPULSIONES,
    NOMBRE_MODELO_GOLES,
)
from ml.dataset import columnas_features, preparar_matriz, splits_origen_movil
from ml.evaluation import metricas
from ml.features import TARGET, agregar_target, construir_features
from ml.features_goles import (
    TARGET_GOLES,
    agregar_target_goles,
    construir_features_goles,
)
from ml.inspection import importancia_por_permutacion
from ml.registry import guardar_modelo, registrar_experimento, registrar_importancias
from ml.training import MODELOS, evaluar_por_folds
from quality.contracts import validar_matriz_ml
from utils.delta import leer_tabla_delta

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConfiguracionModelo:
    """Todo lo que distingue a un modelo del otro."""
    nombre: str
    target: str
    construir_features: Callable
    agregar_target: Callable
    validar_contrato: Callable | None = None
    baseline_extra: Callable | None = None   # ej. el mercado, en el de goles


def _prior_de_entrenamiento(df: pd.DataFrame, cfg: ConfiguracionModelo, min_temporadas: int) -> float:
    """Tasa base calculada SOLO con las temporadas de entrenamiento.

    Es el valor hacia el que se suavizan las tasas históricas. Calcularlo sobre
    todo el dataset metería información del futuro en las features: leakage por
    la puerta de atrás.
    """
    temporadas = sorted(df["temporada"].dropna().unique())
    train = df[df["temporada"].isin(temporadas[:min_temporadas])]
    return float(cfg.agregar_target(train)[cfg.target].mean())


def _baseline_mercado(matriz: pd.DataFrame, min_temporadas: int) -> dict | None:
    """Evalúa la probabilidad implícita del mercado sobre los mismos folds.

    Se lee de la propia matriz de features y no de un DataFrame aparte: alinear
    dos DataFrames ordenados por separado es cómo se cuela un desalineamiento
    silencioso que arruina la comparación.
    """
    if "prob_mercado_over25" not in matriz.columns:
        return None

    y = matriz[TARGET_GOLES].astype(int)
    ys, ps = [], []
    for _, idx_test in splits_origen_movil(matriz, min_temporadas):
        sub = matriz.iloc[idx_test]
        con_cuota = sub["prob_mercado_over25"].notna()
        ps.append(sub.loc[con_cuota, "prob_mercado_over25"].values)
        ys.append(y.iloc[idx_test][con_cuota.values].values)

    Y, P = np.concatenate(ys), np.concatenate(ps)
    resultado = {"modelo": "mercado", "folds": len(ys)}
    resultado.update({k: v for k, v in metricas(Y, P).items()})
    return resultado


def ejecutar_modelo(
    cfg: ConfiguracionModelo, min_temporadas_train: int | None = None
) -> pd.DataFrame:
    """Entrena, evalúa y registra un modelo. Devuelve el resumen por modelo."""
    min_temporadas = min_temporadas_train or ML_MIN_TEMPORADAS_TRAIN

    logger.info("=" * 62)
    logger.info("  Capa ML - Modelo de %s", cfg.nombre)
    logger.info("=" * 62)

    logger.info("[1/6] Leyendo partidos desde bronze (football-data)...")
    df = leer_tabla_delta(DIR_PARTIDOS_FOOTBALLDATA_BRONZE)
    logger.info("      Partidos: %d", len(df))

    logger.info("[2/6] Construyendo features (sin leakage)...")
    prior = _prior_de_entrenamiento(df, cfg, min_temporadas)
    matriz = cfg.construir_features(df, prior=prior)
    features = columnas_features(matriz, cfg.target)
    logger.info("      Prior de entrenamiento: %.4f | features: %d", prior, len(features))

    if cfg.validar_contrato:
        logger.info("[2.5/6] Validando el contrato de la matriz de features...")
        cfg.validar_contrato(matriz)
        logger.info("      Contrato OK (sin columnas del propio partido).")

    logger.info("[3/6] Evaluando con validacion de origen movil...")
    resumen, resultados_por_modelo = [], {}

    # El baseline adversario va PRIMERO: es la vara contra la que se lee todo.
    if cfg.baseline_extra:
        vara = cfg.baseline_extra(matriz, min_temporadas)
        if vara:
            resumen.append(vara)
            logger.info(
                "      %-10s ROC-AUC %.4f | log-loss %.4f  <- VARA ADVERSARIA",
                vara["modelo"], vara["roc_auc"], vara["log_loss"],
            )

    for nombre, fabrica in MODELOS.items():
        folds = evaluar_por_folds(matriz, fabrica, min_temporadas, cfg.target)
        resultados_por_modelo[nombre] = folds
        resumen.append({
            "modelo":      nombre,
            "folds":       len(folds),
            "pr_auc":      folds["pr_auc"].mean(),
            "lift_pr_auc": folds["lift_pr_auc"].mean(),
            "roc_auc":     folds["roc_auc"].mean(),
            "log_loss":    folds["log_loss"].mean(),
            "brier":       folds["brier"].mean(),
        })
        logger.info(
            "      %-10s ROC-AUC %.4f | PR-AUC %.4f | log-loss %.4f",
            nombre, resumen[-1]["roc_auc"], resumen[-1]["pr_auc"], resumen[-1]["log_loss"],
        )

    logger.info("[4/6] Entrenando el modelo de produccion ('%s')...", ML_MODELO_PRODUCCION)
    X, y = preparar_matriz(matriz, cfg.target)
    modelo = MODELOS[ML_MODELO_PRODUCCION]().fit(X, y)

    # Para medir importancia hace falta un modelo que NO haya visto el fold
    # donde se permuta: permutar sobre entrenamiento mide memorizacion.
    idx_train_ultimo, idx_test_ultimo = splits_origen_movil(matriz, min_temporadas)[-1]
    modelo_evaluacion = MODELOS[ML_MODELO_PRODUCCION]().fit(
        X.iloc[idx_train_ultimo], y.iloc[idx_train_ultimo]
    )

    folds_produccion = resultados_por_modelo[ML_MODELO_PRODUCCION]
    metadata = {
        "modelo":      ML_MODELO_PRODUCCION,
        "target":      cfg.target,
        "features":    features,
        "prior":       prior,
        "n_train":     len(X),
        "n_positivos": int(y.sum()),
        "tasa_base":   float(y.mean()),
        "min_temporadas_train": min_temporadas,
        "metricas": {
            k: float(folds_produccion[k].mean())
            for k in ("pr_auc", "lift_pr_auc", "roc_auc", "log_loss", "brier")
        },
    }
    version = guardar_modelo(modelo, cfg.nombre, metadata, DIR_MODELOS)

    logger.info("[5/6] Midiendo importancia por permutacion...")
    importancias = importancia_por_permutacion(
        modelo_evaluacion, X.iloc[idx_test_ultimo], y.iloc[idx_test_ultimo],
        n_repeticiones=20, semilla=0,
    )
    con_senal = importancias[importancias["importancia"] > 2 * importancias["desvio"]]
    logger.info(
        "      %d de %d features superan 2 desvios: %s",
        len(con_senal), len(importancias), list(con_senal["feature"]) or "ninguna",
    )
    registrar_importancias(importancias, DIR_IMPORTANCIAS_GOLD, version, cfg.nombre)

    logger.info("[6/6] Registrando experimentos en gold...")
    for nombre, folds in resultados_por_modelo.items():
        registrar_experimento(
            folds, {**metadata, "modelo": f"{cfg.nombre}:{nombre}"},
            DIR_EXPERIMENTOS_GOLD, version,
        )

    df_resumen = pd.DataFrame(resumen)
    logger.info("Modelo '%s' guardado | version %s", cfg.nombre, version)
    print("\n" + df_resumen.round(4).to_string(index=False) + "\n")
    return df_resumen


# --- Las dos configuraciones -----------------------------------------------

EXPULSIONES = ConfiguracionModelo(
    nombre=NOMBRE_MODELO_EXPULSIONES,
    target=TARGET,
    construir_features=construir_features,
    agregar_target=agregar_target,
    validar_contrato=validar_matriz_ml,
)

GOLES = ConfiguracionModelo(
    nombre=NOMBRE_MODELO_GOLES,
    target=TARGET_GOLES,
    # El mercado entra a la matriz para poder evaluarlo como baseline sobre los
    # MISMOS folds. `ml/dataset.METADATA` lo deja fuera de las features, asi
    # que el modelo no lo ve: se compara contra el, no lo copia.
    construir_features=lambda df, prior: construir_features_goles(
        df, prior=prior, incluir_mercado=True
    ),
    agregar_target=agregar_target_goles,
    validar_contrato=lambda m: validar_matriz_ml(m, target=TARGET_GOLES),
    baseline_extra=_baseline_mercado,
)


def ejecutar(min_temporadas_train: int | None = None) -> pd.DataFrame:
    """Modelo de expulsiones."""
    return ejecutar_modelo(EXPULSIONES, min_temporadas_train)


def ejecutar_goles(min_temporadas_train: int | None = None) -> pd.DataFrame:
    """Modelo de over 2.5 goles, contra el baseline del mercado."""
    return ejecutar_modelo(GOLES, min_temporadas_train)


def ejecutar_todos(min_temporadas_train: int | None = None) -> None:
    """Entrena y registra los dos modelos."""
    ejecutar(min_temporadas_train)
    ejecutar_goles(min_temporadas_train)


if __name__ == "__main__":
    ejecutar_todos()


# =============================================================================
# Tuning: OPT-IN explicito, no parte de la corrida normal.
#
# LA DECISION, con el numero que la sostiene
# ------------------------------------------
# Se midio con validacion anidada sobre los 7 folds: la ganancia fue +0.0017 de
# PR-AUC, dentro del ruido, a cambio de ~12x el tiempo de computo (12
# combinaciones x folds internos x folds externos). Correrlo en cada
# entrenamiento seria pagar mucho por nada.
#
# Pero tampoco se borra: cuando cambie materialmente el dataset —una liga
# nueva, cinco temporadas mas, otro target— la conclusion puede darse vuelta, y
# entonces hay que poder re-medirla sin reescribir nada.
#
# Ni codigo muerto ni default caro: una decision documentada con su medicion.
# =============================================================================

_GRILLA_HGB = {
    "max_depth":        [2, 3, 4],
    "learning_rate":    [0.03, 0.05],
    "min_samples_leaf": [50, 150],
}


def ejecutar_tuning(
    cfg: ConfiguracionModelo | None = None,
    min_temporadas_train: int | None = None,
) -> pd.DataFrame:
    """Busqueda anidada de hiperparametros. No corre en el pipeline normal.

    Uso:  python -c "from pipelines.ml_pipeline import ejecutar_tuning; ejecutar_tuning()"
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    from ml.tuning import evaluar_con_tuning_anidado

    cfg = cfg or EXPULSIONES
    min_temporadas = min_temporadas_train or ML_MIN_TEMPORADAS_TRAIN

    df = leer_tabla_delta(DIR_PARTIDOS_FOOTBALLDATA_BRONZE)
    prior = _prior_de_entrenamiento(df, cfg, min_temporadas)
    matriz = cfg.construir_features(df, prior=prior)

    def fabrica(**params):
        base = dict(
            categorical_features="from_dtype", early_stopping=True,
            validation_fraction=0.15, random_state=42, max_iter=300,
        )
        return HistGradientBoostingClassifier(**{**base, **params})

    logger.info("Tuning anidado de '%s' — %d combinaciones por fold externo",
                cfg.nombre, 3 * 2 * 2)
    resultados = evaluar_con_tuning_anidado(
        matriz, fabrica, _GRILLA_HGB,
        min_temporadas_train=min_temporadas, min_temporadas_interno=2,
    )
    columnas = ["temporada_test", "hiperparametros", "pr_auc", "roc_auc", "log_loss"]
    print()
    print(resultados[columnas].round(4).to_string(index=False))
    print()
    logger.info("PR-AUC promedio con tuning: %.4f", resultados["pr_auc"].mean())
    return resultados
