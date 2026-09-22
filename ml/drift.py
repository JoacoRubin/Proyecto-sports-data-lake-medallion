# =============================================================================
# ml/drift.py — ¿el modelo YA en producción sigue sirviendo?
#
# EN QUÉ SE DIFERENCIA DE ml/promotion.py
# ----------------------------------------
# `promotion.py` responde "¿el candidato reemplaza al modelo en producción?",
# comparando dos modelos entre sí sobre los MISMOS folds históricos. Este
# módulo responde una pregunta distinta: "¿el modelo que ya está sirviendo
# predicciones sigue funcionando igual de bien contra partidos que jugó
# DESPUÉS de promoverse?" — la porción de realidad que ni el entrenamiento
# ni el quality gate llegaron a ver.
#
# NO HAY UN CAMINO DE FEATURES APARTE
# ------------------------------------
# Mismo principio que ml/inference.py: se corre el mismo `construir_features`
# de siempre y se reindexa a `metadata["features"]`, el conjunto y el orden
# exactos con los que se entrenó ese modelo — no los que salgan de correr el
# feature engineering hoy.
#
# POR QUÉ HAY UN MÍNIMO DE PARTIDOS
# -----------------------------------
# Con ~10-15 partidos frescos, una métrica puntual es en su mayoría ruido:
# no hay forma de distinguir "el modelo empeoró" de "esta semana tocaron
# partidos raros". Sin muestra suficiente, el veredicto es "no evaluado", no
# un falso "sin drift" que esconde el problema.
# =============================================================================

from dataclasses import dataclass

import pandas as pd

from ml.evaluation import metricas


@dataclass(frozen=True)
class VeredictoDrift:
    """El resultado de auditar al modelo en producción, con los números que
    lo sostienen — mismo espíritu que `DecisionPromocion` en promotion.py."""
    evaluado: bool
    n_partidos_frescos: int
    razon: str
    drift_detectado: bool = False
    metrica_principal: str | None = None
    valor_reciente: float | None = None
    valor_original: float | None = None
    metrica_calibracion: str | None = None
    valor_calibracion_reciente: float | None = None
    valor_calibracion_original: float | None = None


def evaluar_drift_metricas(
    metricas_recientes: dict,
    metricas_originales: dict,
    metrica_principal: str,
    n_partidos_frescos: int,
    metrica_calibracion: str = "brier",
    tolerancia_ranking: float = 0.05,
    tolerancia_calibracion: float = 0.01,
) -> VeredictoDrift:
    """La regla pura: compara las métricas de HOY contra las que el modelo
    tenía registradas cuando se promovió.

    Marca drift si la métrica principal cae más de `tolerancia_ranking` O
    si la calibración (Brier) empeora más de `tolerancia_calibracion` —
    cualquiera de las dos es una señal real, a diferencia del quality gate
    de promoción, que exige AMBAS condiciones para rechazar un candidato.
    Acá no se está comparando contra una alternativa: cualquier degradación
    real importa.
    """
    valor_reciente = metricas_recientes[metrica_principal]
    valor_original = metricas_originales[metrica_principal]
    calibracion_reciente = metricas_recientes[metrica_calibracion]
    calibracion_original = metricas_originales[metrica_calibracion]

    empeoro_ranking = valor_reciente < valor_original - tolerancia_ranking
    empeoro_calibracion = calibracion_reciente > calibracion_original + tolerancia_calibracion
    drift = empeoro_ranking or empeoro_calibracion

    if drift:
        partes = []
        if empeoro_ranking:
            partes.append(
                f"{metrica_principal} cayo de {valor_original:.4f} a {valor_reciente:.4f} "
                f"(tolerancia {tolerancia_ranking})"
            )
        if empeoro_calibracion:
            partes.append(
                f"{metrica_calibracion} empeoro de {calibracion_original:.4f} a "
                f"{calibracion_reciente:.4f} (tolerancia {tolerancia_calibracion})"
            )
        razon = (
            "DRIFT detectado: " + "; ".join(partes) +
            f" sobre los {n_partidos_frescos} partidos jugados desde la promocion."
        )
    else:
        razon = (
            f"Sin drift: {metrica_principal} {valor_reciente:.4f} "
            f"(original {valor_original:.4f}), {metrica_calibracion} "
            f"{calibracion_reciente:.4f} (original {calibracion_original:.4f}) "
            f"sobre {n_partidos_frescos} partidos frescos."
        )

    return VeredictoDrift(
        evaluado=True,
        n_partidos_frescos=n_partidos_frescos,
        razon=razon,
        drift_detectado=drift,
        metrica_principal=metrica_principal,
        valor_reciente=valor_reciente,
        valor_original=valor_original,
        metrica_calibracion=metrica_calibracion,
        valor_calibracion_reciente=calibracion_reciente,
        valor_calibracion_original=calibracion_original,
    )


def medir_drift(
    modelo,
    metadata_produccion: dict,
    matriz: pd.DataFrame,
    metrica_principal: str,
    metrica_calibracion: str = "brier",
    tolerancia_ranking: float = 0.05,
    tolerancia_calibracion: float = 0.01,
    min_partidos: int = 20,
) -> VeredictoDrift:
    """Reevalúa el modelo YA en producción (sin reentrenar) contra los
    partidos con fecha posterior a su propia promoción.

    Args:
        modelo: el modelo YA entrenado y en producción (no un candidato).
        metadata_produccion: su ficha técnica (`guardado_en`, `features`,
            `target`, `metricas` con las que se promovió).
        matriz: la matriz de features completa, construida con
            `cfg.construir_features(df, prior=metadata_produccion["prior"])`
            -- el prior CON el que este modelo se entrenó, no uno recalculado
            hoy (ver ml/inference.py).
    """
    # `guardado_en` es un timestamp UTC con tz (ver ml/registry.guardar_modelo);
    # `fecha_partido` es solo una fecha, sin hora ni tz. Comparar tz-aware
    # contra tz-naive tira TypeError -- se le saca el tz a `guardado_en` para
    # comparar a nivel de dia, que es toda la resolucion que fecha_partido tiene.
    guardado_en = pd.to_datetime(metadata_produccion["guardado_en"]).tz_localize(None)
    fechas = pd.to_datetime(matriz["fecha_partido"], errors="coerce")
    recientes = matriz[fechas > guardado_en]

    if len(recientes) < min_partidos:
        return VeredictoDrift(
            evaluado=False,
            n_partidos_frescos=len(recientes),
            razon=(
                f"Solo {len(recientes)} partidos jugados desde la promocion "
                f"({min_partidos} minimos) -- muestra insuficiente para medir drift."
            ),
        )

    X = recientes.reindex(columns=metadata_produccion["features"])
    for col in X.columns:
        if recientes[col].dtype == object or str(recientes[col].dtype) == "string":
            X[col] = X[col].astype("category")
    y = recientes[metadata_produccion["target"]].astype(int)

    probabilidades = modelo.predict_proba(X)[:, 1]
    metricas_recientes = metricas(y.to_numpy(), probabilidades)

    return evaluar_drift_metricas(
        metricas_recientes,
        metadata_produccion["metricas"],
        metrica_principal=metrica_principal,
        n_partidos_frescos=len(recientes),
        metrica_calibracion=metrica_calibracion,
        tolerancia_ranking=tolerancia_ranking,
        tolerancia_calibracion=tolerancia_calibracion,
    )
