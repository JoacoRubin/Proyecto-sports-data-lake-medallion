# =============================================================================
# ml/inference.py — Predecir partidos que todavia no se jugaron.
#
# EL PROBLEMA QUE ESTE MODULO EVITA: TRAIN/SERVE SKEW
# ---------------------------------------------------
# La tentacion es escribir un camino aparte que arme las features del partido
# futuro "a mano". Es el error mas caro del ML en produccion: dos
# implementaciones de la misma logica que divergen de a poco, y el modelo
# termina recibiendo features distintas de las que vio al entrenar. Nadie lo
# nota, porque no hay ningun error — solo predicciones cada vez peores.
#
# LA SOLUCION: NO HAY CAMINO APARTE
# ---------------------------------
# El partido futuro se agrega como UNA FILA MAS al historico, con las
# estadisticas en nulo (todavia no existen), y se corre EXACTAMENTE el mismo
# `construir_features` del entrenamiento. Las medias moviles y las tasas
# historicas se calculan solas, con la misma logica de ventana expansiva.
#
# Imposible que diverjan, porque es el mismo codigo.
#
# Ademas se usan `metadata["features"]` y `metadata["prior"]` guardados con el
# modelo: el orden y el conjunto de columnas son los del entrenamiento, no los
# que resulten de correr el feature engineering hoy. Si mañana se agrega una
# feature nueva, este modulo sigue alimentando al modelo viejo como
# corresponde, hasta que se lo reentrene.
# =============================================================================

import logging
from dataclasses import asdict, dataclass, field

import pandas as pd

from ml.features import construir_features
from ml.features_goles import construir_features_goles
from ml.registry import cargar_modelo

logger = logging.getLogger(__name__)

# Que constructor de features corresponde a cada target.
_CONSTRUCTORES = {
    "hubo_expulsion": construir_features,
    "over_2_5": lambda df, prior: construir_features_goles(
        df, prior=prior, incluir_mercado=True
    ),
}


@dataclass
class PartidoAPredecir:
    """Un partido futuro. Solo lo que se sabe ANTES de que se juegue.

    Las cuotas son opcionales: si el partido todavia no tiene mercado, las
    features derivadas quedan nulas y el modelo predice igual — es el mismo
    caso que un equipo sin historia suficiente.
    """
    liga: str
    temporada: str
    equipo_local: str
    equipo_visitante: str
    fecha_partido: str
    hora_partido: str | None = None
    cuota_local: float | None = None
    cuota_empate: float | None = None
    cuota_visitante: float | None = None
    cuota_over25: float | None = None
    cuota_under25: float | None = None
    cuota_local_mercado: float | None = None
    cuota_empate_mercado: float | None = None
    cuota_visitante_mercado: float | None = None


def _fila_futura(partido: PartidoAPredecir, indice: int, columnas: list[str]) -> dict:
    """Convierte el partido en una fila con el esquema del historico.

    Todo lo que describe el partido jugado queda en None: goles, tarjetas,
    tiros. Es literalmente cierto —no se jugo— y es lo que hace que las
    features se calculen solo con el pasado.
    """
    fila = {c: None for c in columnas}
    fila.update({k: v for k, v in asdict(partido).items() if v is not None})
    fila["id_evento"] = f"FUTURO-{indice:04d}"
    fila["nombre_evento"] = f"{partido.equipo_local} vs {partido.equipo_visitante}"
    fila["estado"] = "PENDIENTE"
    return fila


def predecir(
    partidos: list[PartidoAPredecir],
    nombre_modelo: str,
    dir_modelos: str,
    historico: pd.DataFrame,
    version: str | None = None,
    devolver_features: bool = False,
) -> pd.DataFrame:
    """Predice la probabilidad del evento para partidos que no se jugaron.

    Args:
        partidos: los partidos a predecir.
        nombre_modelo: 'expulsiones' o 'goles'.
        dir_modelos: directorio del registro de modelos.
        historico: partidos ya jugados (bronze). De aca sale la historia de
            cada equipo; sin esto las features quedarian todas nulas.
        version: version del modelo. Por defecto, la mas reciente.
        devolver_features: agrega las features usadas con prefijo `feat__`,
            para poder auditar que recibio el modelo.

    Returns:
        Una fila por partido con la probabilidad y la trazabilidad.
    """
    modelo, metadata = cargar_modelo(nombre_modelo, dir_modelos, version)

    if not partidos:
        return pd.DataFrame()

    target = metadata["target"]
    constructor = _CONSTRUCTORES[target]

    # El historico no se toca: se trabaja sobre una copia ampliada.
    #
    # Las filas futuras se tipan con los dtypes del historico ANTES de
    # concatenar. Si no, pandas ve columnas enteras en NA y deduce dtypes
    # distintos para cada mitad — hoy avisa con un FutureWarning y en una
    # version proxima cambia el resultado.
    futuras = pd.DataFrame([
        _fila_futura(p, i, list(historico.columns)) for i, p in enumerate(partidos)
    ]).reindex(columns=historico.columns)
    for col, dtype in historico.dtypes.items():
        try:
            futuras[col] = futuras[col].astype(dtype)
        except (TypeError, ValueError):
            pass          # un dtype que no admite nulos: se deja como viene
    ampliado = pd.concat([historico, futuras], ignore_index=True)

    # EL MISMO feature engineering del entrenamiento, con EL MISMO prior.
    matriz = constructor(ampliado, prior=metadata["prior"])

    es_futuro = matriz["id_evento"].astype(str).str.startswith("FUTURO-")
    matriz_futura = matriz[es_futuro].copy()

    # Las columnas del ENTRENAMIENTO, en su orden. No las que salgan hoy.
    X = matriz_futura.reindex(columns=metadata["features"])
    for col in X.columns:
        if matriz[col].dtype == object or str(matriz[col].dtype) == "string":
            X[col] = X[col].astype("category")

    probabilidades = modelo.predict_proba(X)[:, 1]

    salida = pd.DataFrame({
        "liga":            matriz_futura["liga"].values,
        "fecha_partido":   matriz_futura["fecha_partido"].values,
        "equipo_local":    matriz_futura["equipo_local"].values,
        "equipo_visitante": matriz_futura["equipo_visitante"].values,
        "probabilidad":    probabilidades,
        "target":          target,
        "modelo":          nombre_modelo,
        "version_modelo":  metadata["version"],
    })

    if devolver_features:
        for col in metadata["features"]:
            salida[f"feat__{col}"] = X[col].values

    logger.info(
        "Predichos %d partidos con '%s' v%s | probabilidad media %.4f",
        len(salida), nombre_modelo, metadata["version"], probabilidades.mean(),
    )
    return salida
