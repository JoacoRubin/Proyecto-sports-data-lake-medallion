# =============================================================================
# ml/registry.py — Registro de modelos y tracking de experimentos.
#
# POR QUÉ NO MLflow (NI LANGSMITH)
# --------------------------------
# LangSmith es observabilidad para aplicaciones con LLM: prompts, cadenas,
# agentes. Acá no hay ningún LLM — es pandas -> Delta -> scikit-learn. No hay
# nada que trazar.
#
# MLflow sí resuelve el problema correcto, pero trae un servidor y un artifact
# store para versionar un modelo. Para este proyecto es infraestructura de más.
#
# La alternativa que elegimos usa lo que el proyecto YA tiene: los experimentos
# van a una tabla Delta en gold, como cualquier otro dato del lake. Se
# consultan con el mismo `leer_tabla_delta`, se muestran en el mismo Streamlit,
# y Delta aporta versionado y time travel de regalo. Coherencia arquitectónica
# en vez de una herramienta más.
#
# El modelo en sí va a joblib, versionado por timestamp y SIEMPRE acompañado
# de su metadata: con qué features se entrenó, cuándo, y qué métricas dio. Un
# .joblib suelto es un binario que nadie puede auditar.
# =============================================================================

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

import joblib
import pandas as pd

from utils.delta import asegurar_directorio, guardar_en_delta, leer_tabla_delta, tabla_delta_existe

logger = logging.getLogger(__name__)


def _dir_modelo(nombre: str, dir_modelos: str) -> str:
    return os.path.join(dir_modelos, nombre)


def nueva_version() -> str:
    """Versión basada en timestamp UTC: ordenable alfabéticamente."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")


def guardar_modelo(
    modelo: Any, nombre: str, metadata: dict, dir_modelos: str
) -> str:
    """Persiste un modelo entrenado junto con su ficha técnica.

    Nunca sobreescribe: cada guardado es una versión nueva, así siempre se
    puede volver a la anterior si la nueva resulta peor en producción.

    Returns:
        La versión creada.
    """
    version = nueva_version()
    destino = _dir_modelo(nombre, dir_modelos)
    asegurar_directorio(destino)

    joblib.dump(modelo, os.path.join(destino, f"{version}.joblib"))

    ficha = {**metadata, "version": version, "guardado_en": datetime.now(timezone.utc).isoformat()}
    with open(os.path.join(destino, f"{version}.json"), "w", encoding="utf-8") as f:
        json.dump(ficha, f, ensure_ascii=False, indent=2)

    logger.info("Modelo '%s' guardado | version %s", nombre, version)
    return version


def listar_versiones(nombre: str, dir_modelos: str) -> list[str]:
    """Versiones disponibles de un modelo, de la más vieja a la más nueva."""
    destino = _dir_modelo(nombre, dir_modelos)
    if not os.path.isdir(destino):
        return []
    return sorted(
        f.removesuffix(".joblib") for f in os.listdir(destino) if f.endswith(".joblib")
    )


def cargar_modelo(
    nombre: str, dir_modelos: str, version: str | None = None
) -> tuple[Any, dict]:
    """Carga un modelo y su metadata. Sin `version`, la más reciente.

    Raises:
        FileNotFoundError: si no hay ninguna versión de ese modelo.
    """
    versiones = listar_versiones(nombre, dir_modelos)
    if not versiones:
        raise FileNotFoundError(
            f"No hay ninguna version del modelo '{nombre}' en '{dir_modelos}'."
        )
    version = version or versiones[-1]
    destino = _dir_modelo(nombre, dir_modelos)

    modelo = joblib.load(os.path.join(destino, f"{version}.joblib"))
    with open(os.path.join(destino, f"{version}.json"), encoding="utf-8") as f:
        metadata = json.load(f)
    return modelo, metadata


def registrar_experimento(
    resultados_folds: pd.DataFrame,
    metadata: dict,
    ruta_tabla: str,
    version: str,
) -> None:
    """Agrega la corrida a la tabla Delta de experimentos, una fila por fold.

    Es el historial: sin él, "el modelo da 0.21" es un número sin contexto que
    nadie puede reproducir ni comparar contra la corrida anterior.
    """
    filas = resultados_folds.copy()
    filas["version"]      = version
    filas["modelo"]       = metadata.get("modelo", "desconocido")
    filas["features"]     = ", ".join(metadata.get("features", []))
    filas["n_features"]   = len(metadata.get("features", []))
    filas["prior"]        = metadata.get("prior")
    filas["ejecutado_en"] = datetime.now(timezone.utc).isoformat()

    guardar_en_delta(filas, ruta_tabla, modo="append", modo_esquema="merge")
    logger.info(
        "Experimento registrado | version %s | modelo %s | %d folds",
        version, filas["modelo"].iloc[0], len(filas),
    )


def leer_experimentos(ruta_tabla: str) -> pd.DataFrame:
    """Historial completo de experimentos. Vacío si todavía no se entrenó nada."""
    if not tabla_delta_existe(ruta_tabla):
        return pd.DataFrame()
    return leer_tabla_delta(ruta_tabla)


def registrar_importancias(
    importancias: pd.DataFrame, ruta_tabla: str, version: str, modelo: str
) -> None:
    """Guarda la importancia por permutacion, una fila por feature.

    Queda junto a las metricas y no dentro de un log: es la unica forma de
    comparar entre corridas si una feature dejo de aportar. `senal` marca las
    que superan dos desvios — el resto puede ser ruido de la permutacion.
    """
    filas = importancias.copy()
    filas["version"]      = version
    filas["modelo"]       = modelo
    filas["senal"]        = filas["importancia"] > 2 * filas["desvio"]
    filas["metrica"]      = importancias.attrs.get("metrica", "average_precision")
    filas["ejecutado_en"] = datetime.now(timezone.utc).isoformat()

    guardar_en_delta(filas, ruta_tabla, modo="append", modo_esquema="merge")
    logger.info(
        "Importancias registradas | version %s | %d features | %d con senal",
        version, len(filas), int(filas["senal"].sum()),
    )


def leer_importancias(ruta_tabla: str) -> pd.DataFrame:
    """Importancias historicas. Vacio si todavia no se entreno nada."""
    if not tabla_delta_existe(ruta_tabla):
        return pd.DataFrame()
    return leer_tabla_delta(ruta_tabla)
