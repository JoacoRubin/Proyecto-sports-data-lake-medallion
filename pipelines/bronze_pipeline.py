# =============================================================================
# pipelines/bronze_pipeline.py — Pipeline TP1: extracción y almacenamiento
# =============================================================================

import logging

import pandas as pd

from config import DIR_EQUIPOS_BRONZE, DIR_PARTIDOS_BRONZE, LIGAS
from bronze.extractors import extraer_equipos, extraer_partidos
from bronze.loaders import guardar_equipos_bronze, guardar_partidos_bronze
from bronze.schemas import inicializar_tabla_equipos, inicializar_tabla_partidos
from utils.delta import mostrar_resumen_tabla

logger = logging.getLogger(__name__)


def _extraer_todas_las_ligas() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extrae equipos y partidos de todas las ligas configuradas y los concatena.

    Una liga que falle (sin datos, error de red) se registra y se saltea, sin
    abortar el resto de la ingesta.
    """
    dfs_equipos: list[pd.DataFrame] = []
    dfs_partidos: list[pd.DataFrame] = []

    for liga in LIGAS:
        try:
            dfs_equipos.append(extraer_equipos(liga["nombre"]))
            dfs_partidos.append(extraer_partidos(liga["id"], liga["temporada"]))
        except Exception as e:  # noqa: BLE001 — una liga caída no debe frenar el resto
            logger.warning("Liga '%s' omitida: %s", liga["nombre"], e)

    if not dfs_partidos:
        raise RuntimeError("Ninguna liga devolvió datos. Abortando ingesta bronze.")

    equipos = pd.concat(dfs_equipos, ignore_index=True).drop_duplicates(
        subset=["id_equipo"], keep="first"
    )
    partidos = pd.concat(dfs_partidos, ignore_index=True).drop_duplicates(
        subset=["id_evento", "fecha_extraccion"], keep="first"
    )
    return equipos, partidos


def ejecutar() -> None:
    """TP1 — Extrae datos de TheSportsDB (multi-liga) y los guarda en bronze."""
    logger.info("=" * 62)
    logger.info("  TP1 - Extraccion y Almacenamiento (bronze)")
    logger.info("  Ligas configuradas: %d", len(LIGAS))
    logger.info("=" * 62)

    logger.info("[0/4] Inicializando estructura del data lake...")
    inicializar_tabla_equipos(DIR_EQUIPOS_BRONZE)
    inicializar_tabla_partidos(DIR_PARTIDOS_BRONZE)

    equipos, partidos = _extraer_todas_las_ligas()

    guardar_equipos_bronze(equipos,  DIR_EQUIPOS_BRONZE)
    guardar_partidos_bronze(partidos, DIR_PARTIDOS_BRONZE)

    logger.info("--- Verificacion bronze ---")
    mostrar_resumen_tabla(DIR_EQUIPOS_BRONZE,  "bronze/equipos  (FULL)")
    mostrar_resumen_tabla(DIR_PARTIDOS_BRONZE, "bronze/partidos (INCREMENTAL)")
