# =============================================================================
# pipelines/bronze_pipeline.py — Pipeline TP1: extracción y almacenamiento
# =============================================================================

import logging

from config import (
    DIR_EQUIPOS_BRONZE, DIR_PARTIDOS_BRONZE,
    LIGA_NOMBRE, LIGA_ID, TEMPORADA,
)
from bronze.extractors import extraer_equipos, extraer_partidos
from bronze.loaders import guardar_equipos_bronze, guardar_partidos_bronze
from bronze.schemas import inicializar_tabla_equipos, inicializar_tabla_partidos
from utils.delta import mostrar_resumen_tabla

logger = logging.getLogger(__name__)


def ejecutar() -> None:
    """TP1 — Extrae datos de TheSportsDB y los guarda en la capa bronze."""
    logger.info("=" * 62)
    logger.info("  TP1 - Extraccion y Almacenamiento (bronze)")
    logger.info("  Liga: %s | Temporada: %s", LIGA_NOMBRE, TEMPORADA)
    logger.info("=" * 62)

    logger.info("[0/4] Inicializando estructura del data lake...")
    inicializar_tabla_equipos(DIR_EQUIPOS_BRONZE)
    inicializar_tabla_partidos(DIR_PARTIDOS_BRONZE)

    equipos  = extraer_equipos(LIGA_NOMBRE)
    partidos = extraer_partidos(LIGA_ID, TEMPORADA)

    guardar_equipos_bronze(equipos,  DIR_EQUIPOS_BRONZE)
    guardar_partidos_bronze(partidos, DIR_PARTIDOS_BRONZE)

    logger.info("--- Verificacion bronze ---")
    mostrar_resumen_tabla(DIR_EQUIPOS_BRONZE,  "bronze/equipos  (FULL)")
    mostrar_resumen_tabla(DIR_PARTIDOS_BRONZE, "bronze/partidos (INCREMENTAL)")
