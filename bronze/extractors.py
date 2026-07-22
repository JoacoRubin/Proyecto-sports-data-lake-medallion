# =============================================================================
# bronze/extractors.py — Extracción de datos desde TheSportsDB
# =============================================================================

import logging
from datetime import datetime, timezone

import pandas as pd

from config import URL_BASE
from bronze.mappers import (
    ESTADOS_FINALIZADOS, mapear_equipo, mapear_partido, tipar_equipos, tipar_partidos,
)
from utils.api import obtener_datos_api

logger = logging.getLogger(__name__)


def _fecha_extraccion_actual() -> tuple[str, str]:
    """Retorna (fecha_YYYY-MM-DD, timestamp_ISO) en UTC."""
    ahora = datetime.now(timezone.utc)
    return ahora.strftime("%Y-%m-%d"), ahora.isoformat()


def extraer_equipos(liga_nombre: str) -> pd.DataFrame:
    """Extrae metadatos de equipos — estrategia FULL.

    Datos estáticos: nombre, estadio, ciudad, año de fundación.
    Se sobreescribe la tabla completa en cada ejecución.
    """
    logger.info("[1/4] Extrayendo equipos — Liga: %s", liga_nombre)

    equipos = obtener_datos_api(URL_BASE, "/search_all_teams.php", {"l": liga_nombre}, "teams")
    if not equipos:
        raise ValueError(f"No se encontraron equipos para '{liga_nombre}'.")

    _, timestamp = _fecha_extraccion_actual()

    registros = [mapear_equipo(equipo, timestamp) for equipo in equipos]

    df = pd.DataFrame(registros).drop_duplicates(subset=["id_equipo"], keep="first")
    df = tipar_equipos(df)

    logger.info("  > %d equipos extraídos.", len(df))
    return df


def extraer_partidos(liga_id: str, temporada: str) -> pd.DataFrame:
    """Extrae partidos de una temporada — estrategia INCREMENTAL.

    Incluye fecha de extracción para INSERT-OVERWRITE incremental por día.
    """
    logger.info("[2/4] Extrayendo partidos — Liga ID: %s | Temporada: %s", liga_id, temporada)

    eventos = obtener_datos_api(
        URL_BASE, "/eventsseason.php", {"id": liga_id, "s": temporada}, "events"
    )
    if not eventos:
        raise ValueError(f"No se encontraron eventos para liga '{liga_id}', temporada '{temporada}'.")

    fecha_extraccion, timestamp = _fecha_extraccion_actual()

    registros = [mapear_partido(evento, timestamp, fecha_extraccion) for evento in eventos]

    df = pd.DataFrame(registros).drop_duplicates(subset=["id_evento", "fecha_extraccion"], keep="first")
    df = tipar_partidos(df)

    jugados    = df["estado"].str.strip().str.upper().isin(ESTADOS_FINALIZADOS).sum()
    pendientes = len(df) - jugados
    logger.info(
        "  > %d partidos extraídos — jugados: %d | pendientes: %d",
        len(df), jugados, pendientes,
    )
    return df
