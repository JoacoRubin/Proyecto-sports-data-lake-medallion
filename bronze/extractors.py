# =============================================================================
# bronze/extractors.py — Extracción de datos desde TheSportsDB
# =============================================================================

import logging
from datetime import datetime, timezone

import pandas as pd

from config import URL_BASE
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

    registros = [
        {
            "id_equipo":            equipo.get("idTeam"),
            "nombre":               equipo.get("strTeam"),
            "nombre_alternativo":   equipo.get("strAlternate"),
            "pais":                 equipo.get("strCountry"),
            "ciudad":               equipo.get("strCity") or equipo.get("strStadiumLocation"),
            "liga":                 equipo.get("strLeague"),
            "id_liga":              equipo.get("idLeague"),
            "estadio":              equipo.get("strStadium"),
            "capacidad_estadio":    equipo.get("intStadiumCapacity"),
            "ubicacion_estadio":    equipo.get("strStadiumLocation"),
            "anio_fundacion":       equipo.get("intFormedYear"),
            "descripcion_es":       equipo.get("strDescriptionES"),
            "sitio_web":            equipo.get("strWebsite"),
            "timestamp_extraccion": timestamp,
        }
        for equipo in equipos
    ]

    df = pd.DataFrame(registros).drop_duplicates(subset=["id_equipo"], keep="first")
    df["capacidad_estadio"] = pd.to_numeric(df["capacidad_estadio"], errors="coerce")
    df["anio_fundacion"]    = pd.to_numeric(df["anio_fundacion"],    errors="coerce")
    df = df.astype({
        "id_equipo": "string", "nombre": "string", "nombre_alternativo": "string",
        "pais": "string", "ciudad": "string", "liga": "string", "id_liga": "string",
        "estadio": "string", "capacidad_estadio": "Int64", "ubicacion_estadio": "string",
        "anio_fundacion": "Int64", "descripcion_es": "string",
        "sitio_web": "string", "timestamp_extraccion": "string",
    })

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

    registros = [
        {
            "id_evento":            evento.get("idEvent"),
            "nombre_evento":        evento.get("strEvent"),
            "temporada":            evento.get("strSeason"),
            "liga":                 evento.get("strLeague"),
            "id_liga":              evento.get("idLeague"),
            "equipo_local":         evento.get("strHomeTeam"),
            "id_equipo_local":      evento.get("idHomeTeam"),
            "goles_local":          evento.get("intHomeScore"),
            "equipo_visitante":     evento.get("strAwayTeam"),
            "id_equipo_visitante":  evento.get("idAwayTeam"),
            "goles_visitante":      evento.get("intAwayScore"),
            "fecha_partido":        evento.get("dateEvent"),
            "hora_partido":         evento.get("strTime"),
            "estadio":              evento.get("strVenue"),
            "estado":               evento.get("strStatus"),
            "timestamp_extraccion": timestamp,
            "fecha_extraccion":     fecha_extraccion,
        }
        for evento in eventos
    ]

    df = pd.DataFrame(registros).drop_duplicates(subset=["id_evento", "fecha_extraccion"], keep="first")
    df["goles_local"]     = pd.to_numeric(df["goles_local"],     errors="coerce")
    df["goles_visitante"] = pd.to_numeric(df["goles_visitante"], errors="coerce")
    df = df.astype({
        "id_evento": "string", "nombre_evento": "string", "temporada": "string",
        "liga": "string", "id_liga": "string", "equipo_local": "string",
        "id_equipo_local": "string", "goles_local": "Int64",
        "equipo_visitante": "string", "id_equipo_visitante": "string",
        "goles_visitante": "Int64", "fecha_partido": "string",
        "hora_partido": "string", "estadio": "string", "estado": "string",
        "timestamp_extraccion": "string", "fecha_extraccion": "string",
    })

    jugados    = df["estado"].str.contains("Finished", case=False, na=False).sum()
    pendientes = len(df) - jugados
    logger.info(
        "  > %d partidos extraídos — jugados: %d | pendientes: %d",
        len(df), jugados, pendientes,
    )
    return df
