# =============================================================================
# bronze/mappers.py — Mapeo y tipado canónico de la capa bronze.
#
# Fuente ÚNICA de verdad del esquema bronze. Lo usan por igual:
#   - bronze/extractors.py   (ingesta batch desde la API)
#   - producer_thesportsdb.py (ingesta streaming vía Kafka)
#
# Así el stream y el batch producen exactamente el mismo esquema y tipos,
# y pueden escribir a la MISMA tabla Delta sin divergencias.
# =============================================================================

import pandas as pd

# Códigos que TheSportsDB usa para un partido ya jugado. Según el endpoint y la
# versión de la API puede ser 'FT' (Full Time), 'AET' (tras alargue), 'PEN'
# (penales) o el texto 'Match Finished'. Se compara en mayúsculas.
ESTADOS_FINALIZADOS = {"FT", "AET", "PEN", "MATCH FINISHED", "FINISHED"}


def mapear_equipo(raw: dict, timestamp: str) -> dict:
    """Convierte un equipo crudo de la API al esquema bronze completo."""
    return {
        "id_equipo":            raw.get("idTeam"),
        "nombre":               raw.get("strTeam"),
        "nombre_alternativo":   raw.get("strAlternate"),
        "pais":                 raw.get("strCountry"),
        "ciudad":               raw.get("strCity") or raw.get("strStadiumLocation"),
        "liga":                 raw.get("strLeague"),
        "id_liga":              raw.get("idLeague"),
        "estadio":              raw.get("strStadium"),
        "capacidad_estadio":    raw.get("intStadiumCapacity"),
        "ubicacion_estadio":    raw.get("strStadiumLocation"),
        "anio_fundacion":       raw.get("intFormedYear"),
        "descripcion_es":       raw.get("strDescriptionES"),
        "sitio_web":            raw.get("strWebsite"),
        "timestamp_extraccion": timestamp,
    }


def mapear_partido(raw: dict, timestamp: str, fecha_extraccion: str) -> dict:
    """Convierte un partido crudo de la API al esquema bronze completo."""
    return {
        "id_evento":            raw.get("idEvent"),
        "nombre_evento":        raw.get("strEvent"),
        "temporada":            raw.get("strSeason"),
        "liga":                 raw.get("strLeague"),
        "id_liga":              raw.get("idLeague"),
        "equipo_local":         raw.get("strHomeTeam"),
        "id_equipo_local":      raw.get("idHomeTeam"),
        "goles_local":          raw.get("intHomeScore"),
        "equipo_visitante":     raw.get("strAwayTeam"),
        "id_equipo_visitante":  raw.get("idAwayTeam"),
        "goles_visitante":      raw.get("intAwayScore"),
        "fecha_partido":        raw.get("dateEvent"),
        "hora_partido":         raw.get("strTime"),
        "estadio":              raw.get("strVenue"),
        "estado":               raw.get("strStatus"),
        "timestamp_extraccion": timestamp,
        "fecha_extraccion":     fecha_extraccion,
    }


def tipar_equipos(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica los tipos canónicos de la tabla bronze de equipos."""
    df = df.copy()
    df["capacidad_estadio"] = pd.to_numeric(df["capacidad_estadio"], errors="coerce")
    df["anio_fundacion"]    = pd.to_numeric(df["anio_fundacion"],    errors="coerce")
    return df.astype({
        "id_equipo": "string", "nombre": "string", "nombre_alternativo": "string",
        "pais": "string", "ciudad": "string", "liga": "string", "id_liga": "string",
        "estadio": "string", "capacidad_estadio": "Int64", "ubicacion_estadio": "string",
        "anio_fundacion": "Int64", "descripcion_es": "string",
        "sitio_web": "string", "timestamp_extraccion": "string",
    })


def tipar_partidos(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica los tipos canónicos de la tabla bronze de partidos."""
    df = df.copy()
    df["goles_local"]     = pd.to_numeric(df["goles_local"],     errors="coerce")
    df["goles_visitante"] = pd.to_numeric(df["goles_visitante"], errors="coerce")
    return df.astype({
        "id_evento": "string", "nombre_evento": "string", "temporada": "string",
        "liga": "string", "id_liga": "string", "equipo_local": "string",
        "id_equipo_local": "string", "goles_local": "Int64",
        "equipo_visitante": "string", "id_equipo_visitante": "string",
        "goles_visitante": "Int64", "fecha_partido": "string",
        "hora_partido": "string", "estadio": "string", "estado": "string",
        "timestamp_extraccion": "string", "fecha_extraccion": "string",
    })
