# =============================================================================
# bronze/schemas.py — Definición e inicialización de tablas Delta bronze
# =============================================================================
#
# Los constraints (id no nulo) se definen una sola vez al crear las tablas,
# no como parte del flujo de carga de datos.
# =============================================================================

import logging

import pyarrow as pa
from deltalake import DeltaTable

from utils.delta import asegurar_directorio, tabla_delta_existe

logger = logging.getLogger(__name__)

_SCHEMA_EQUIPOS = pa.schema([
    pa.field("id_equipo",            pa.large_string()),
    pa.field("nombre",               pa.large_string()),
    pa.field("nombre_alternativo",   pa.large_string()),
    pa.field("pais",                 pa.large_string()),
    pa.field("ciudad",               pa.large_string()),
    pa.field("liga",                 pa.large_string()),
    pa.field("id_liga",              pa.large_string()),
    pa.field("estadio",              pa.large_string()),
    pa.field("capacidad_estadio",    pa.int64()),
    pa.field("ubicacion_estadio",    pa.large_string()),
    pa.field("anio_fundacion",       pa.int64()),
    pa.field("descripcion_es",       pa.large_string()),
    pa.field("sitio_web",            pa.large_string()),
    pa.field("timestamp_extraccion", pa.large_string()),
])

_SCHEMA_PARTIDOS = pa.schema([
    pa.field("id_evento",            pa.large_string()),
    pa.field("nombre_evento",        pa.large_string()),
    pa.field("temporada",            pa.large_string()),
    pa.field("liga",                 pa.large_string()),
    pa.field("id_liga",              pa.large_string()),
    pa.field("equipo_local",         pa.large_string()),
    pa.field("id_equipo_local",      pa.large_string()),
    pa.field("goles_local",          pa.int64()),
    pa.field("equipo_visitante",     pa.large_string()),
    pa.field("id_equipo_visitante",  pa.large_string()),
    pa.field("goles_visitante",      pa.int64()),
    pa.field("fecha_partido",        pa.large_string()),
    pa.field("hora_partido",         pa.large_string()),
    pa.field("estadio",              pa.large_string()),
    pa.field("estado",               pa.large_string()),
    pa.field("timestamp_extraccion", pa.large_string()),
    pa.field("fecha_extraccion",     pa.large_string()),
])


def inicializar_tabla_equipos(ruta: str) -> None:
    """Crea la tabla Delta de equipos con schema y constraints si no existe."""
    if tabla_delta_existe(ruta):
        return
    asegurar_directorio(ruta)
    DeltaTable.create(
        ruta,
        schema=_SCHEMA_EQUIPOS,
        description="Equipos de la Argentinian Primera Division. Extraccion FULL.",
    )
    DeltaTable(ruta).alter.add_constraint({
        "ck_equipos_id_no_nulo":     "id_equipo IS NOT NULL",
        "ck_equipos_nombre_no_nulo": "nombre IS NOT NULL",
    })
    logger.info("Tabla 'equipos' inicializada en '%s'.", ruta)


def inicializar_tabla_partidos(ruta: str) -> None:
    """Crea la tabla Delta de partidos con schema, partición y constraints si no existe."""
    if tabla_delta_existe(ruta):
        return
    asegurar_directorio(ruta)
    DeltaTable.create(
        ruta,
        schema=_SCHEMA_PARTIDOS,
        partition_by=["fecha_extraccion"],
        description="Partidos de la Argentinian Primera Division. Extraccion INCREMENTAL diaria.",
    )
    DeltaTable(ruta).alter.add_constraint({"ck_partidos_id_no_nulo": "id_evento IS NOT NULL"})
    logger.info("Tabla 'partidos' inicializada en '%s'.", ruta)
