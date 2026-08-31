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


# =============================================================================
# Tabla bronze de la fuente football-data.co.uk.
#
# Tabla SEPARADA de la de TheSportsDB a propósito: bronze preserva cada fuente
# con su fidelidad original. La unificación es trabajo de silver, que las lee
# a las dos y las conforma a un único esquema.
#
# Comparte las 17 columnas canónicas de _SCHEMA_PARTIDOS y agrega las
# estadísticas de partido y las cuotas, que TheSportsDB no publica.
# =============================================================================

_ESTADISTICAS_FOOTBALLDATA = [
    pa.field("arbitro",                     pa.large_string()),
    pa.field("goles_local_entretiempo",     pa.int64()),
    pa.field("goles_visitante_entretiempo", pa.int64()),
    pa.field("tiros_local",                 pa.int64()),
    pa.field("tiros_visitante",             pa.int64()),
    pa.field("tiros_arco_local",            pa.int64()),
    pa.field("tiros_arco_visitante",        pa.int64()),
    pa.field("faltas_local",                pa.int64()),
    pa.field("faltas_visitante",            pa.int64()),
    pa.field("corners_local",               pa.int64()),
    pa.field("corners_visitante",           pa.int64()),
    pa.field("amarillas_local",             pa.int64()),
    pa.field("amarillas_visitante",         pa.int64()),
    pa.field("rojas_local",                 pa.int64()),
    pa.field("rojas_visitante",             pa.int64()),
    pa.field("cuota_local",                 pa.float64()),
    pa.field("cuota_empate",                pa.float64()),
    pa.field("cuota_visitante",             pa.float64()),
    # Cuotas del mercado (promedio de casas): habilitan el baseline adversario.
    pa.field("cuota_over25",                pa.float64()),
    pa.field("cuota_under25",               pa.float64()),
    pa.field("cuota_local_mercado",         pa.float64()),
    pa.field("cuota_empate_mercado",        pa.float64()),
    pa.field("cuota_visitante_mercado",     pa.float64()),
]

_SCHEMA_PARTIDOS_FOOTBALLDATA = pa.schema(
    list(_SCHEMA_PARTIDOS) + _ESTADISTICAS_FOOTBALLDATA
)


def inicializar_tabla_partidos_footballdata(ruta: str) -> None:
    """Crea la tabla Delta de partidos de football-data.co.uk si no existe.

    Particiona por (id_liga, temporada): es la unidad natural de carga —un CSV
    por liga y temporada—, lo que permite re-ingerir una temporada puntual
    sobreescribiendo solo su partición.
    """
    if tabla_delta_existe(ruta):
        return
    asegurar_directorio(ruta)
    DeltaTable.create(
        ruta,
        schema=_SCHEMA_PARTIDOS_FOOTBALLDATA,
        partition_by=["id_liga", "temporada"],
        description=(
            "Partidos historicos de football-data.co.uk. Incluye estadisticas "
            "de partido y cuotas. Carga por liga-temporada (INSERT-OVERWRITE)."
        ),
    )
    DeltaTable(ruta).alter.add_constraint({
        "ck_fd_partidos_id_no_nulo":     "id_evento IS NOT NULL",
        "ck_fd_partidos_equipos_no_nulos":
            "equipo_local IS NOT NULL AND equipo_visitante IS NOT NULL",
    })
    logger.info("Tabla 'partidos_footballdata' inicializada en '%s'.", ruta)
