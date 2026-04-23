# =============================================================================
# ENTREGA1 + ENTREGA2 - Extraccion, Almacenamiento y Procesamiento de Datos
# Alumno: Joaquin Rubinstein
# =============================================================================
#
# API seleccionada: TheSportsDB (https://www.thesportsdb.com/)
# ---------------------------------------------------------------
# Elegí TheSportsDB porque es una API pública, gratuita y bien documentada
# sobre deportes. Ofrece datos de ligas, equipos y partidos de todo el mundo.
# Para este trabajo utilicé la liga "Argentinian Primera Division", temporada
# 2022, con la API key pública "3".
#
# Endpoints utilizados:
# ---------------------------------------------------------------
# 1. /search_all_teams.php?l={liga}
#    Devuelve todos los equipos de una liga con sus metadatos: nombre,
#    estadio, ciudad, año de fundación, sitio web, etc.
#
# 2. /eventsseason.php?id={liga_id}&s={temporada}
#    Devuelve todos los partidos de una temporada con sus resultados:
#    equipos, goles, fecha, estado del partido, etc.
#
# TP1 — Extracción y almacenamiento (capa bronze):
# ---------------------------------------------------------------
# - Equipos  -> FULL (overwrite total):
#   Los metadatos son datos estáticos que no cambian durante la temporada.
#
# - Partidos -> INCREMENTAL (INSERT-OVERWRITE por fecha):
#   Particionados por fecha_extraccion; se sobreescribe solo la partición
#   del día actual, preservando el historial sin duplicar datos.
#
# Elegí Delta Lake por soporte ACID, versionado y escrituras atómicas.
#
# TP2 — Procesamiento y enriquecimiento (capas silver y gold):
# ---------------------------------------------------------------
#   T1 | Deduplicacion       — registro más reciente de cada id_evento
#   T2 | Manejo de nulos     — elimina filas inútiles, rellena opcionales
#   T3 | Conversion de tipos — fecha a datetime64, goles a int64
#   T4 | Columnas derivadas  — resultado, diferencia_goles, es_goleada
#   T5 | Renombrado          — normaliza nombres ambiguos
#   T6 | JOIN / enriquec.    — ciudad y capacidad del estadio del local
#   T7 | Agregaciones        — tabla de posiciones por equipo (GROUP BY)
#
# Estructura del data lake:
#   data/bronze/thesportsdb/equipos/             - datos crudos (FULL)
#   data/bronze/thesportsdb/partidos/            - datos crudos (INCREMENTAL)
#   data/silver/thesportsdb/partidos_procesados/ - partidos limpios + enriquecidos
#   data/gold/thesportsdb/estadisticas_equipos/  - tabla de posiciones
#
# Modulos:
#   config.py                  - variables de entorno y rutas
#   utils/api.py               - cliente HTTP con retry/backoff
#   utils/delta.py             - helpers de lectura/escritura Delta Lake
#   bronze/extractors.py       - extraccion desde la API
#   bronze/schemas.py          - definicion de tablas y constraints
#   bronze/loaders.py          - persistencia en bronze
#   silver/transformations.py  - transformaciones T1-T6
#   gold/aggregations.py       - agregaciones T7
#   pipelines/bronze_pipeline.py
#   pipelines/silver_pipeline.py
#   pipelines/gold_pipeline.py
# =============================================================================

import subprocess
import sys

def _instalar_dependencias() -> None:
    paquetes = ["requests", "pandas", "deltalake", "pyarrow", "python-dotenv"]
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet"] + paquetes
    )

_instalar_dependencias()

from pipelines import bronze_pipeline, silver_pipeline, gold_pipeline

if __name__ == "__main__":
    bronze_pipeline.ejecutar()
    silver_pipeline.ejecutar()
    gold_pipeline.ejecutar()
