# =============================================================================
# config.py — Configuración global del proyecto
# =============================================================================

import os
import logging

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# API
API_KEY  = os.getenv("THESPORTSDB_API_KEY")
URL_BASE = f"https://www.thesportsdb.com/api/v1/json/{API_KEY}"

LIGA_NOMBRE = os.getenv("THESPORTSDB_LIGA")
LIGA_ID     = os.getenv("THESPORTSDB_LIGA_ID")
TEMPORADA   = os.getenv("THESPORTSDB_TEMPORADA")

# Rutas bronze
DIR_EQUIPOS_BRONZE  = "data/bronze/thesportsdb/equipos"
DIR_PARTIDOS_BRONZE = "data/bronze/thesportsdb/partidos"

# Rutas silver
DIR_PARTIDOS_SILVER = "data/silver/thesportsdb/partidos_procesados"

# Rutas gold
DIR_ESTADISTICAS_GOLD = "data/gold/thesportsdb/estadisticas_equipos"
