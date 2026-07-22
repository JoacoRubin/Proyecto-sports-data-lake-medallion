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
# La key "3" es la clave pública de test de TheSportsDB: sirve como default para
# que el proyecto funcione sin configuración (por ej. en un deploy). Para datos
# completos se puede definir THESPORTSDB_API_KEY con una key premium.
API_KEY  = os.getenv("THESPORTSDB_API_KEY", "3")
URL_BASE = f"https://www.thesportsdb.com/api/v1/json/{API_KEY}"

LIGA_NOMBRE = os.getenv("THESPORTSDB_LIGA")
LIGA_ID     = os.getenv("THESPORTSDB_LIGA_ID")
TEMPORADA   = os.getenv("THESPORTSDB_TEMPORADA")

# -----------------------------------------------------------------------------
# Ligas del pipeline batch (multi-liga).
#
# La API key pública "3" limita eventsseason a 5 partidos y search_all_teams a
# 10 equipos POR liga. Para enriquecer el data lake se ingiere un conjunto de
# ligas: cada una aporta sus 5 partidos y sus 10 equipos.
#
# Ojo con la temporada: las ligas sudamericanas / MLS son de año calendario
# ("2024"); las europeas cruzan año ("2024-2025").
# -----------------------------------------------------------------------------
LIGAS = [
    {"nombre": "Argentinian Primera Division", "id": "4406", "temporada": "2024"},
    {"nombre": "Brazilian Serie A",            "id": "4351", "temporada": "2024"},
    {"nombre": "American Major League Soccer",  "id": "4346", "temporada": "2024"},
    {"nombre": "English Premier League",        "id": "4328", "temporada": "2024-2025"},
    {"nombre": "Spanish La Liga",               "id": "4335", "temporada": "2024-2025"},
    {"nombre": "Italian Serie A",               "id": "4332", "temporada": "2024-2025"},
    {"nombre": "German Bundesliga",             "id": "4331", "temporada": "2024-2025"},
    {"nombre": "French Ligue 1",                "id": "4334", "temporada": "2024-2025"},
    {"nombre": "Portuguese Primeira Liga",      "id": "4344", "temporada": "2024-2025"},
]

# Rutas bronze
DIR_EQUIPOS_BRONZE  = "data/bronze/thesportsdb/equipos"
DIR_PARTIDOS_BRONZE = "data/bronze/thesportsdb/partidos"

# Rutas silver
DIR_PARTIDOS_SILVER = "data/silver/thesportsdb/partidos_procesados"

# Rutas gold
DIR_ESTADISTICAS_GOLD = "data/gold/thesportsdb/estadisticas_equipos"
