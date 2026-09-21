# =============================================================================
# services/inference_api/schemas.py — Contrato HTTP de entrada y salida.
#
# PartidoRequest espeja PartidoAPredecir (ml/inference.py) campo a campo: es
# el mismo dato, solo que llega por HTTP en vez de instanciarse en Python. No
# se reinventa la validación de negocio acá, solo se tipa el borde.
# =============================================================================

from pydantic import BaseModel, ConfigDict


class PartidoRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    liga: str
    temporada: str
    equipo_local: str
    equipo_visitante: str
    fecha_partido: str
    hora_partido: str | None = None
    cuota_local: float | None = None
    cuota_empate: float | None = None
    cuota_visitante: float | None = None
    cuota_over25: float | None = None
    cuota_under25: float | None = None
    cuota_local_mercado: float | None = None
    cuota_empate_mercado: float | None = None
    cuota_visitante_mercado: float | None = None


class PrediccionResponse(BaseModel):
    liga: str
    fecha_partido: str
    equipo_local: str
    equipo_visitante: str
    probabilidad: float
    prediccion: bool
    target: str
    modelo: str
    version_modelo: str
    tiempo_ms: float
