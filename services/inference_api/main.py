# =============================================================================
# services/inference_api/main.py — Sirve los modelos entrenados como HTTP.
#
# POR QUÉ ESTE SERVICIO EXISTE
# -----------------------------
# `ml/inference.py` ya resuelve la parte difícil: reusa el mismo
# `construir_features` del entrenamiento para que un partido futuro nunca vea
# un camino de feature engineering distinto (train/serve skew, ver el
# docstring de ese módulo). Pero antes de este archivo nada lo invocaba fuera
# de sus propios tests: el dashboard (app.py) solo mostraba la ficha y el
# historial de cada modelo, nunca predecía un partido nuevo.
#
# Esta capa es deliberadamente delgada: valida el borde HTTP y delega TODO el
# trabajo de dominio en ml/inference.py y ml/registry.py. No reimplementa
# carga de modelos ni feature engineering — ejecutar el entrenamiento sigue
# siendo trabajo exclusivo de `pipelines/ml_pipeline.py` (ver ese módulo):
# este servicio solo lee modelos ya aprobados y registrados.
#
# POR QUÉ UNA RUTA GENÉRICA Y NO ENDPOINTS COPIADOS POR MODELO
# ---------------------------------------------------------------
# El resto del proyecto ya resolvió este mismo problema (dos modelos, una
# sola maquinaria) con un runner genérico y dos configuraciones — ver
# `pipelines/ml_pipeline.py` y `app.py::_seccion_modelo`. Esta API sigue el
# mismo patrón: `/predictions/{modelo}` y `/models/{modelo}`, no cuatro rutas.
#
# Correr:  uvicorn services.inference_api.main:app --reload
# =============================================================================

import time

from fastapi import Depends, FastAPI, HTTPException

from ml.inference import PartidoAPredecir, predecir
from ml.registry import cargar_modelo
from services.inference_api.dependencies import get_dir_modelos, get_historico
from services.inference_api.schemas import PartidoRequest, PrediccionResponse

# Los mismos nombres que config.NOMBRE_MODELO_EXPULSIONES / NOMBRE_MODELO_GOLES.
_MODELOS_VALIDOS = {"expulsiones", "goles"}

app = FastAPI(
    title="Football ML — Inference API",
    description=(
        "Sirve los modelos de expulsiones y over 2.5 goles entrenados por "
        "pipelines/ml_pipeline.py. Entrenamiento e inferencia estan separados: "
        "este servicio nunca entrena, solo carga modelos ya aprobados."
    ),
)


def _validar_nombre_modelo(modelo: str) -> None:
    if modelo not in _MODELOS_VALIDOS:
        raise HTTPException(
            status_code=404,
            detail=f"Modelo '{modelo}' no existe. Validos: {sorted(_MODELOS_VALIDOS)}.",
        )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/models/{modelo}")
def model_info(modelo: str, dir_modelos: str = Depends(get_dir_modelos)) -> dict:
    """La ficha completa del modelo registrada por `ml.registry.guardar_modelo`."""
    _validar_nombre_modelo(modelo)
    try:
        _, metadata = cargar_modelo(modelo, dir_modelos)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return metadata


@app.post("/predictions/{modelo}", response_model=PrediccionResponse)
def predict(
    modelo: str,
    partido: PartidoRequest,
    dir_modelos: str = Depends(get_dir_modelos),
    historico=Depends(get_historico),
) -> PrediccionResponse:
    _validar_nombre_modelo(modelo)

    inicio = time.perf_counter()
    try:
        salida = predecir(
            [PartidoAPredecir(**partido.model_dump())],
            modelo, dir_modelos, historico=historico,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    tiempo_ms = (time.perf_counter() - inicio) * 1000

    fila = salida.iloc[0]
    return PrediccionResponse(
        liga=fila["liga"],
        fecha_partido=fila["fecha_partido"],
        equipo_local=fila["equipo_local"],
        equipo_visitante=fila["equipo_visitante"],
        probabilidad=float(fila["probabilidad"]),
        # Umbral 0.5 fijo por ahora: el proyecto reporta la probabilidad
        # calibrada, no una decision dura (ver README, seccion class_weight).
        # Si alguna vez hace falta otro punto de corte, que sea explicito y
        # medido, no un default silencioso aca.
        prediccion=bool(fila["probabilidad"] >= 0.5),
        target=fila["target"],
        modelo=fila["modelo"],
        version_modelo=fila["version_modelo"],
        tiempo_ms=tiempo_ms,
    )
