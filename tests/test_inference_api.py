# =============================================================================
# tests/test_inference_api.py — Contrato HTTP del servicio de inferencia.
#
# QUE PRUEBA ESTE ARCHIVO Y QUE NO
# ---------------------------------
# El feature engineering, el train/serve skew y la carga del modelo ya estan
# probados en test_ml_inference.py y test_ml_registry.py sobre ml/inference.py
# y ml/registry.py. Este servicio es una capa HTTP fina encima de esos dos
# modulos: los tests de aca verifican el CONTRATO (rutas, codigos de estado,
# forma del JSON), no vuelven a probar el feature engineering.
#
# Los dependency overrides de FastAPI apuntan `dir_modelos` e `historico` a
# los mismos fixtures sinteticos que usa ml/inference.py, para no levantar
# Delta Lake ni entrenar contra el dataset real en cada test.
# =============================================================================

import pytest
from fastapi.testclient import TestClient

from services.inference_api.dependencies import get_dir_modelos, get_historico
from services.inference_api.main import app


@pytest.fixture
def cliente(modelo_guardado, modelo_goles_guardado):
    dir_modelos, hist = modelo_guardado  # mismo tmp_path que modelo_goles_guardado
    app.dependency_overrides[get_dir_modelos] = lambda: dir_modelos
    app.dependency_overrides[get_historico] = lambda: hist
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def cliente_sin_modelos(tmp_path):
    """Ningun modelo entrenado todavia — el estado de un deploy nuevo."""
    from conftest import _historico
    app.dependency_overrides[get_dir_modelos] = lambda: str(tmp_path)
    app.dependency_overrides[get_historico] = lambda: _historico()
    yield TestClient(app)
    app.dependency_overrides.clear()


_PARTIDO_JSON = {
    "liga": "Spanish La Liga",
    "temporada": "2024-2025",
    "equipo_local": "Real Madrid",
    "equipo_visitante": "Barcelona",
    "fecha_partido": "2025-06-01",
}


# --- /health -----------------------------------------------------------------
def test_health_no_depende_de_ningun_modelo(cliente_sin_modelos):
    r = cliente_sin_modelos.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# --- /models/{modelo} ---------------------------------------------------------
def test_model_info_devuelve_la_ficha(cliente):
    r = cliente.get("/models/expulsiones")
    assert r.status_code == 200
    body = r.json()
    assert body["modelo"] == "logistico"
    assert body["target"] == "hubo_expulsion"
    assert "version" in body and body["version"]


def test_model_info_sin_modelo_entrenado_da_404(cliente_sin_modelos):
    r = cliente_sin_modelos.get("/models/expulsiones")
    assert r.status_code == 404


def test_model_info_nombre_invalido_da_404(cliente):
    r = cliente.get("/models/no-existe")
    assert r.status_code == 404


# --- POST /predictions/{modelo} -----------------------------------------------
def test_predice_expulsiones(cliente):
    r = cliente.post("/predictions/expulsiones", json=_PARTIDO_JSON)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["probabilidad"] <= 1.0
    assert isinstance(body["prediccion"], bool)
    assert body["modelo"] == "expulsiones"
    assert body["version_modelo"]


def test_predice_goles(cliente):
    r = cliente.post("/predictions/goles", json=_PARTIDO_JSON)
    assert r.status_code == 200
    assert 0.0 <= r.json()["probabilidad"] <= 1.0


def test_la_prediccion_sigue_al_umbral_de_la_probabilidad(cliente):
    body = cliente.post("/predictions/expulsiones", json=_PARTIDO_JSON).json()
    assert body["prediccion"] == (body["probabilidad"] >= 0.5)


def test_predecir_con_nombre_de_modelo_invalido_da_404(cliente):
    r = cliente.post("/predictions/no-existe", json=_PARTIDO_JSON)
    assert r.status_code == 404


def test_predecir_sin_modelo_entrenado_da_404(cliente_sin_modelos):
    r = cliente_sin_modelos.post("/predictions/expulsiones", json=_PARTIDO_JSON)
    assert r.status_code == 404


def test_predecir_sin_campos_obligatorios_da_422(cliente):
    r = cliente.post("/predictions/expulsiones", json={"liga": "Spanish La Liga"})
    assert r.status_code == 422


def test_predecir_acepta_cuotas_opcionales(cliente):
    """Las cuotas son opcionales en PartidoAPredecir: un partido sin mercado
    todavia debe poder predecirse (ver ml/inference.py)."""
    partido = {**_PARTIDO_JSON, "cuota_local": 2.1, "cuota_empate": 3.3, "cuota_visitante": 3.6}
    r = cliente.post("/predictions/goles", json=partido)
    assert r.status_code == 200


def test_un_equipo_sin_historia_no_revienta_la_api(cliente):
    """Un ascendido: mismo caso de borde que cubre test_ml_inference.py, pero
    a traves del contrato HTTP."""
    partido = {**_PARTIDO_JSON, "equipo_local": "Equipo Recien Ascendido"}
    r = cliente.post("/predictions/expulsiones", json=partido)
    assert r.status_code == 200
    assert 0.0 <= r.json()["probabilidad"] <= 1.0
