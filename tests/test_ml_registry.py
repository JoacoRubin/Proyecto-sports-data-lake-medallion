# =============================================================================
# tests/test_ml_registry.py — Tests del registro de modelos y experimentos.
#
# Un modelo sin metadata es un archivo binario que nadie puede auditar: no se
# sabe con qué features se entrenó, cuándo, ni qué métricas dio. Cuando en
# producción prediga algo raro, no hay forma de reconstruir el contexto.
#
# El registro guarda el modelo Y su ficha, versionados juntos.
# =============================================================================

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from ml.registry import (
    cargar_modelo,
    guardar_modelo,
    leer_experimentos,
    listar_versiones,
    registrar_experimento,
)


@pytest.fixture
def modelo_entrenado():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [0.0, 1.0, 0.0, 1.0]})
    y = pd.Series([0, 1, 0, 1])
    return DummyClassifier(strategy="prior").fit(X, y), X


def _metadata(**extra):
    base = {
        "modelo": "logistico",
        "features": ["a", "b"],
        "n_train": 100,
        "prior": 0.168,
        "metricas": {"pr_auc": 0.2134, "lift_pr_auc": 1.269},
    }
    base.update(extra)
    return base


# --- Guardar y cargar ------------------------------------------------------
def test_el_modelo_cargado_predice_igual_que_el_original(tmp_path, modelo_entrenado):
    modelo, X = modelo_entrenado
    guardar_modelo(modelo, "expulsiones", _metadata(), str(tmp_path))

    cargado, _ = cargar_modelo("expulsiones", str(tmp_path))
    np.testing.assert_allclose(
        modelo.predict_proba(X)[:, 1], cargado.predict_proba(X)[:, 1]
    )


def test_la_metadata_sobrevive_al_viaje(tmp_path, modelo_entrenado):
    modelo, _ = modelo_entrenado
    guardar_modelo(modelo, "expulsiones", _metadata(), str(tmp_path))

    _, meta = cargar_modelo("expulsiones", str(tmp_path))
    assert meta["features"] == ["a", "b"]
    assert meta["metricas"]["pr_auc"] == pytest.approx(0.2134)
    assert meta["modelo"] == "logistico"


def test_la_metadata_registra_cuando_se_entreno(tmp_path, modelo_entrenado):
    """Sin fecha no se puede saber si el modelo quedó viejo."""
    modelo, _ = modelo_entrenado
    guardar_modelo(modelo, "expulsiones", _metadata(), str(tmp_path))
    _, meta = cargar_modelo("expulsiones", str(tmp_path))
    assert "guardado_en" in meta and meta["guardado_en"]


# --- Versionado ------------------------------------------------------------
def test_cada_guardado_crea_una_version_nueva(tmp_path, modelo_entrenado):
    """Nunca se pisa un modelo anterior: se puede volver atrás."""
    modelo, _ = modelo_entrenado
    v1 = guardar_modelo(modelo, "expulsiones", _metadata(), str(tmp_path))
    v2 = guardar_modelo(modelo, "expulsiones", _metadata(), str(tmp_path))

    assert v1 != v2
    assert set(listar_versiones("expulsiones", str(tmp_path))) == {v1, v2}


def test_por_defecto_se_carga_la_ultima_version(tmp_path, modelo_entrenado):
    modelo, _ = modelo_entrenado
    guardar_modelo(modelo, "expulsiones", _metadata(nota="vieja"), str(tmp_path))
    guardar_modelo(modelo, "expulsiones", _metadata(nota="nueva"), str(tmp_path))

    _, meta = cargar_modelo("expulsiones", str(tmp_path))
    assert meta["nota"] == "nueva"


def test_se_puede_cargar_una_version_puntual(tmp_path, modelo_entrenado):
    modelo, _ = modelo_entrenado
    v1 = guardar_modelo(modelo, "expulsiones", _metadata(nota="vieja"), str(tmp_path))
    guardar_modelo(modelo, "expulsiones", _metadata(nota="nueva"), str(tmp_path))

    _, meta = cargar_modelo("expulsiones", str(tmp_path), version=v1)
    assert meta["nota"] == "vieja"


def test_cargar_un_modelo_inexistente_falla_claro(tmp_path):
    with pytest.raises(FileNotFoundError, match="expulsiones"):
        cargar_modelo("expulsiones", str(tmp_path))


# --- Tabla de experimentos -------------------------------------------------
def _folds():
    return pd.DataFrame([
        {"fold": 0, "temporada_test": "2023-2024", "n_test": 1752, "n_positivos": 290,
         "pr_auc": 0.1937, "lift_pr_auc": 1.170, "roc_auc": 0.58,
         "log_loss": 0.446, "brier": 0.138, "tasa_base": 0.166},
        {"fold": 1, "temporada_test": "2024-2025", "n_test": 1752, "n_positivos": 287,
         "pr_auc": 0.1914, "lift_pr_auc": 1.169, "roc_auc": 0.58,
         "log_loss": 0.446, "brier": 0.138, "tasa_base": 0.164},
    ])


def test_el_experimento_queda_registrado_con_una_fila_por_fold(tmp_path):
    ruta = str(tmp_path / "experimentos")
    registrar_experimento(_folds(), _metadata(), ruta, version="v1")

    guardado = leer_experimentos(ruta)
    assert len(guardado) == 2


def test_el_experimento_guarda_la_trazabilidad(tmp_path):
    """Sin saber qué features y qué versión, un número de métrica no sirve."""
    ruta = str(tmp_path / "experimentos")
    registrar_experimento(_folds(), _metadata(), ruta, version="v1")

    g = leer_experimentos(ruta)
    for col in ("version", "modelo", "features", "ejecutado_en", "pr_auc", "fold"):
        assert col in g.columns, f"falta la columna de trazabilidad '{col}'"
    assert set(g["version"]) == {"v1"}


def test_los_experimentos_se_acumulan_no_se_pisan(tmp_path):
    """Es el historial: comparar corridas es todo el punto de tener la tabla."""
    ruta = str(tmp_path / "experimentos")
    registrar_experimento(_folds(), _metadata(), ruta, version="v1")
    registrar_experimento(_folds(), _metadata(), ruta, version="v2")

    g = leer_experimentos(ruta)
    assert len(g) == 4
    assert set(g["version"]) == {"v1", "v2"}


def test_leer_experimentos_inexistentes_devuelve_vacio(tmp_path):
    """El dashboard no debe romperse si todavía no se entrenó nada."""
    assert leer_experimentos(str(tmp_path / "no_existe")).empty
