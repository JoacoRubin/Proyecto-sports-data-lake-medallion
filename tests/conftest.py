# =============================================================================
# tests/conftest.py — Fixtures compartidas de la capa ML.
#
# El modelo entrenado y el historico sintetico los necesitaban solo los tests
# de ml/inference.py; ahora tambien los necesitan los del servicio FastAPI que
# lo expone (services/inference_api). Es la MISMA garantia —mismo codigo de
# features, mismo registro— la que hay que probar en los dos lugares, asi que
# vive en un solo sitio en vez de duplicarse.
# =============================================================================

import pandas as pd
import pytest

from ml.inference import PartidoAPredecir


def _historico(n=40):
    """Historico sintetico con dos temporadas y equipos repetidos."""
    equipos = ["Real Madrid", "Barcelona", "Sevilla", "Valencia"]
    filas = []
    for i in range(n):
        temporada = "2023-2024" if i < n // 2 else "2024-2025"
        filas.append({
            "id_evento": f"FD-{i:04d}", "nombre_evento": "x vs y",
            "temporada": temporada, "liga": "Spanish La Liga", "id_liga": "FD-SP1",
            "equipo_local": equipos[i % 4], "id_equipo_local": f"FD-{i % 4}",
            "equipo_visitante": equipos[(i + 1) % 4], "id_equipo_visitante": f"FD-{(i + 1) % 4}",
            "goles_local": i % 3, "goles_visitante": (i + 1) % 3,
            "fecha_partido": f"2024-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}",
            "hora_partido": "20:00", "estadio": None, "estado": "FT",
            "timestamp_extraccion": "2026-01-01T00:00:00+00:00",
            "fecha_extraccion": "2026-01-01",
            "faltas_local": 10 + i % 5, "faltas_visitante": 12,
            "amarillas_local": 2, "amarillas_visitante": 3,
            "rojas_local": 1 if i % 7 == 0 else 0, "rojas_visitante": 0,
            "tiros_local": 12, "tiros_visitante": 10,
            "tiros_arco_local": 4, "tiros_arco_visitante": 3,
            "corners_local": 5, "corners_visitante": 4,
            "goles_local_entretiempo": 0, "goles_visitante_entretiempo": 0,
            "cuota_local": 2.0, "cuota_empate": 3.4, "cuota_visitante": 3.8,
            "cuota_over25": 1.9, "cuota_under25": 1.9,
            "cuota_local_mercado": 2.0, "cuota_empate_mercado": 3.4,
            "cuota_visitante_mercado": 3.8,
        })
    return pd.DataFrame(filas)


def _partido(local="Real Madrid", visitante="Barcelona", fecha="2025-06-01"):
    return PartidoAPredecir(
        liga="Spanish La Liga", temporada="2024-2025",
        equipo_local=local, equipo_visitante=visitante, fecha_partido=fecha,
    )


@pytest.fixture
def modelo_guardado(tmp_path):
    """Entrena y registra un modelo de expulsiones real, como haria el pipeline."""
    from ml.dataset import columnas_features, preparar_matriz
    from ml.features import TARGET, agregar_target, construir_features
    from ml.registry import guardar_modelo
    from ml.training import modelo_logistico

    hist = _historico()
    prior = float(agregar_target(hist)[TARGET].mean())
    matriz = construir_features(hist, prior=prior)
    X, y = preparar_matriz(matriz, TARGET)
    modelo = modelo_logistico().fit(X, y)

    guardar_modelo(modelo, "expulsiones", {
        "modelo": "logistico", "target": TARGET, "prior": prior,
        "features": columnas_features(matriz, TARGET),
    }, str(tmp_path))
    return str(tmp_path), hist


@pytest.fixture
def modelo_goles_guardado(tmp_path):
    """Entrena y registra el modelo de goles en el MISMO dir_modelos que expulsiones.

    `tmp_path` es del mismo test cuando un test pide las dos fixtures a la vez
    (alcance por funcion): asi las dos fichas conviven en un solo directorio,
    igual que las deja `pipelines.ml_pipeline.ejecutar_todos` en produccion, y
    la API puede servir ambos modelos desde el mismo `dir_modelos`.
    """
    from ml.dataset import columnas_features, preparar_matriz
    from ml.features_goles import TARGET_GOLES, agregar_target_goles, construir_features_goles
    from ml.registry import guardar_modelo
    from ml.training import modelo_logistico

    hist = _historico()
    prior = float(agregar_target_goles(hist)[TARGET_GOLES].mean())
    matriz = construir_features_goles(hist, prior=prior, incluir_mercado=True)
    X, y = preparar_matriz(matriz, TARGET_GOLES)
    modelo = modelo_logistico().fit(X, y)

    guardar_modelo(modelo, "goles", {
        "modelo": "logistico", "target": TARGET_GOLES, "prior": prior,
        "features": columnas_features(matriz, TARGET_GOLES),
    }, str(tmp_path))
    return str(tmp_path), hist
