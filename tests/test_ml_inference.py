# =============================================================================
# tests/test_ml_inference.py — Prediccion de partidos que todavia no se jugaron.
#
# EL PROBLEMA: TRAIN/SERVE SKEW
#
# La tentacion es escribir un camino aparte que calcule las features del
# partido futuro. Es el error mas caro del ML en produccion: dos
# implementaciones de la misma logica que divergen de a poco, y el modelo
# termina recibiendo features distintas de las que vio al entrenar. Nadie se
# entera, porque no hay ningun error — solo predicciones cada vez peores.
#
# La solucion de este modulo: NO hay camino aparte. El partido futuro se
# agrega como una fila mas al historico, se corre EXACTAMENTE el mismo
# `construir_features`, y se predice sobre esa fila. Imposible que diverjan
# porque es el mismo codigo.
# =============================================================================

import numpy as np
import pandas as pd
import pytest

from ml.inference import PartidoAPredecir, predecir


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
    """Entrena y registra un modelo real, como haria el pipeline."""
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


# --- Contrato basico -------------------------------------------------------
def test_devuelve_una_fila_por_partido_pedido(modelo_guardado):
    dir_modelos, hist = modelo_guardado
    out = predecir([_partido(), _partido("Sevilla", "Valencia")],
                   "expulsiones", dir_modelos, historico=hist)
    assert len(out) == 2


def test_la_probabilidad_esta_en_rango(modelo_guardado):
    dir_modelos, hist = modelo_guardado
    out = predecir([_partido()], "expulsiones", dir_modelos, historico=hist)
    p = out["probabilidad"].iloc[0]
    assert 0.0 <= p <= 1.0


def test_devuelve_los_datos_del_partido_para_poder_leerlo(modelo_guardado):
    dir_modelos, hist = modelo_guardado
    out = predecir([_partido("Sevilla", "Valencia", "2025-07-01")],
                   "expulsiones", dir_modelos, historico=hist)
    fila = out.iloc[0]
    assert fila["equipo_local"] == "Sevilla"
    assert fila["equipo_visitante"] == "Valencia"
    assert fila["fecha_partido"] == "2025-07-01"


def test_informa_con_que_version_del_modelo_se_predijo(modelo_guardado):
    """Sin la version, una prediccion vieja no se puede auditar."""
    dir_modelos, hist = modelo_guardado
    out = predecir([_partido()], "expulsiones", dir_modelos, historico=hist)
    assert out["version_modelo"].iloc[0]
    assert out["target"].iloc[0] == "hubo_expulsion"


# --- LA garantia: mismo codigo que en entrenamiento ------------------------
def test_usa_exactamente_las_features_con_las_que_se_entreno(modelo_guardado):
    """Si el orden o el conjunto difiere, el modelo recibe basura en silencio."""
    dir_modelos, hist = modelo_guardado
    from ml.registry import cargar_modelo

    _, meta = cargar_modelo("expulsiones", dir_modelos)
    out = predecir([_partido()], "expulsiones", dir_modelos,
                   historico=hist, devolver_features=True)
    usadas = [c for c in out.columns if c.startswith("feat__")]
    assert [c.removeprefix("feat__") for c in usadas] == meta["features"]


def test_el_partido_futuro_no_altera_el_historico(modelo_guardado):
    """Predecir no puede modificar el DataFrame que recibe."""
    dir_modelos, hist = modelo_guardado
    antes = hist.copy()
    predecir([_partido()], "expulsiones", dir_modelos, historico=hist)
    pd.testing.assert_frame_equal(hist, antes)


def test_las_features_salen_del_historial_de_los_equipos(modelo_guardado):
    """Dos partidos con equipos distintos no pueden dar features identicas."""
    dir_modelos, hist = modelo_guardado
    out = predecir(
        [_partido("Real Madrid", "Barcelona"), _partido("Sevilla", "Valencia")],
        "expulsiones", dir_modelos, historico=hist, devolver_features=True,
    )
    feats = [c for c in out.columns if c.startswith("feat__")]
    assert not out[feats].iloc[0].equals(out[feats].iloc[1])


# --- Casos de borde --------------------------------------------------------
def test_un_equipo_desconocido_no_revienta(modelo_guardado):
    """Un ascendido no tiene historia: las medias moviles quedan nulas."""
    dir_modelos, hist = modelo_guardado
    out = predecir([_partido("Equipo Recien Ascendido", "Barcelona")],
                   "expulsiones", dir_modelos, historico=hist)
    assert 0.0 <= out["probabilidad"].iloc[0] <= 1.0


def test_sin_partidos_devuelve_vacio(modelo_guardado):
    dir_modelos, hist = modelo_guardado
    assert predecir([], "expulsiones", dir_modelos, historico=hist).empty


def test_modelo_inexistente_falla_claro(tmp_path):
    with pytest.raises(FileNotFoundError, match="expulsiones"):
        predecir([_partido()], "expulsiones", str(tmp_path), historico=_historico())
