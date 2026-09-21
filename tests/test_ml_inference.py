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

import pandas as pd
import pytest

from conftest import _historico, _partido
from ml.inference import predecir


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
