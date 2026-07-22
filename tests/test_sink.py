# =============================================================================
# tests/test_sink.py — Test de integración del sink de streaming (Camino A).
#
# Ejercita el loop Kafka->bronze SIN Kafka: se alimenta el sink con mensajes
# (formato bronze completo) y se verifica la tabla Delta resultante.
# =============================================================================

from deltalake import DeltaTable

from streaming.sink import BronzeSink


def _equipo(id_equipo: str, nombre: str) -> dict:
    """Mensaje de equipo con el esquema bronze completo (como lo emite el producer)."""
    return {
        "id_equipo": id_equipo, "nombre": nombre, "nombre_alternativo": None,
        "pais": "Argentina", "ciudad": "Buenos Aires", "liga": "Primera",
        "id_liga": "4406", "estadio": "Estadio", "capacidad_estadio": "50000",
        "ubicacion_estadio": "BA", "anio_fundacion": "1901",
        "descripcion_es": None, "sitio_web": None,
        "timestamp_extraccion": "2022-01-01T00:00:00+00:00",
    }


def _nuevo_sink(tmp_path, tam_microbatch=10):
    return BronzeSink(
        ruta_equipos=str(tmp_path / "equipos"),
        ruta_partidos=str(tmp_path / "partidos"),
        topico_equipos="equipos",
        topico_partidos="partidos",
        tam_microbatch=tam_microbatch,
    )


def test_sink_persiste_equipos_en_bronze(tmp_path):
    sink = _nuevo_sink(tmp_path)
    sink.agregar("equipos", _equipo("1", "River"))
    sink.agregar("equipos", _equipo("2", "Boca"))
    sink.cerrar()

    df = DeltaTable(str(tmp_path / "equipos")).to_pandas()
    assert set(df["id_equipo"]) == {"1", "2"}
    assert df.loc[df["id_equipo"] == "1", "nombre"].iloc[0] == "River"


def test_sink_merge_es_idempotente(tmp_path):
    """Reprocesar el mismo id no duplica filas (MERGE upsert)."""
    sink = _nuevo_sink(tmp_path, tam_microbatch=1)  # flush inmediato por mensaje
    sink.agregar("equipos", _equipo("1", "River"))
    sink.agregar("equipos", _equipo("1", "River Plate"))  # mismo id, otro nombre
    sink.cerrar()

    df = DeltaTable(str(tmp_path / "equipos")).to_pandas()
    filas = df[df["id_equipo"] == "1"]
    assert len(filas) == 1                       # no se duplicó
    assert filas.iloc[0]["nombre"] == "River Plate"  # última versión gana


def test_sink_flush_por_tamano_de_microbatch(tmp_path):
    """Al alcanzar el tamaño del micro-batch se escribe sin esperar a cerrar."""
    sink = _nuevo_sink(tmp_path, tam_microbatch=2)
    sink.agregar("equipos", _equipo("1", "River"))
    sink.agregar("equipos", _equipo("2", "Boca"))  # dispara el flush

    df = DeltaTable(str(tmp_path / "equipos")).to_pandas()
    assert len(df) == 2  # ya persistido antes de cerrar
