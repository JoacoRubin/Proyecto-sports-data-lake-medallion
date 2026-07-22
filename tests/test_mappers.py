# =============================================================================
# tests/test_mappers.py — Tests del mapeo canónico bronze (fuente compartida
# entre ingesta batch y streaming).
# =============================================================================

from bronze.mappers import mapear_equipo, mapear_partido

_EQUIPO_RAW = {
    "idTeam": "133602", "strTeam": "River Plate", "strAlternate": "River",
    "strCountry": "Argentina", "strCity": "Buenos Aires", "strLeague": "Primera",
    "idLeague": "4406", "strStadium": "Monumental", "intStadiumCapacity": "83214",
    "strStadiumLocation": "Nunez", "intFormedYear": "1901",
    "strDescriptionES": "desc", "strWebsite": "cariverplate.com.ar",
}

_PARTIDO_RAW = {
    "idEvent": "1", "strEvent": "River vs Boca", "strSeason": "2022",
    "strLeague": "Primera", "idLeague": "4406", "strHomeTeam": "River",
    "idHomeTeam": "133602", "intHomeScore": "2", "strAwayTeam": "Boca",
    "idAwayTeam": "133600", "intAwayScore": "1", "dateEvent": "2022-05-15",
    "strTime": "20:00:00", "strVenue": "Monumental", "strStatus": "Match Finished",
}


def test_mapear_equipo_produce_esquema_bronze_completo():
    d = mapear_equipo(_EQUIPO_RAW, "2022-01-01T00:00:00+00:00")
    assert d["id_equipo"] == "133602"
    assert d["nombre"] == "River Plate"
    assert d["capacidad_estadio"] == "83214"
    assert d["timestamp_extraccion"] == "2022-01-01T00:00:00+00:00"
    # 14 campos: el esquema bronze de equipos.
    assert len(d) == 14


def test_mapear_partido_produce_esquema_bronze_completo():
    d = mapear_partido(_PARTIDO_RAW, "2022-01-01T00:00:00+00:00", "2022-01-01")
    assert d["id_evento"] == "1"
    assert d["id_equipo_local"] == "133602"
    assert d["goles_local"] == "2"
    assert d["fecha_extraccion"] == "2022-01-01"
    # 17 campos: el esquema bronze de partidos.
    assert len(d) == 17
