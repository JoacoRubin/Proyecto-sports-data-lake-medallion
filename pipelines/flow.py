# =============================================================================
# pipelines/flow.py — Orquestación del pipeline medallion con Prefect (Camino B).
#
# Reemplaza la ejecución "a mano" (llamar ejecutar() de cada capa en secuencia)
# por un flujo orquestado que aporta:
#   - Dependencias explícitas: gold no corre si silver falló; silver no corre si
#     bronze falló (fail-fast). El orden es parte de la definición, no del azar.
#   - Reintentos por tarea: bronze reintenta ante fallos transitorios de la API;
#     silver/gold reintentan ante bloqueos momentáneos de Delta.
#   - Observabilidad: cada tarea tiene estado, duración y logs propios. Con
#     `prefect server start` se ve el DAG y el historial de corridas en la UI.
#
# Ejecutar:  python -m pipelines.flow
#        o:  python JoaquinRubinstein_TP2.py  (usa este flow internamente)
# =============================================================================

from prefect import flow, task, get_run_logger

from pipelines import (
    bronze_pipeline, silver_pipeline, gold_pipeline,
    footballdata_pipeline, ml_pipeline,
)


@task(name="bronze", retries=2, retry_delay_seconds=5)
def tarea_bronze() -> None:
    """Ingesta multi-liga desde la API a la capa bronze. Reintenta ante fallos de red."""
    bronze_pipeline.ejecutar()


@task(name="silver", retries=1, retry_delay_seconds=5)
def tarea_silver() -> None:
    """Transformaciones bronze → silver, con validación de contrato de calidad."""
    silver_pipeline.ejecutar()


@task(name="gold", retries=1, retry_delay_seconds=5)
def tarea_gold() -> None:
    """Agregaciones silver → gold (tabla de posiciones), con validación de invariantes."""
    gold_pipeline.ejecutar()


@flow(name="medallion-thesportsdb", log_prints=True)
def pipeline_medallion() -> None:
    """Orquesta bronze → silver → gold. El orden secuencial impone las dependencias:
    si una tarea falla (tras agotar reintentos), las siguientes no se ejecutan."""
    logger = get_run_logger()
    logger.info("=== Pipeline medallion — inicio ===")
    tarea_bronze()
    tarea_silver()
    tarea_gold()
    logger.info("=== Pipeline medallion — completado ===")


# =============================================================================
# Rama de datos historicos + ML.
#
# Va en un flow aparte, no colgada del medallion, porque tiene otra CADENCIA:
# el medallion se corre seguido (datos frescos de la API), mientras que la
# carga historica de football-data son ~10 anios que casi no cambian.
# Mezclarlas obligaria a bajar 50 CSVs en cada corrida sin necesidad.
# =============================================================================

@task(name="footballdata", retries=2, retry_delay_seconds=10)
def tarea_footballdata() -> None:
    """Ingesta historica de football-data.co.uk a bronze. Idempotente."""
    footballdata_pipeline.ejecutar()


@task(name="ml-expulsiones", retries=1, retry_delay_seconds=5)
def tarea_ml_expulsiones() -> None:
    """Entrena, evalua y registra el modelo de expulsiones."""
    ml_pipeline.ejecutar()


@task(name="ml-goles", retries=1, retry_delay_seconds=5)
def tarea_ml_goles() -> None:
    """Entrena el modelo de over 2.5 goles y lo compara contra el MERCADO.

    Los dos modelos son tareas hermanas y no encadenadas: comparten la ingesta
    pero ninguno depende del otro. Si el de expulsiones falla, este igual corre.
    """
    ml_pipeline.ejecutar_goles()


@flow(name="ml-modelos", log_prints=True)
def pipeline_ml() -> None:
    """Ingesta historica -> los dos modelos.

    La ingesta va primero porque ambos la necesitan; los modelos despues, uno
    al lado del otro.
    """
    logger = get_run_logger()
    logger.info("=== Pipeline ML - inicio ===")
    tarea_footballdata()
    tarea_ml_expulsiones()
    tarea_ml_goles()
    logger.info("=== Pipeline ML - completado ===")


@flow(name="pipeline-completo", log_prints=True)
def pipeline_completo() -> None:
    """Corre todo: el medallion de TheSportsDB y la rama historica + ML."""
    pipeline_medallion()
    pipeline_ml()


if __name__ == "__main__":
    pipeline_medallion()
