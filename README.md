# ⚽ TheSportsDB — Data Lake (arquitectura medallion)

Pipeline de ingeniería de datos sobre la API pública de **TheSportsDB**, con
arquitectura **medallion** (bronze → silver → gold), calidad de datos, orquestación
y un dashboard interactivo.

- **Bronze**: ingesta multi-liga desde la API → Delta Lake (crudo).
- **Silver**: limpieza, tipado, columnas derivadas y JOIN con equipos.
- **Gold**: agregaciones → tabla de posiciones por liga.
- **Calidad**: contratos Pandera en las fronteras (fallan ruidosamente ante datos corruptos).
- **Orquestación**: flow de Prefect (tareas con reintentos y dependencias).
- **Dashboard**: Streamlit + Plotly (tabla, KPIs, bubble chart).
- **Extra**: streaming con Kafka (Aiven) que ingesta a la misma capa bronze.

---

