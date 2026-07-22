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

## 🚀 Deploy en Streamlit Community Cloud

El dashboard se despliega solo; **genera sus propios datos** en el primer arranque
(corre el pipeline batch contra la API pública). No necesita base de datos externa.

1. Subí el repo a GitHub (ya está en `origin`):
   ```bash
   git add -A && git commit -m "deploy: dashboard streamlit" && git push
   ```
2. Entrá a **https://share.streamlit.io** e iniciá sesión con GitHub.
3. **New app** → elegí el repo `Proyecto-sports-data-lake-medallion`, branch `main`,
   y como *Main file path*: `app.py`.
4. En **Advanced settings** elegí **Python 3.12** (no 3.14, que aún no está soportado).
5. Deploy. En el primer arranque tarda unos segundos generando el data lake; después
   queda cacheado.

> **Secrets (opcional):** el proyecto usa la API key pública `"3"` por defecto (5
> partidos y 10 equipos por liga). Si tenés una key premium de TheSportsDB, cargala
> en *App → Settings → Secrets*:
> ```toml
> THESPORTSDB_API_KEY = "tu_key_premium"
> ```

---

## 💻 Correr en local

```bash
# 1. Solo el dashboard (genera los datos en el primer arranque)
python -m pip install -r requirements.txt
streamlit run app.py

# 2. Pipeline orquestado con Prefect (Camino B)
python -m pip install -r requirements-orchestration.txt
python JoaquinRubinstein_TP2.py           # o: python -m pipelines.flow

# 3. Streaming Kafka (extra, requiere .env con credenciales Aiven)
python -m pip install -r requirements-streaming.txt
python producer_thesportsdb.py

# 4. Todo + tests
python -m pip install -r requirements-dev.txt
pytest
```

---

## 🗂️ Estructura

```
bronze/         extractores, mappers (esquema canónico), schemas y loaders Delta
silver/         transformaciones T1–T6 (funciones puras)
gold/           agregaciones T7 (tabla de posiciones)
quality/        contratos de datos (Pandera) entre capas
streaming/      sink Kafka → bronze (micro-batch + MERGE idempotente)
pipelines/      bronze/silver/gold + flow de Prefect (orquestación)
tests/          suite pytest (transformaciones, agregaciones, contratos, sink)
app.py          dashboard Streamlit
config.py       configuración (ligas, rutas, API key)
```

## 🧪 Tests

```bash
pytest        # 22 tests: transformaciones, agregaciones, contratos y sink
```
