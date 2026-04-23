# =============================================================================
# utils/delta.py — Utilidades para leer y escribir tablas Delta Lake
# =============================================================================

import os
import logging

import pandas as pd
import pyarrow as pa
from deltalake import DeltaTable, write_deltalake

logger = logging.getLogger(__name__)


def asegurar_directorio(ruta: str) -> None:
    """Crea el directorio y sus padres si no existen."""
    os.makedirs(ruta, exist_ok=True)


def tabla_delta_existe(ruta: str) -> bool:
    """Verifica si existe una tabla Delta válida en la ruta."""
    return DeltaTable.is_deltatable(ruta)


def normalizar_tipos_para_delta(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte columnas object/category a string para compatibilidad con Delta Lake."""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype in ("object", "category"):
            df[col] = df[col].astype("string")
    return df


def guardar_en_delta(
    df: pd.DataFrame,
    ruta: str,
    modo: str = "overwrite",
    columnas_particion: list | None = None,
    predicado: str | None = None,
    modo_esquema: str = "overwrite",
) -> None:
    """Guarda un DataFrame en Delta Lake.

    Args:
        df: datos a guardar.
        ruta: directorio de la tabla Delta.
        modo: "overwrite", "append", "error" o "ignore".
        columnas_particion: columnas de partición (opcional).
        predicado: expresión para INSERT-OVERWRITE selectivo (opcional).
        modo_esquema: "overwrite" o "merge".
    """
    asegurar_directorio(ruta)
    write_deltalake(
        ruta,
        normalizar_tipos_para_delta(df),
        mode=modo,
        partition_by=columnas_particion,
        predicate=predicado,
        schema_mode=modo_esquema,
    )


def leer_tabla_delta(ruta: str) -> pd.DataFrame:
    """Lee todos los datos de una tabla Delta Lake.

    Raises:
        FileNotFoundError: si la tabla no existe.
    """
    if not tabla_delta_existe(ruta):
        raise FileNotFoundError(f"No se encontró la tabla Delta en '{ruta}'.")
    return DeltaTable(ruta).to_pandas()


def mostrar_resumen_tabla(ruta: str, nombre: str) -> None:
    """Imprime un resumen de una tabla Delta Lake."""
    sep = "-" * 62
    if not tabla_delta_existe(ruta):
        print(f"\n{sep}\n [{nombre}] La tabla no existe en '{ruta}'.\n{sep}")
        return
    tabla = DeltaTable(ruta)
    df    = tabla.to_pandas()
    print(f"\n{sep}")
    print(f" Tabla    : {nombre}")
    print(f" Ruta     : {ruta}")
    print(f" Version  : {tabla.version()} | Filas: {len(df):,} | Columnas: {df.shape[1]}")
    print(sep)
    print(df.head(10).to_string(index=False))
    print()


def merge_en_delta(
    df: pd.DataFrame,
    ruta: str,
    predicado_merge: str,
    columnas_particion: list | None = None,
    modo_esquema: str = "merge",
) -> None:
    """MERGE (upsert) en una tabla Delta existente.

    Actualiza registros que coinciden con el predicado e inserta los nuevos.
    Si la tabla todavía no existe, la crea con overwrite (primera carga).

    Args:
        df: datos a fusionar.
        ruta: directorio de la tabla Delta.
        predicado_merge: condición JOIN entre src y tgt (ej. "src.id = tgt.id").
        columnas_particion: particiones aplicadas solo en la creación inicial.
        modo_esquema: "merge" o "overwrite" para evolución de esquema.
    """
    df_norm = normalizar_tipos_para_delta(df)
    if tabla_delta_existe(ruta):
        dt = DeltaTable(ruta)
        (
            dt.merge(
                source=pa.Table.from_pandas(df_norm),
                source_alias="src",
                target_alias="tgt",
                predicate=predicado_merge,
            )
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute()
        )
    else:
        asegurar_directorio(ruta)
        write_deltalake(
            ruta,
            df_norm,
            mode="overwrite",
            partition_by=columnas_particion,
            schema_mode=modo_esquema,
        )
