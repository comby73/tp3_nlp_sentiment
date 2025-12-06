"""
data_loading.py
Funciones para cargar y guardar datos (CSV, pickles, modelos).

Según consigna TP3:
- SOLO carga y guardado de datos.
- NO hace lógica de modelos, preprocesamiento, ni análisis.
- Maneja los datasets de tweets y objetos serializados.
"""

import pandas as pd
from typing import Optional
import pickle

from src.config import (
    RAW_DATA_DIR, 
    PROCESSED_DATA_DIR, 
    TRAINING_FILE, 
    TEST_FILE,
    COLUMN_NAMES
)


def load_raw_training_data(nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Carga el dataset de entrenamiento desde data/raw/.
    
    Args:
        nrows: Número de filas a cargar (None para cargar todas).
    
    Returns:
        DataFrame con el dataset de entrenamiento.
    """
    file_path = RAW_DATA_DIR / TRAINING_FILE
    
    if not file_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {file_path}")
    
    df = pd.read_csv(
        file_path,
        encoding="latin-1",
        names=COLUMN_NAMES,
        nrows=nrows
    )
    
    print(f"✓ Dataset de entrenamiento cargado: {len(df)} filas")
    return df


def load_raw_test_data(nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Carga el dataset de test desde data/raw/.
    
    Args:
        nrows: Número de filas a cargar (None para cargar todas).
    
    Returns:
        DataFrame con el dataset de test.
    """
    file_path = RAW_DATA_DIR / TEST_FILE
    
    if not file_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {file_path}")
    
    df = pd.read_csv(
        file_path,
        encoding="latin-1",
        names=COLUMN_NAMES,
        nrows=nrows
    )
    
    print(f"✓ Dataset de test cargado: {len(df)} filas")
    return df


def save_processed_data(df: pd.DataFrame, filename: str) -> None:
    """
    Guarda un DataFrame procesado en data/processed/.
    
    Args:
        df: DataFrame a guardar.
        filename: Nombre del archivo (con extensión .csv).
    """
    file_path = PROCESSED_DATA_DIR / filename
    df.to_csv(file_path, index=False)
    print(f"✓ Datos guardados en: {file_path}")


def load_processed_data(filename: str) -> pd.DataFrame:
    """
    Carga un DataFrame procesado desde data/processed/.
    
    Args:
        filename: Nombre del archivo (con extensión .csv).
    
    Returns:
        DataFrame procesado.
    """
    file_path = PROCESSED_DATA_DIR / filename
    
    if not file_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {file_path}")
    
    df = pd.read_csv(file_path)
    print(f"✓ Datos procesados cargados: {len(df)} filas")
    return df


def save_object(obj, filename: str) -> None:
    """
    Guarda un objeto Python en data/processed/ usando pickle.
    
    Args:
        obj: Objeto a guardar.
        filename: Nombre del archivo (con extensión .pkl).
    """
    file_path = PROCESSED_DATA_DIR / filename
    
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)
    
    print(f"✓ Objeto guardado en: {file_path}")


def load_object(filename: str):
    """
    Carga un objeto Python desde data/processed/.
    
    Args:
        filename: Nombre del archivo (con extensión .pkl).
    
    Returns:
        Objeto cargado.
    """
    file_path = PROCESSED_DATA_DIR / filename
    
    if not file_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {file_path}")
    
    with open(file_path, 'rb') as f:
        obj = pickle.load(f)
    
    print(f"✓ Objeto cargado desde: {file_path}")
    return obj
