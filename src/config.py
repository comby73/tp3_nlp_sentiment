"""
config.py
Configuración global del proyecto: rutas, parámetros y constantes.
Según consigna TP3: define parámetros centralizados para todo el pipeline.
"""

from pathlib import Path

# Rutas base del proyecto
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figuras"

# Nombres de archivos
TRAINING_FILE = "training.1600000.processed.noemoticon.csv"
TEST_FILE = "testdata.manual.2009.06.14.csv"

# Semilla aleatoria para reproducibilidad
RANDOM_SEED = 42

# Parámetros de preprocesamiento
STOP_WORDS_LANGUAGE = "english"
MIN_WORD_LENGTH = 2

# Parámetros de vectorización
MAX_FEATURES_TFIDF = 5000
MIN_DF = 2
MAX_DF = 0.95

# Parámetros de Word2Vec
W2V_VECTOR_SIZE = 100
W2V_WINDOW = 5
W2V_MIN_COUNT = 5
W2V_WORKERS = 4
W2V_EPOCHS = 10

# Columnas del dataset
COL_POLARITY = "polarity"
COL_ID = "id"
COL_DATE = "date"
COL_QUERY = "query"
COL_USER = "user"
COL_TEXT = "text"

COLUMN_NAMES = [COL_POLARITY, COL_ID, COL_DATE, COL_QUERY, COL_USER, COL_TEXT]

# Mapeo de polaridad
POLARITY_MAP = {
    0: "negative",
    2: "neutral",
    4: "positive"
}

# Asegurar que los directorios existen
for directory in [PROCESSED_DATA_DIR, FIGURES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
