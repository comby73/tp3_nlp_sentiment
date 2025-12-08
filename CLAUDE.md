# CLAUDE.md - AI Assistant Guide for TP3 NLP Sentiment Analysis

> **Last Updated:** 2025-12-08
> **Repository:** tp3_nlp_sentiment
> **Purpose:** Comprehensive guide for AI assistants working with this NLP sentiment analysis codebase

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Code Organization](#code-organization)
4. [Development Workflow](#development-workflow)
5. [Coding Conventions](#coding-conventions)
6. [Common Tasks](#common-tasks)
7. [Testing Requirements](#testing-requirements)
8. [Git Workflow](#git-workflow)
9. [Important Constraints](#important-constraints)
10. [Quick Reference](#quick-reference)

---

## 🎯 Project Overview

### What This Project Does
A production-ready sentiment analysis system for Twitter data (Sentiment140 dataset) following CRISP-DM methodology. The project classifies tweets as Positive/Negative using a Linear SVM model trained on 1.6M tweets with **85.18% F1-score**.

### Key Technologies
- **ML/NLP:** scikit-learn (SVM, LogReg, RF), Gensim (Word2Vec), NLTK (preprocessing)
- **Pre-trained models:** BERT/RoBERTa (Transformers), TextBlob, VADER
- **Data:** Pandas, NumPy, SciPy (sparse matrices)
- **Visualization:** Matplotlib, Seaborn, Plotly, WordCloud
- **Hardware Context:** Trained on Intel i9 + 128GB RAM + NVIDIA RTX 4080 SUPER

### Project Status
- **Phase:** Complete (all 12 notebooks executed, models trained, reports generated)
- **Production Ready:** Yes (`predict_sentiment.py` for standalone inference)
- **Documentation:** Comprehensive README.md with results and methodology

---

## 📂 Repository Structure

### Directory Tree with Purpose

```
tp3_nlp_sentiment/
├── 📁 data/                          # All datasets (raw → processed → vectorized)
│   ├── raw/                          # Original Sentiment140 CSVs (250MB excluded from git)
│   │   └── testdata.manual.2009.06.14.csv  # 359 labeled test tweets
│   ├── processed/                    # Cleaned CSVs with 7 numeric features
│   │   └── test_processed.csv
│   ├── vectorized/                   # Serialized TF-IDF matrices & labels
│   │   ├── X_test.pkl (53K)
│   │   ├── y_test.pkl (3K)
│   │   ├── y_train.pkl (13M)         # Large due to 1.6M samples
│   │   └── tfidf_vectorizer.pkl (426K)
│   └── predictions/                  # Model output CSVs
│       └── testdata_predictions.csv
│
├── 📁 models/                        # Trained ML models (46MB total)
│   ├── best_model_linear_svm.pkl     # Production SVM (79K)
│   ├── model_metrics.pkl             # Model metadata (accuracy, F1, etc.)
│   └── word2vec_model.pkl (46M)      # Word2Vec embeddings (100 dims)
│
├── 📁 notebooks/                     # 12 sequential CRISP-DM notebooks
│   ├── 01_eda.ipynb                  # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb        # Text cleaning + feature engineering
│   ├── 03_vectorizacion.ipynb        # TF-IDF & Word2Vec training
│   ├── 04_modelado.ipynb             # Model training & selection (4 algorithms)
│   ├── 05_experimentos_hiperparametros.ipynb  # GridSearchCV tuning
│   ├── 06_analisis_errores.ipynb     # Error analysis + confidence scores
│   ├── 07_analisis_sesgos.ipynb      # Bias audit (by text length, metadata)
│   ├── 08_prediccion.ipynb           # Model inference pipeline
│   ├── 09_comparacion_modelos_preentrenados.ipynb  # Benchmark vs industry
│   ├── 10_sopa_letras.ipynb          # Word2Vec semantic word search game
│   ├── 11_word2vec_tetris.ipynb      # Word2Vec Tetris with analogies
│   └── 12_dashboard_interactivo.ipynb  # Interactive Plotly dashboard
│
├── 📁 src/                           # Reusable Python modules (2,128 LOC)
│   ├── __init__.py                   # Package initialization
│   ├── config.py (59 lines)          # Centralized configuration & paths
│   ├── data_loading.py (144 lines)   # CSV/pickle I/O operations
│   ├── preprocessing.py (315 lines)  # Text cleaning & tokenization
│   ├── features.py (281 lines)       # TF-IDF, BoW, Word2Vec vectorization
│   ├── models.py (322 lines)         # Model training (5 algorithms)
│   ├── evaluation.py (389 lines)     # Metrics + PMI + cosine similarity
│   └── visualization.py (613 lines)  # Matplotlib/Plotly charts
│
├── 📁 reports/                       # Generated analysis outputs
│   ├── dashboard_interactivo.html (4.4M)  # Standalone Plotly dashboard
│   ├── eda_summary.json              # EDA statistics
│   └── figuras/                      # 20+ PNG visualizations
│
├── 📁 tests/                         # Unit test suite
│   └── test_preprocessing.py (151 lines, 22 test cases)
│
├── predict_sentiment.py (10.9K)      # Standalone production script
├── requirements.txt                  # 31 Python dependencies
├── README.md (12.5K)                 # Main documentation
└── .gitignore                        # Excludes >100MB files
```

### Critical Files for AI Assistants

| File | Purpose | When to Modify |
|------|---------|----------------|
| `src/config.py` | Global constants (paths, seeds, hyperparameters) | Adding new paths or parameters |
| `src/preprocessing.py` | Text cleaning pipeline | Changing text cleaning logic |
| `src/models.py` | Model training/inference | Adding new algorithms |
| `tests/test_preprocessing.py` | Unit tests | After modifying preprocessing |
| `predict_sentiment.py` | Production inference script | API changes needed |
| `requirements.txt` | Dependency management | Adding new libraries |

---

## 🏗️ Code Organization

### Module Dependency Hierarchy

```
config.py (ROOT - constants & paths)
    ↓
data_loading.py (CSV/pickle I/O)
    ↓
preprocessing.py (NLTK-based text cleaning)
    ↓
features.py (TF-IDF, BoW, Word2Vec)
    ├→ models.py (Classification algorithms)
    └→ evaluation.py (Metrics: PMI, cosine similarity)
         ↓
visualization.py (Matplotlib/Seaborn/Plotly)
```

### Module Summaries

#### `config.py` - Configuration Hub
- **Purpose:** Centralized constants for reproducibility
- **Key Exports:**
  - Paths: `PROJECT_ROOT`, `DATA_DIR`, `RAW_DATA_DIR`, `PROCESSED_DATA_DIR`, `MODELS_DIR`, `REPORTS_DIR`
  - Seeds: `RANDOM_SEED = 42` (used throughout for reproducibility)
  - Hyperparameters: `MAX_FEATURES_TFIDF=5000`, `MIN_DF=2`, `MAX_DF=0.95`
  - Word2Vec config: `VECTOR_SIZE=100`, `WINDOW=5`, `MIN_COUNT=5`, `EPOCHS=10`
  - Dataset schema: `COLUMNS` (6 Sentiment140 columns)
  - Polarity mapping: `POLARITY_MAP = {0: 'Negative', 4: 'Positive'}`

#### `data_loading.py` - I/O Operations
- **Purpose:** Load/save CSV and pickle files
- **Key Functions:**
  - `load_raw_training_data()` → DataFrame (1.6M tweets, encoding=latin-1)
  - `load_processed_data()` → DataFrame with cleaned text + 7 numeric features
  - `load_object(pkl_path)` → Deserialize models/vectorizers
  - `save_object(obj, pkl_path)` → Serialize objects with joblib
- **Important:** Uses `latin-1` encoding for Sentiment140 compatibility

#### `preprocessing.py` - Text Cleaning Pipeline
- **Purpose:** Transform raw tweets into clean text + numeric features
- **Core Functions:**
  - `clean_text(text)` → Lowercased, URLs/mentions removed, hashtags converted, normalized
  - `extract_urls(text)` → List[str] (5.1% of corpus has URLs)
  - `extract_mentions(text)` → List[str] (46.2% of corpus has @mentions)
  - `extract_hashtags(text)` → List[str] (2.2% of corpus has hashtags)
  - `detect_intensified_words(text)` → int (count of repeated chars: "loooove")
  - `calculate_uppercase_ratio(text)` → float (% uppercase chars)
  - `apply_stemming(tokens)` → List[str] (PorterStemmer: running→run)
  - `apply_lemmatization(tokens)` → List[str] (WordNetLemmatizer: better→good)
  - `remove_stopwords(tokens)` → List[str] (KEEPS: "not", "no", "very")
  - `preprocess_dataframe(df)` → DataFrame with 7 added features
- **7 Numeric Features Extracted:**
  1. `length` (character count)
  2. `word_count` (token count)
  3. `num_hashtags`
  4. `num_mentions`
  5. `num_urls`
  6. `uppercase_count` (ALL CAPS words)
  7. `uppercase_percentage` (% uppercase chars)

#### `features.py` - Vectorization
- **Purpose:** Convert text to numerical representations
- **Key Functions:**
  - `create_tfidf_features(train_texts, test_texts, max_features=5000)` → Sparse matrices
    - **Config:** `ngram_range=(1,2)` (unigrams + bigrams for negations)
    - **Sparsity:** 99.89% zeros (efficient storage)
  - `create_bow_features(train_texts, test_texts)` → Count-based vectors
  - `train_word2vec_model(texts, vector_size=100, window=5)` → Gensim Word2Vec
  - `create_word2vec_features(texts, model)` → 100-dim dense embeddings
  - `get_top_tfidf_words(vectorizer, feature_names, n=20)` → Dict[class, List[words]]
  - `analyze_word_analogies(model, word_pairs)` → Tests algebraic relationships
  - `get_similar_words(model, word, topn=10)` → List[(word, similarity)]

#### `models.py` - Classification
- **Purpose:** Train and manage ML models
- **Supported Algorithms:**
  1. `train_logistic_regression()` → LogisticRegression (F1: 84.2%)
  2. `train_svm()` → LinearSVC (F1: 85.18% ⭐ **SELECTED**)
  3. `train_naive_bayes()` → MultinomialNB (F1: 78.5%)
  4. `train_random_forest()` → RandomForestClassifier (F1: 81.3%)
  5. `train_decision_tree()` → DecisionTreeClassifier (F1: 72.1%)
- **Key Functions:**
  - `train_and_evaluate(model, X_train, y_train, X_test, y_test)` → Dict[metrics]
  - `predict_proba(model, X)` → Confidence scores
  - `save_model(model, path)` → Serialize with joblib
  - `load_model(path)` → Deserialize
- **Selection Criterion:** F1-score (balanced metric for sentiment classification)

#### `evaluation.py` - Metrics & Analysis
- **Purpose:** Calculate performance metrics + special NLP metrics (PMI, cosine similarity)
- **Standard Metrics:**
  - `calculate_accuracy(y_true, y_pred)` → float
  - `calculate_f1(y_true, y_pred, average='weighted')` → float
  - `get_confusion_matrix(y_true, y_pred)` → 2×2 matrix
- **Special Metrics (TP3 Requirements):**
  - `calculate_cosine_similarity(vec1, vec2)` → float (-1 to 1)
    - Measures pairwise similarity between Word2Vec embeddings
    - Used for intra-class cohesion analysis
  - `calculate_pmi(word, class, corpus)` → float
    - **Formula:** log₂(P(word,class) / P(word)×P(class))
    - Measures how discriminative a word is for a sentiment class
    - High PMI = strongly associated with that sentiment
- **Comparison:**
  - `compare_models(models_results)` → DataFrame (sorted by F1)

#### `visualization.py` - Plotting
- **Purpose:** Generate charts for notebooks and reports
- **Key Functions:**
  - `plot_wordcloud_by_polarity(df)` → Side-by-side WordClouds
  - `plot_confusion_matrix(cm, class_names)` → Heatmap
  - `plot_temporal_heatmap(df)` → Hour-of-day sentiment patterns
  - `plot_embeddings_2d(embeddings, labels)` → t-SNE/UMAP projections
  - `plot_metrics_comparison(models_results)` → Bar chart
  - `plot_hashtag_frequency(df, top_n=20)` → Horizontal bar chart
  - `save_figure(fig, filename)` → Saves to `reports/figuras/`
- **Styling:** Uses Seaborn's `whitegrid` style + custom color palettes

---

## 🔄 Development Workflow

### CRISP-DM Pipeline (Sequential Execution)

```
PHASE 1: Data Understanding (Notebook 01)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: data/raw/training.1600000.processed.noemoticon.csv
Tasks:
  - Load dataset (1.6M tweets)
  - Analyze polarity distribution (50/50 balanced)
  - Generate WordClouds by sentiment
  - Temporal analysis (hour-of-day patterns)
  - Element analysis: URLs (5.1%), mentions (46.2%), hashtags (2.2%)
Output: reports/eda_summary.json, figures/

PHASE 2: Data Preparation (Notebook 02)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: Raw data
Tasks:
  - Apply preprocessing.clean_text()
  - Extract 7 numeric features
  - Verify no data leakage (test set isolated)
  - Save cleaned data
Output: data/processed/train_processed.csv, test_processed.csv

PHASE 3: Feature Engineering (Notebook 03)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: Processed text
Tasks:
  - Create TF-IDF features (10K dims, bigrams)
  - Train Word2Vec model (100 dims)
  - Serialize vectorizers for production use
Output: data/vectorized/*.pkl, models/word2vec_model.pkl

PHASE 4: Modeling (Notebook 04)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: X_train, y_train
Tasks:
  - Train 5 algorithms (LogReg, SVM, NB, RF, DT)
  - Compare F1-scores
  - Select best model (Linear SVM)
  - Re-train on full training data (train + validation)
  - Verify overfitting (compare train vs test metrics)
Output: models/best_model_linear_svm.pkl, model_metrics.pkl

PHASE 5: Evaluation (Notebooks 05-09)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Tasks:
  05: GridSearchCV hyperparameter tuning (C parameter)
  06: Error analysis (confidence scores, failure patterns)
  07: Bias audit (performance by text length, metadata)
  08: Production inference pipeline testing
  09: Benchmark vs pre-trained models (TextBlob, VADER, BERT)
Output: Figures, analysis reports

PHASE 6: Deployment & Demos (Notebooks 10-12)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Tasks:
  10: Word2Vec semantic word search game (with analogies mode)
  11: Word2Vec Tetris game (dynamic level generation)
  12: Interactive Plotly dashboard (HTML output)
Output: reports/dashboard_interactivo.html
```

### Data Flow Pattern

```python
# Standard workflow for adding new features:

# 1. EDA (01) - Discover patterns
pattern_found = analyze_data(df)

# 2. Preprocessing (02) - Add cleaning logic
def new_cleaning_function(text):
    # Implement new cleaning step
    return cleaned_text

# 3. Update tests (tests/)
class TestNewCleaning(unittest.TestCase):
    def test_new_cleaning(self):
        self.assertEqual(new_cleaning_function("test"), "expected")

# 4. Vectorization (03) - Extract features
def extract_new_feature(df):
    df['new_feature'] = df['text'].apply(new_cleaning_function)
    return df

# 5. Modeling (04) - Evaluate impact
X_train_new = vectorize_with_new_features(train_df)
model = train_svm(X_train_new, y_train)
metrics = evaluate(model, X_test, y_test)

# 6. If improvement: Update production script
# predict_sentiment.py → Add new feature to pipeline
```

---

## 📐 Coding Conventions

### Naming Standards

| Entity | Pattern | Examples |
|--------|---------|----------|
| **Functions** | `snake_case` with action verbs | `load_raw_training_data()`, `calculate_f1()`, `plot_confusion_matrix()` |
| **Classes** | `PascalCase` | `TestPreprocessing`, `SentimentPredictor` |
| **Variables** | `snake_case` | `X_train`, `y_pred`, `text_clean`, `model_results` |
| **Constants** | `UPPER_SNAKE_CASE` | `MAX_FEATURES_TFIDF`, `RANDOM_SEED`, `PROJECT_ROOT` |
| **Private functions** | `_leading_underscore` | `_normalize_text()`, `_validate_input()` |
| **Module files** | `lowercase_underscore.py` | `preprocessing.py`, `data_loading.py` |
| **Notebooks** | `##_descriptive_name.ipynb` | `01_eda.ipynb`, `12_dashboard_interactivo.ipynb` |

### Type Hints (Required)

```python
# Always use type hints for function signatures
from typing import List, Tuple, Optional, Dict, Any

def extract_urls(text: str) -> List[str]:
    """Extract all URLs from text."""
    return re.findall(r'http[s]?://\S+', text)

def create_tfidf_features(
    train_texts: List[str],
    test_texts: Optional[List[str]] = None,
    max_features: int = MAX_FEATURES_TFIDF,
) -> Tuple:
    """Create TF-IDF features for train and optional test sets."""
    # Implementation
    return (X_train, vectorizer) if test_texts is None else (X_train, X_test, vectorizer)

def compare_models(models_results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Compare multiple model results."""
    return pd.DataFrame(models_results).T.sort_values('f1', ascending=False)
```

### Docstring Style (Google-style with Spanish)

```python
def preprocess_dataframe(
    df: pd.DataFrame,
    text_column: str = 'text',
    apply_stemming_flag: bool = False,
) -> pd.DataFrame:
    """
    Preprocesa un DataFrame completo aplicando limpieza de texto y extracción de features.

    Args:
        df: DataFrame con columna de texto a procesar.
        text_column: Nombre de la columna que contiene el texto.
        apply_stemming_flag: Si True, aplica stemming después de la limpieza.

    Returns:
        DataFrame con columna 'text_clean' y 7 features numéricas adicionales:
        - length: Número de caracteres
        - word_count: Número de palabras
        - num_hashtags: Cantidad de hashtags
        - num_mentions: Cantidad de mentions (@usuario)
        - num_urls: Cantidad de URLs
        - uppercase_count: Palabras en mayúsculas
        - uppercase_percentage: Porcentaje de caracteres en mayúsculas

    Raises:
        KeyError: Si text_column no existe en df.
        ValueError: Si df está vacío.

    Example:
        >>> df = pd.DataFrame({'text': ['I love this! #great']})
        >>> df_processed = preprocess_dataframe(df)
        >>> print(df_processed.columns)
        ['text', 'text_clean', 'length', 'word_count', 'num_hashtags', ...]
    """
    # Implementation
```

### Import Organization (3-tier structure)

```python
# 1. Standard library (alphabetically)
import math
import pickle
import re
import string
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

# 2. Third-party libraries (alphabetically)
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# 3. Local modules (relative imports)
from src.config import MAX_FEATURES_TFIDF, RANDOM_SEED, PROJECT_ROOT
from src.data_loading import load_processed_data
from src.preprocessing import clean_text
```

### Print Formatting (Unicode indicators)

```python
# Use emojis for visual feedback in notebooks/scripts
print(f"✓ Dataset cargado: {len(df)} filas")           # Success
print(f"📊 MÉTRICAS DEL MODELO:")                       # Section headers
print(f"⚠️ Advertencia: {warning_message}")            # Warnings
print(f"❌ Error: {error_message}")                     # Errors
print(f"🚀 Entrenando modelo...")                       # Progress
print(f"💾 Guardando modelo en {path}...")              # I/O operations
```

### Error Handling Pattern

```python
def load_model(model_path: str) -> Any:
    """
    Carga un modelo serializado desde disco.

    Args:
        model_path: Ruta al archivo .pkl del modelo.

    Returns:
        Modelo deserializado.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        pickle.UnpicklingError: Si el archivo está corrupto.
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(f"❌ Modelo no encontrado: {model_path}")

    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"✓ Modelo cargado desde {model_path}")
        return model
    except pickle.UnpicklingError as e:
        raise pickle.UnpicklingError(f"❌ Error al deserializar {model_path}: {e}")
```

---

## 🛠️ Common Tasks

### Task 1: Add a New Preprocessing Step

```python
# 1. Implement function in src/preprocessing.py
def remove_emojis(text: str) -> str:
    """
    Elimina emojis del texto usando regex Unicode.

    Args:
        text: Texto de entrada.

    Returns:
        Texto sin emojis.
    """
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

# 2. Add test in tests/test_preprocessing.py
class TestPreprocessing(unittest.TestCase):
    def test_remove_emojis(self):
        text = "I love Python 🐍❤️"
        result = remove_emojis(text)
        self.assertEqual(result, "I love Python ")

    def test_remove_emojis_empty(self):
        result = remove_emojis("")
        self.assertEqual(result, "")

# 3. Run tests
# python -m unittest tests/test_preprocessing.py

# 4. Integrate into clean_text() pipeline
def clean_text(text: str) -> str:
    text = text.lower()
    text = remove_urls(text)
    text = remove_mentions(text)
    text = remove_emojis(text)  # ← NEW STEP
    text = convert_hashtags(text)
    # ... rest of pipeline
    return text

# 5. Update notebook 02_preprocessing.ipynb to use new function
# 6. Re-run pipeline from notebook 02 onwards
```

### Task 2: Add a New ML Model

```python
# 1. Implement in src/models.py
from sklearn.ensemble import GradientBoostingClassifier

def train_gradient_boosting(
    X_train,
    y_train,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    random_state: int = RANDOM_SEED,
) -> GradientBoostingClassifier:
    """
    Entrena un modelo Gradient Boosting para clasificación de sentimientos.

    Args:
        X_train: Matriz de features de entrenamiento.
        y_train: Etiquetas de entrenamiento.
        n_estimators: Número de árboles.
        learning_rate: Tasa de aprendizaje.
        random_state: Semilla para reproducibilidad.

    Returns:
        Modelo entrenado.
    """
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model

# 2. Add to notebook 04_modelado.ipynb
models = {
    'Logistic Regression': train_logistic_regression(X_train, y_train),
    'Linear SVM': train_svm(X_train, y_train),
    'Naive Bayes': train_naive_bayes(X_train, y_train),
    'Random Forest': train_random_forest(X_train, y_train),
    'Gradient Boosting': train_gradient_boosting(X_train, y_train),  # NEW
}

# 3. Evaluate and compare
results = {}
for name, model in models.items():
    y_pred = model.predict(X_test)
    results[name] = {
        'accuracy': calculate_accuracy(y_test, y_pred),
        'f1': calculate_f1(y_test, y_pred),
    }

df_results = compare_models(results)
print(df_results)
```

### Task 3: Generate a New Visualization

```python
# 1. Implement in src/visualization.py
def plot_feature_importance(
    model,
    feature_names: List[str],
    top_n: int = 20,
    figsize: Tuple[int, int] = (10, 8),
) -> None:
    """
    Visualiza la importancia de features para modelos tree-based.

    Args:
        model: Modelo entrenado con atributo feature_importances_.
        feature_names: Nombres de las features.
        top_n: Número de features a mostrar.
        figsize: Tamaño de la figura.

    Raises:
        AttributeError: Si el modelo no tiene feature_importances_.
    """
    if not hasattr(model, 'feature_importances_'):
        raise AttributeError("Modelo no tiene feature_importances_")

    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]

    plt.figure(figsize=figsize)
    plt.barh(range(top_n), importances[indices])
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('Importancia')
    plt.title(f'Top {top_n} Features Más Importantes')
    plt.tight_layout()

    save_figure(plt.gcf(), 'feature_importance.png')
    plt.show()

# 2. Use in notebook
from src.visualization import plot_feature_importance

rf_model = train_random_forest(X_train, y_train)
feature_names = tfidf_vectorizer.get_feature_names_out()
plot_feature_importance(rf_model, feature_names, top_n=30)
```

### Task 4: Run Full Pipeline from Scratch

```bash
# Prerequisites: Download Sentiment140 dataset
# Place training.1600000.processed.noemoticon.csv in data/raw/

# Execute notebooks in sequence
jupyter notebook notebooks/01_eda.ipynb                # EDA
jupyter notebook notebooks/02_preprocessing.ipynb      # Text cleaning
jupyter notebook notebooks/03_vectorizacion.ipynb      # TF-IDF + Word2Vec
jupyter notebook notebooks/04_modelado.ipynb           # Model training
jupyter notebook notebooks/05_experimentos_hiperparametros.ipynb  # Tuning
jupyter notebook notebooks/06_analisis_errores.ipynb   # Error analysis
jupyter notebook notebooks/07_analisis_sesgos.ipynb    # Bias audit
jupyter notebook notebooks/08_prediccion.ipynb         # Inference
jupyter notebook notebooks/09_comparacion_modelos_preentrenados.ipynb  # Benchmarks
jupyter notebook notebooks/10_sopa_letras.ipynb        # Word search game
jupyter notebook notebooks/11_word2vec_tetris.ipynb    # Tetris game
jupyter notebook notebooks/12_dashboard_interactivo.ipynb  # Dashboard

# Alternative: Run as Python scripts
jupyter nbconvert --to script notebooks/01_eda.ipynb
python notebooks/01_eda.py
```

### Task 5: Make a Single Prediction

```python
# Option A: Using production script
python predict_sentiment.py "I love this product!"
# Output: {"text": "I love this product!", "sentiment": "Positive", "confidence": 0.87}

# Option B: Programmatic usage
from predict_sentiment import SentimentPredictor

predictor = SentimentPredictor(
    model_dir='models',
    data_dir='data/vectorized'
)

result = predictor.predict_single("I hate waiting in line")
print(result)
# {'text': 'I hate waiting in line', 'sentiment': 'Negative', 'confidence': 0.92, 'features': {...}}

# Option C: Batch predictions
texts = ["Great product!", "Terrible service", "It's okay"]
results = predictor.predict_batch(texts)
for r in results:
    print(f"{r['text']} → {r['sentiment']} (confidence: {r['confidence']:.2f})")
```

### Task 6: Update Documentation

```bash
# 1. Update README.md with new findings
# - Add new metrics to results table
# - Document new features
# - Update installation instructions if dependencies changed

# 2. Update CLAUDE.md (this file)
# - Add new modules/functions to code organization
# - Document new conventions if introduced
# - Update task examples

# 3. Update notebook markdown cells
# - Explain methodology changes
# - Document hyperparameter choices
# - Add interpretation of results

# 4. Commit with descriptive message
git add README.md CLAUDE.md notebooks/
git commit -m "docs: Update documentation with new feature X"
```

---

## 🧪 Testing Requirements

### Test Suite Overview

```
tests/
└── test_preprocessing.py (151 lines, 22 test cases)
    ├── TestTextCleaning (11 tests)
    │   ├── test_clean_text_basic
    │   ├── test_remove_urls
    │   ├── test_remove_mentions
    │   ├── test_convert_hashtags
    │   ├── test_remove_numbers
    │   ├── test_remove_punctuation
    │   ├── test_normalize_intensified_words
    │   ├── test_clean_text_none
    │   ├── test_clean_text_empty
    │   ├── test_clean_text_stopwords_only
    │   └── test_clean_text_complete_pipeline
    │
    └── TestFeatureExtraction (11 tests)
        ├── test_extract_urls
        ├── test_extract_mentions
        ├── test_extract_hashtags
        ├── test_calculate_uppercase_ratio
        ├── test_detect_intensified_words
        ├── test_apply_stemming
        ├── test_apply_lemmatization
        ├── test_remove_stopwords
        ├── test_feature_extraction_none
        ├── test_feature_extraction_empty
        └── test_feature_extraction_edge_cases
```

### Running Tests

```bash
# Run all tests
python -m unittest discover tests/

# Run specific test file
python -m unittest tests/test_preprocessing.py

# Run specific test class
python -m unittest tests.test_preprocessing.TestTextCleaning

# Run specific test method
python -m unittest tests.test_preprocessing.TestTextCleaning.test_clean_text_basic

# Run with verbose output
python -m unittest tests/test_preprocessing.py -v

# Run with coverage (requires coverage.py)
pip install coverage
coverage run -m unittest discover tests/
coverage report
coverage html  # Generates htmlcov/index.html
```

### Test Writing Guidelines

```python
import unittest
from src.preprocessing import clean_text, extract_urls

class TestNewFeature(unittest.TestCase):
    """Test suite for new feature extraction function."""

    def setUp(self):
        """Set up test fixtures (runs before each test)."""
        self.sample_text = "I love Python! #coding"
        self.empty_text = ""

    def tearDown(self):
        """Clean up after tests (runs after each test)."""
        pass

    def test_basic_functionality(self):
        """Test basic use case."""
        result = clean_text(self.sample_text)
        self.assertIsInstance(result, str)
        self.assertNotIn('#', result)

    def test_edge_case_empty_input(self):
        """Test behavior with empty input."""
        result = clean_text(self.empty_text)
        self.assertEqual(result, "")

    def test_edge_case_none_input(self):
        """Test behavior with None input."""
        with self.assertRaises(AttributeError):
            clean_text(None)

    def test_expected_output(self):
        """Test specific expected output."""
        text = "Check out https://example.com @user #hashtag"
        result = clean_text(text)
        expected = "check out hashtag"
        self.assertEqual(result, expected)

    def test_multiple_assertions(self):
        """Test multiple conditions."""
        text = "HELLO world!!!"
        result = clean_text(text)
        self.assertNotIn('!', result)
        self.assertTrue(result.islower())
        self.assertGreater(len(result), 0)

if __name__ == '__main__':
    unittest.main()
```

### When to Write Tests

**ALWAYS write tests when:**
- Adding new preprocessing functions (text cleaning, feature extraction)
- Modifying existing preprocessing logic
- Adding data validation functions
- Implementing utility functions used across modules

**Tests NOT required for:**
- Notebook-specific exploratory code
- Visualization functions (manual inspection sufficient)
- One-off analysis scripts
- Model training code (use cross-validation instead)

### Test Coverage Goals

| Module | Current Coverage | Target |
|--------|------------------|--------|
| `preprocessing.py` | ~80% | 90% |
| `data_loading.py` | ~40% | 70% |
| `features.py` | ~30% | 60% |
| `models.py` | N/A | 50% (focus on I/O) |
| `evaluation.py` | N/A | 60% |

---

## 🔀 Git Workflow

### Branch Strategy

```bash
# Current branch (as specified in task context)
main branch: (not specified - ask user)
feature branch: claude/claude-md-miwyldw94nwjw4bl-01TgzgPAbU6EmEeiPMYZC96H

# Typical workflow
git checkout -b feature/add-new-model     # Create feature branch
# ... make changes ...
git add src/models.py tests/test_models.py
git commit -m "feat: Add Gradient Boosting classifier"
git push -u origin feature/add-new-model
```

### Commit Message Convention

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```bash
# Format: <type>(<scope>): <description>

# Types:
feat:     # New feature
fix:      # Bug fix
docs:     # Documentation changes
style:    # Code style (formatting, no logic change)
refactor: # Code restructuring (no behavior change)
test:     # Adding/updating tests
chore:    # Maintenance (dependencies, config)
perf:     # Performance improvement

# Examples:
git commit -m "feat(models): Add Gradient Boosting classifier"
git commit -m "fix(preprocessing): Handle empty tweets correctly"
git commit -m "docs: Update README with new model results"
git commit -m "test(preprocessing): Add edge case tests for URLs"
git commit -m "refactor(features): Extract TF-IDF config to constants"
git commit -m "chore: Update scikit-learn to 1.3.0"
```

### Recent Commits (Context)

```
edb5e4e chore: Remove informe_tp3.md (redundant with README)
7c0583d feat: Add interactive dashboard with Plotly (notebook 12)
d16367d docs: Add Word2Vec verification output
ffe6d08 docs: Add Word2Vec training documentation to 03_vectorizacion.ipynb
dc36a8c feat: Word2Vec Games - Tetris + Word Search with semantic engine
```

### Push Protocol (CRITICAL)

```bash
# ALWAYS use -u flag for new branches starting with 'claude/'
git push -u origin claude/claude-md-miwyldw94nwjw4bl-01TgzgPAbU6EmEeiPMYZC96H

# Branch naming convention: claude/<session-id>
# If push fails with 403, verify branch name matches session ID

# Retry logic for network failures (exponential backoff)
for attempt in 1 2 3 4; do
    git push -u origin <branch> && break
    sleep $((2 ** attempt))  # 2s, 4s, 8s, 16s
done
```

### Files to .gitignore

```gitignore
# Large data files (>100MB)
data/raw/training.1600000.processed.noemoticon.csv
data/processed/train_processed.csv
data/vectorized/X_train.pkl

# Python
__pycache__/
*.py[cod]
*$py.class
.ipynb_checkpoints/

# Models (if too large)
models/*.pkl
!models/best_model_linear_svm.pkl  # Keep production model

# Reports (optional)
reports/figuras/*.png

# Environment
venv/
.env
.vscode/
```

---

## ⚠️ Important Constraints

### 1. File Size Limitations

**GitHub Limit:** 100MB per file

**Excluded Files (must be regenerated):**

| File | Size | How to Regenerate |
|------|------|-------------------|
| `data/raw/training.1600000.processed.noemoticon.csv` | 250MB | Download from [Sentiment140](http://help.sentiment140.com/for-students) |
| `data/processed/train_processed.csv` | 267MB | Run `02_preprocessing.ipynb` |
| `data/vectorized/X_train.pkl` | 210MB | Run `03_vectorizacion.ipynb` |

**Workaround for AI assistants:**
- Work with test set only (359 tweets, <100KB)
- Use pre-trained models in `models/` directory
- Skip notebooks 01-03 if large files unavailable

### 2. Data Leakage Prevention

**CRITICAL RULES:**

```python
# ❌ WRONG: Fitting on combined data
vectorizer = TfidfVectorizer()
X_all = vectorizer.fit_transform(train_texts + test_texts)  # LEAKAGE!

# ✓ CORRECT: Fit on train, transform test
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)  # Uses train vocabulary

# ❌ WRONG: Using test statistics
mean = np.mean(np.concatenate([X_train, X_test]))  # LEAKAGE!

# ✓ CORRECT: Use train statistics only
mean = np.mean(X_train)
X_train_norm = X_train - mean
X_test_norm = X_test - mean  # Same normalization
```

**Verification Checklist:**
- [ ] TF-IDF fitted ONLY on training data
- [ ] No test labels used during hyperparameter tuning
- [ ] Feature engineering uses train statistics only
- [ ] Test set preprocessed identically to train (same functions)

### 3. Encoding Requirements

```python
# Sentiment140 dataset uses latin-1 encoding (NOT utf-8)
df = pd.read_csv('data/raw/training.csv', encoding='latin-1', header=None)

# Always specify encoding explicitly:
df.to_csv('output.csv', encoding='latin-1', index=False)
```

### 4. Reproducibility Requirements

```python
# ALWAYS set random seeds at start of notebooks/scripts
import random
import numpy as np
from src.config import RANDOM_SEED

random.seed(RANDOM_SEED)  # Python random module
np.random.seed(RANDOM_SEED)  # NumPy

# For scikit-learn models
model = LogisticRegression(random_state=RANDOM_SEED)

# For train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=RANDOM_SEED, stratify=y
)
```

### 5. Memory Considerations

**Large Dataset Handling:**

```python
# ✓ Efficient: Sparse matrix for TF-IDF (99.89% zeros)
from scipy.sparse import save_npz, load_npz

X_train_sparse = vectorizer.fit_transform(texts)  # csr_matrix
save_npz('X_train.npz', X_train_sparse)  # Compressed storage

# ❌ Memory-intensive: Dense conversion
X_train_dense = X_train_sparse.toarray()  # Avoid unless necessary

# ✓ Efficient: Batch processing for large datasets
def process_in_batches(df, batch_size=10000):
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        yield process_batch(batch)
```

### 6. Dependency Compatibility

```bash
# Known compatible versions (tested on Python 3.8+)
scikit-learn>=1.2.0,<2.0.0
pandas>=1.5.0,<2.0.0
numpy>=1.23.0,<2.0.0

# GPU acceleration for BERT (optional)
torch>=2.0.0  # Requires CUDA for GPU support
transformers>=4.30.0

# If conflicts arise, use virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 7. Special NLP Requirements (TP3 Grading)

**Mandatory Implementations:**

1. **PMI (Pointwise Mutual Information)**
   - Must be implemented in `evaluation.py`
   - Used to identify discriminative words per sentiment class
   - Formula: `PMI(word, class) = log₂(P(word|class) / P(word))`

2. **Cosine Similarity**
   - Must be implemented for Word2Vec embeddings
   - Used for intra-class cohesion analysis
   - Applied in notebooks 10-11 for word search games

3. **Word2Vec with 3 Operations**
   - `most_similar(word)` - Find semantically similar words
   - `similarity(word1, word2)` - Calculate cosine distance
   - `most_similar(positive=[A, C], negative=[B])` - Solve analogies

---

## 📖 Quick Reference

### Key File Paths

```python
from pathlib import Path
from src.config import (
    PROJECT_ROOT,        # /home/user/tp3_nlp_sentiment
    DATA_DIR,            # PROJECT_ROOT / 'data'
    RAW_DATA_DIR,        # DATA_DIR / 'raw'
    PROCESSED_DATA_DIR,  # DATA_DIR / 'processed'
    VECTORIZED_DATA_DIR, # DATA_DIR / 'vectorized'
    MODELS_DIR,          # PROJECT_ROOT / 'models'
    REPORTS_DIR,         # PROJECT_ROOT / 'reports'
    FIGURES_DIR,         # REPORTS_DIR / 'figuras'
)

# Production model
BEST_MODEL = MODELS_DIR / 'best_model_linear_svm.pkl'

# TF-IDF vectorizer
VECTORIZER = VECTORIZED_DATA_DIR / 'tfidf_vectorizer.pkl'

# Test data
TEST_DATA = RAW_DATA_DIR / 'testdata.manual.2009.06.14.csv'
TEST_PROCESSED = PROCESSED_DATA_DIR / 'test_processed.csv'
```

### Common Commands

```bash
# Environment setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download NLTK data (required for preprocessing)
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt')"

# Run tests
python -m unittest discover tests/

# Make prediction
python predict_sentiment.py "I love this product!"

# Start Jupyter
jupyter notebook notebooks/

# Check notebook execution order
ls -1 notebooks/*.ipynb

# View model metrics
python -c "import pickle; print(pickle.load(open('models/model_metrics.pkl', 'rb')))"

# Git workflow
git status
git add .
git commit -m "feat: Add new feature"
git push -u origin claude/<session-id>
```

### Performance Benchmarks

| Model | F1-Score | Training Time | Inference (1K tweets) |
|-------|----------|---------------|----------------------|
| **Linear SVM** | **85.18%** | 8.2s | 0.05s |
| Logistic Regression | 84.2% | 6.1s | 0.04s |
| Random Forest | 81.3% | 45.3s | 0.12s |
| Naive Bayes | 78.5% | 2.7s | 0.03s |
| **BERT (RoBERTa)** | **94.57%** | 2h 15m | 2.5s |

### Key Hyperparameters

```python
# TF-IDF
MAX_FEATURES_TFIDF = 5000
NGRAM_RANGE = (1, 2)  # Unigrams + bigrams
MIN_DF = 2            # Minimum document frequency
MAX_DF = 0.95         # Maximum document frequency

# Word2Vec
VECTOR_SIZE = 100     # Embedding dimensions
WINDOW = 5            # Context window
MIN_COUNT = 5         # Minimum word frequency
EPOCHS = 10           # Training epochs

# SVM (Best model)
C = 1.0               # Regularization parameter
KERNEL = 'linear'
MAX_ITER = 1000

# Data split
TRAIN_SIZE = 0.85
VAL_SIZE = 0.15
TEST_SIZE = 359       # Fixed test set
```

### Troubleshooting

| Issue | Solution |
|-------|----------|
| `FileNotFoundError: training.csv` | Download Sentiment140 dataset from official site |
| `MemoryError` during TF-IDF | Reduce `MAX_FEATURES_TFIDF` or process in batches |
| `UnicodeDecodeError` | Use `encoding='latin-1'` for Sentiment140 data |
| `ImportError: No module named 'nltk'` | Run `pip install -r requirements.txt` |
| NLTK data not found | Run `nltk.download('stopwords')` and `nltk.download('wordnet')` |
| Git push 403 error | Verify branch name starts with `claude/` and ends with session ID |
| Test failures | Ensure NLTK data downloaded, check encoding in test files |
| Notebook kernel crash | Reduce dataset size or increase RAM allocation |
| BERT inference slow | Use CPU if GPU unavailable, or reduce batch size |

---

## 📝 Additional Notes for AI Assistants

### When to Ask User for Clarification

- **Large file operations:** If asked to modify files >100MB (excluded from git)
- **Destructive operations:** Deleting models, overwriting processed data
- **Architectural changes:** Major refactoring of module structure
- **Hyperparameter changes:** If tuning will require re-training (hours)
- **Dependency updates:** Major version bumps that may break compatibility

### When to Proactively Use Tools

- **Grep/Glob:** When user asks about code locations ("where is X implemented?")
- **Read:** When discussing specific functions (always read before modifying)
- **Task (Explore agent):** When user asks "how does X work?" (codebase exploration)
- **Edit:** For targeted changes to existing code (preferred over Write)
- **Write:** Only for new files or complete rewrites

### Code Quality Checklist

Before committing code changes, verify:

- [ ] Type hints added to all function signatures
- [ ] Docstrings (Google-style) with Args/Returns/Raises
- [ ] Imports organized (stdlib → third-party → local)
- [ ] No hardcoded paths (use `config.py` constants)
- [ ] Random seeds set for reproducibility
- [ ] Tests added/updated if modifying preprocessing
- [ ] No data leakage (fit on train, transform on test)
- [ ] Unicode emoji indicators in print statements
- [ ] Error handling for edge cases (None, empty input)
- [ ] Code follows PEP 8 (use `black` formatter if available)

---

## 🔗 External Resources

- **Dataset:** [Sentiment140 Official Site](http://help.sentiment140.com/for-students)
- **Scikit-learn Docs:** https://scikit-learn.org/stable/
- **Gensim Word2Vec:** https://radimrehurek.com/gensim/models/word2vec.html
- **VADER Sentiment:** https://github.com/cjhutto/vaderSentiment
- **Transformers (BERT):** https://huggingface.co/docs/transformers
- **CRISP-DM Methodology:** https://www.datascience-pm.com/crisp-dm-2/

---

## 📅 Maintenance Schedule

| Task | Frequency | Last Done |
|------|-----------|-----------|
| Update dependencies | Quarterly | 2025-12-08 |
| Re-train models | When dataset updated | 2025-12-08 |
| Run full test suite | Before commits | 2025-12-08 |
| Update documentation | Per feature addition | 2025-12-08 |
| Audit for data leakage | Per methodology change | 2025-12-08 |
| Benchmark vs new models | Annually | 2025-12-08 |

---

**END OF CLAUDE.MD**

*This guide is maintained alongside the codebase. Update this file when making structural changes.*
