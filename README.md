# 🐦 Análisis de Sentimientos en Twitter (Sentiment140)

**Autor:** Omar Alejandro González  
**Diplomatura en Inteligencia Artificial - Universidad de Palermo**  
**Trabajo Práctico 3 - NLP**

---

## 📌 Descripción del Proyecto

Este proyecto implementa un pipeline completo de **Procesamiento de Lenguaje Natural (NLP)** para clasificar el sentimiento de tweets como **Positivo** o **Negativo**, utilizando el dataset **Sentiment140** (1.6 millones de tweets).

El desarrollo sigue la metodología **CRISP-DM** y está estructurado en notebooks modulares que cubren desde el análisis exploratorio hasta la comparación con modelos pre-entrenados de la industria.

---

## 📊 Dataset

| Característica | Valor |
|----------------|-------|
| **Nombre** | Sentiment140 |
| **Tamaño** | 1,600,000 tweets |
| **Clases** | Binario (0=Negativo, 4=Positivo) |
| **Balance** | 50% / 50% (perfectamente balanceado) |
| **Periodo** | Abril - Junio 2009 |
| **Fuente** | [Sentiment140](http://help.sentiment140.com/for-students) |

> **Nota sobre neutrales:** El dataset de entrenamiento NO contiene tweets neutrales. Los 139 tweets neutrales del conjunto de test fueron excluidos para mantener coherencia metodológica.

---

## 🏆 Resultados Destacados

### Modelo Final: Linear SVM

| Métrica | Valor |
|---------|-------|
| **F1-Score** | **85.18%** |
| **Accuracy** | 84.68% |
| **Precision** | 85.07% |

| Modelo | Accuracy | Velocidad | Tipo |
|--------|----------|-----------|------|
| **Nuestro SVM** | 84.68% | ⚡ Muy rápida | Entrenado específicamente |
| TextBlob | ~65% | ⚡ Rápida | Basado en reglas |
| VADER | ~71% | ⚡ Rápida | Optimizado para redes sociales |
| BERT (RoBERTa) | **94.57%** | 🐢 Lenta (50x) | Transformer pre-entrenado |

> **Decisión:** Se eligió Linear SVM porque es **49x más rápido** que BERT con performance competitiva, ideal para entornos productivos con recursos estándar.

---

## 📂 Estructura del Proyecto
```
tp3_nlp_sentiment/
├── 📁 data/
│   ├── predictions/            # Predicciones generadas
│   ├── processed/              # Datos limpios (CSV)
│   ├── raw/                    # Datasets originales
│   └── vectorized/             # Matrices TF-IDF (.pkl)
│
├── 📁 models/
│   ├── best_model_linear_svm.pkl
│   ├── model_metrics.pkl
│   └── word2vec_model.pkl
│
├── 📁 notebooks/               # 11 notebooks (01-11)
│
├── 📁 reports/
│   ├── figuras/                # Visualizaciones generadas
│   ├── eda_summary.json
│   └── informe_tp3.md
│
├── 📁 src/                     # Módulos Python reutilizables
│   ├── config.py
│   ├── data_loading.py
│   ├── evaluation.py
│   ├── features.py
│   ├── models.py
│   ├── preprocessing.py
│   └── visualization.py
│
├── 📁 tests/                   # Tests unitarios
├── predict_sentiment.py        # Script de predicción standalone
├── requirements.txt            # Dependencias
└── README.md
```
---

## 🔬 Metodología (CRISP-DM)

### 1. Comprensión de los Datos (`01_eda.ipynb`)
- Análisis de distribución de clases (50/50 balanceado)
- Estadísticas de longitud de tweets
- WordClouds por polaridad
- Identificación de elementos: URLs (5.1%), Mentions (46.2%), Hashtags (2.2%)

### 2. Preparación de Datos (`02_preprocessing.ipynb`)
- Eliminación de URLs, mentions
- Conversión de hashtags a texto (#fail → fail)
- Normalización de caracteres repetidos (goooood → good)
- Extracción de 8 features numéricas
- Verificación de data leakage

### 3. Feature Engineering (`03_vectorizacion.ipynb`)
- **TF-IDF** con 10,000 features
- **Bigramas** (ngram_range=(1,2)) para capturar negaciones
- Stopwords personalizadas (conserva "not", "no", "very")
- Matriz sparse eficiente (99.89% sparsity)

### 4. Modelado (`04_modelado.ipynb`)
- Split: Train (85%) / Validación (15%) / Test (359 tweets)
- Modelos evaluados: LogReg, SVM, NaiveBayes, RandomForest
- Selección por F1-Score
- Re-entrenamiento del mejor modelo con datos completos (Train + Val)
- Verificación de Overfitting: Diferencia mínima entre métricas de Train y Test

### 5. Evaluación (`05-09`)
- Optimización de hiperparámetros (GridSearchCV)
- Análisis de errores y confidence scores
- Auditoría de sesgos por longitud y metadatos
- Comparación con TextBlob, VADER, BERT

---

## 🎮 Demos Interactivas (Word2Vec)

El proyecto incluye dos juegos que demuestran las capacidades de **Word2Vec** entrenado en los 1.6M tweets:

### Sopa de Letras Semántica (`10_sopa_letras.ipynb`)
- Encuentra palabras relacionadas semánticamente
- Puntuación basada en similitud coseno
- Interfaz bilingüe (inglés/español)

### Word2Vec Tetris (`11_word2vec_tetris.ipynb`)
- Forma palabras en cualquier dirección
- Detección horizontal, vertical y diagonal
- Animaciones de explosión al formar palabras

> Estos juegos demuestran cómo Word2Vec captura relaciones semánticas: palabras como "happy", "love", "great" aparecen cercanas en el espacio vectorial.

---

## 🚀 Instalación y Uso

### Requisitos

```bash
# Dependencias principales
pip install pandas numpy scikit-learn matplotlib seaborn joblib scipy

# Para comparación con pre-entrenados
pip install textblob vaderSentiment transformers torch

# Para juegos Word2Vec
pip install gensim
```

### Ejecución

1. **Clonar/descargar** el proyecto
2. **Descargar** el dataset Sentiment140 y colocarlo en `data/raw/`
3. **Ejecutar notebooks** en orden numérico (01 → 11)

```bash
# Para reproducir desde cero:
jupyter notebook notebooks/01_eda.ipynb
```

> **Atajo:** Si solo quieres usar el modelo, los archivos `.pkl` en `models/` permiten saltar directamente al notebook `08_prediccion.ipynb`.

---

## 📈 Hallazgos Clave

### Del Análisis de Errores
- **Confianza promedio en aciertos:** 0.629
- **Confianza promedio en errores:** 0.300
- Los errores tienden a tener baja confianza (comportamiento esperado)

### Del Análisis de Sesgos
- Tweets muy cortos (<50 chars) tienen menor rendimiento
- Tweets con solo URLs o múltiples mentions son propensos a errores
- El modelo es robusto para tweets de longitud típica (50-140 chars)

### Del Análisis Temporal (Nuevo)
- **Patrón Nocturno:** Se observa una mayor concentración de tweets negativos en horas de la madrugada (00:00 - 06:00).
- **Validación de Hipótesis:** Confirma la intuición de que el horario influye en el sentimiento (usuarios más críticos/negativos de noche).

### Patrones Difíciles
- **Negaciones complejas:** "how can you not love..." (positivo pero tiene "not")
- **Sarcasmo/ironía:** Requeriría contexto adicional
- **Jerga de 2009:** Algunas expresiones han cambiado de significado

### Mejoras Futuras Identificadas
Aunque el EDA reveló patrones temporales (madrugada más negativa), **no se incluyó la hora como feature en el modelo final** por las siguientes razones:
1.  **Prioridad del Texto:** El contenido semántico es el predictor dominante (>95% de la señal).
2.  **Complejidad vs Beneficio:** Incorporar la hora requiere *codificación cíclica* (Seno/Coseno) para evitar distorsiones numéricas (23 vs 0), lo cual aumentaría la complejidad del pipeline para una ganancia marginal estimada.
3.  **Estrategia:** Se deja planteado como la principal vía de optimización para una futura iteración "v2.0" del modelo.

---

## 🛠️ Tecnologías

| Categoría | Herramientas |
|-----------|--------------|
| **Lenguaje** | Python 3.8+ |
| **ML/NLP** | Scikit-Learn, Gensim |
| **Datos** | Pandas, NumPy, SciPy |
| **Visualización** | Matplotlib, Seaborn, Plotly |
| **Pre-entrenados** | TextBlob, VADER, Transformers (BERT) |
| **Persistencia** | Joblib, Pickle |

---

## 📚 Referencias

- [Sentiment140 Dataset](http://help.sentiment140.com/for-students)
- [Scikit-Learn Documentation](https://scikit-learn.org/)
- [VADER Sentiment](https://github.com/cjhutto/vaderSentiment)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)

---

## 📝 Licencia

Proyecto académico - Universidad de Palermo
---

## ⚠️ Archivos No Incluidos (por tamaño)

Los siguientes archivos superan el límite de GitHub (100MB) y deben descargarse o regenerarse:

| Archivo | Tamaño | Cómo obtenerlo |
|---------|--------|----------------|
| `data/raw/training.1600000.processed.noemoticon.csv` | ~250 MB | [Descargar de Sentiment140](http://help.sentiment140.com/for-students) |
| `data/processed/train_processed.csv` | ~267 MB | Ejecutar `02_preprocessing.ipynb` |
| `data/vectorized/X_train.pkl` | ~210 MB | Ejecutar `03_vectorizacion.ipynb` |

### Pasos para regenerar:
```bash
# 1. Descargar dataset y colocar en data/raw/
# 2. Ejecutar notebooks en orden:
jupyter notebook notebooks/01_eda.ipynb
jupyter notebook notebooks/02_preprocessing.ipynb
jupyter notebook notebooks/03_vectorizacion.ipynb
```
```
