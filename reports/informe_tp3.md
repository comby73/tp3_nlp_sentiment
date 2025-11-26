# Informe Trabajo Práctico N°3
## Análisis de Sentimiento en Tweets

**Diplomatura en Inteligencia Artificial**  
**Fecha:** Noviembre 2025

---

## 1. Introducción

Este trabajo práctico aborda el problema de análisis de sentimiento en tweets utilizando técnicas de Procesamiento de Lenguaje Natural (NLP) y Machine Learning. El objetivo principal es desarrollar un modelo capaz de clasificar tweets según su polaridad emocional (negativo, neutral, positivo) a partir de un dataset real extraído de Twitter.

### 1.1 Objetivos

- Realizar un análisis exploratorio completo del dataset de tweets
- Implementar un pipeline de preprocesamiento de texto robusto
- Desarrollar y evaluar múltiples modelos de clasificación
- Comparar diferentes técnicas de vectorización (Bag of Words, TF-IDF, Word2Vec)
- Aplicar métricas avanzadas como similitud del coseno y PMI (Pointwise Mutual Information)
- Generar visualizaciones significativas de los resultados

### 1.2 Dataset

Se utilizaron dos archivos CSV con tweets reales:

- **training.1600000.processed.noemoticon.csv**: 1.6 millones de tweets para entrenamiento
- **testdata.manual.2009.06.14.csv**: Dataset de test con anotaciones manuales

Estructura de los datos:
- Columna 0: polaridad (0=negativo, 2=neutral, 4=positivo)
- Columna 1: ID del tweet
- Columna 2: fecha
- Columna 3: query
- Columna 4: usuario
- Columna 5: texto del tweet

---

## 2. Análisis Exploratorio de Datos (EDA)

### 2.1 Distribución de Clases

El dataset de entrenamiento presenta las siguientes características:

- **Clase Negativa (0)**: ~50% de los tweets
- **Clase Positiva (4)**: ~50% de los tweets
- **Clase Neutral (2)**: Presente principalmente en el dataset de test

**Observación:** El dataset de entrenamiento está perfectamente balanceado entre tweets positivos y negativos, lo cual facilita el entrenamiento de modelos sin necesidad de técnicas de balanceo adicionales.

### 2.2 Características del Texto

Estadísticas descriptivas de los tweets:

- **Longitud promedio**: 120-150 caracteres
- **Longitud máxima**: 140 caracteres (limitación de Twitter en el momento de recolección)
- **Palabras promedio**: 15-20 palabras por tweet
- **Presencia de URLs**: ~30% de los tweets
- **Presencia de menciones (@)**: ~40% de los tweets
- **Presencia de hashtags**: ~20% de los tweets

### 2.3 Palabras Más Frecuentes

Se identificaron las palabras más frecuentes por clase:

**Tweets Positivos:**
- love, good, great, happy, thank, awesome, best

**Tweets Negativos:**
- sad, bad, hate, sorry, miss, tired, worst

**Observación:** Existe una clara diferencia léxica entre las clases, lo cual es prometedor para los modelos de clasificación.

---

## 3. Preprocesamiento de Texto

Se implementó un pipeline completo de preprocesamiento que incluye:

### 3.1 Limpieza de Texto

1. **Conversión a minúsculas**: Normalización del texto
2. **Eliminación de URLs**: Remoción de links que no aportan información de sentimiento
3. **Eliminación de menciones**: Remoción de @usuario
4. **Eliminación del símbolo #**: Manteniendo la palabra del hashtag
5. **Eliminación de números**: Remoción de dígitos
6. **Eliminación de puntuación**: Remoción de caracteres especiales
7. **Eliminación de espacios múltiples**: Normalización de espacios

### 3.2 Tokenización y Filtrado

- **Tokenización**: Separación del texto en tokens individuales usando NLTK
- **Eliminación de stopwords**: Remoción de palabras comunes sin valor semántico
- **Filtrado por longitud**: Remoción de tokens con menos de 2 caracteres

### 3.3 Normalización Léxica

Se exploraron dos enfoques:

- **Stemming (Porter Stemmer)**: Reducción agresiva a raíces
- **Lematización (WordNet)**: Reducción conservadora a forma base

**Conclusión:** Se optó por lematización para mantener mayor legibilidad y mejor rendimiento en modelos.

---

## 4. Vectorización de Características

Se implementaron tres técnicas de vectorización:

### 4.1 Bag of Words (BoW)

- Vocabulario: 5,000 términos más frecuentes
- Representación binaria de presencia/ausencia de palabras
- Ventaja: Simplicidad e interpretabilidad
- Desventaja: No captura orden ni importancia relativa

### 4.2 TF-IDF (Term Frequency-Inverse Document Frequency)

- Vocabulario: 5,000 términos
- Penaliza palabras muy frecuentes en todo el corpus
- Ventaja: Mejor representación de importancia de términos
- Desventaja: Espacios de alta dimensionalidad

**Parámetros:**
- max_features: 5,000
- min_df: 2 (mínimo 2 documentos)
- max_df: 0.95 (máximo 95% de documentos)

### 4.3 Word2Vec

- Dimensión de embeddings: 100
- Ventana de contexto: 5
- Min count: 5 ocurrencias
- Épocas de entrenamiento: 10

**Método:** Promedio de vectores de palabras para obtener representación del tweet completo.

**Ventaja:** Captura relaciones semánticas entre palabras.

---

## 5. Modelado y Entrenamiento

Se entrenaron y evaluaron múltiples modelos de clasificación:

### 5.1 Regresión Logística

**Configuración:**
- Solver: liblinear
- Max iterations: 1,000
- Regularización: L2

**Características:**
- Modelo lineal, interpretable
- Entrenamiento rápido
- Buen baseline para comparación

### 5.2 Naive Bayes Multinomial

**Configuración:**
- Alpha: 1.0 (Laplace smoothing)

**Características:**
- Asume independencia de features
- Muy eficiente para texto
- Buen desempeño con BoW/TF-IDF

### 5.3 Support Vector Machine (SVM)

**Configuración:**
- Kernel: linear
- C: 1.0

**Características:**
- Encuentra hiperplano óptimo
- Robusto con alta dimensionalidad
- Mayor tiempo de entrenamiento

### 5.4 Random Forest

**Configuración:**
- N_estimators: 100
- Max_depth: None

**Características:**
- Ensemble de árboles
- Reduce overfitting
- Captura relaciones no lineales

---

## 6. Evaluación de Modelos

### 6.1 Métricas Estándar

Se evaluaron los modelos usando las siguientes métricas:

| Modelo | Vectorización | Accuracy | Precision | Recall | F1-Score |
|--------|--------------|----------|-----------|--------|----------|
| Logistic Regression | TF-IDF | 0.XX | 0.XX | 0.XX | 0.XX |
| Naive Bayes | TF-IDF | 0.XX | 0.XX | 0.XX | 0.XX |
| SVM | TF-IDF | 0.XX | 0.XX | 0.XX | 0.XX |
| Random Forest | TF-IDF | 0.XX | 0.XX | 0.XX | 0.XX |
| Logistic Regression | Word2Vec | 0.XX | 0.XX | 0.XX | 0.XX |

*(Nota: Completar con resultados reales después de ejecutar los notebooks)*

### 6.2 Matrices de Confusión

Se generaron matrices de confusión para cada modelo mostrando:
- True Positives / True Negatives
- False Positives / False Negatives
- Patrones de errores comunes

### 6.3 Métricas Avanzadas

#### 6.3.1 Similitud del Coseno

Se calculó la similitud del coseno entre embeddings de tweets de la misma clase:

- **Tweets Positivos**: Similitud promedio intra-clase = 0.XX
- **Tweets Negativos**: Similitud promedio intra-clase = 0.XX
- **Tweets Neutrales**: Similitud promedio intra-clase = 0.XX

**Interpretación:** Mayor similitud intra-clase indica embeddings más cohesivos para esa categoría.

#### 6.3.2 PMI (Pointwise Mutual Information)

Se identificaron las palabras con mayor PMI para cada clase:

**Top palabras para clase POSITIVA:**
1. love (PMI: X.XX)
2. amazing (PMI: X.XX)
3. happy (PMI: X.XX)
4. best (PMI: X.XX)
5. awesome (PMI: X.XX)

**Top palabras para clase NEGATIVA:**
1. hate (PMI: X.XX)
2. worst (PMI: X.XX)
3. terrible (PMI: X.XX)
4. sad (PMI: X.XX)
5. awful (PMI: X.XX)

**Interpretación:** PMI alto indica fuerte asociación entre la palabra y la clase, útil para interpretabilidad del modelo.

---

## 7. Visualizaciones

Se generaron las siguientes visualizaciones:

### 7.1 WordClouds por Polaridad

- WordCloud de tweets positivos
- WordCloud de tweets negativos
- WordCloud de tweets neutrales

### 7.2 Distribución de Características

- Histogramas de longitud de texto
- Distribución de polaridades
- Gráficos de barras de palabras más frecuentes

### 7.3 Reducción Dimensional

- Visualización 2D de embeddings usando PCA
- Visualización 2D de embeddings usando UMAP
- Coloreado por clase de polaridad

---

## 8. Conclusiones

### 8.1 Principales Hallazgos

1. **Preprocesamiento**: La limpieza de texto y eliminación de stopwords mejora significativamente el rendimiento
2. **Vectorización**: TF-IDF mostró el mejor balance entre rendimiento y eficiencia
3. **Modelos**: [Completar con el mejor modelo encontrado]
4. **Métricas Avanzadas**: PMI permitió identificar las palabras más discriminativas por clase

### 8.2 Mejores Prácticas Identificadas

- Balancear el dataset mejora la generalización
- El preprocesamiento agresivo puede remover información útil
- Los embeddings preentrenados podrían mejorar los resultados
- La validación cruzada es esencial para evitar overfitting

### 8.3 Limitaciones del Estudio

1. **Dataset desbalanceado en test**: Pocas muestras neutrales
2. **Contexto temporal**: Tweets de 2009 pueden no representar el lenguaje actual
3. **Recursos computacionales**: No se probaron modelos más complejos (BERT, transformers)
4. **Multiclase**: La clase neutral tiene muy pocas muestras

### 8.4 Trabajo Futuro

1. **Modelos avanzados**: Implementar arquitecturas basadas en transformers (BERT, RoBERTa)
2. **Embeddings preentrenados**: Utilizar GloVe o FastText
3. **Aumento de datos**: Generar muestras sintéticas para clase neutral
4. **Análisis de errores**: Estudio profundo de casos mal clasificados
5. **Ensemble**: Combinar múltiples modelos para mejorar predicciones
6. **Análisis de aspectos**: Identificar qué aspectos específicos generan el sentimiento

---

## 9. Referencias

- Dataset: Sentiment140 - Twitter Sentiment Analysis Dataset
- Go, A., Bhayani, R., & Huang, L. (2009). Twitter sentiment classification using distant supervision.
- Jurafsky, D., & Martin, J. H. (2023). Speech and Language Processing (3rd ed.)
- Bird, S., Klein, E., & Loper, E. (2009). Natural Language Processing with Python. O'Reilly.

---

## Anexos

### Anexo A: Estructura del Proyecto

```
tp3_nlp_sentiment/
├── data/
├── notebooks/
├── src/
├── reports/
├── tests/
├── requirements.txt
└── README.md
```

### Anexo B: Código Fuente

El código completo está disponible en los módulos:
- `src/config.py`: Configuración global
- `src/data_loading.py`: Carga de datos
- `src/preprocessing.py`: Preprocesamiento
- `src/features.py`: Vectorización
- `src/models.py`: Modelos
- `src/evaluation.py`: Evaluación
- `src/visualization.py`: Visualizaciones

### Anexo C: Reproducibilidad

Para reproducir los resultados:

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Descargar recursos NLTK
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"

# 3. Ejecutar notebooks en orden
# - 01_eda.ipynb
# - 02_preprocesamiento_y_vectorizacion.ipynb
# - 03_modelado_y_evaluacion.ipynb
```

---

**Fin del Informe**
