"""
features.py
Funciones para generar representaciones vectoriales numéricas a partir de texto limpio.

Según consigna TP3:
- Recibe texto YA PREPROCESADO (de preprocessing.py).
- Genera matrices numéricas: Bag of Words, TF-IDF, Word2Vec.
- NO debe hacer limpieza de texto aquí (eso va en preprocessing.py).
- Preparado para agregar luego: topic modeling (LDA), análisis de analogías con embeddings.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

from .config import (
    MAX_FEATURES_TFIDF, 
    MIN_DF, 
    MAX_DF,
    W2V_VECTOR_SIZE,
    W2V_WINDOW,
    W2V_MIN_COUNT,
    W2V_WORKERS,
    W2V_EPOCHS
)


def create_bow_features(
    train_texts: List[str], 
    test_texts: Optional[List[str]] = None,
    max_features: int = MAX_FEATURES_TFIDF
) -> Tuple:
    """
    Crea representaciones Bag of Words para textos de entrenamiento y test.
    
    Args:
        train_texts: Lista de textos de entrenamiento.
        test_texts: Lista de textos de test (opcional).
        max_features: Número máximo de features.
    
    Returns:
        Si test_texts es None: (X_train, vectorizer)
        Si test_texts no es None: (X_train, X_test, vectorizer)
    """
    vectorizer = CountVectorizer(max_features=max_features)
    
    X_train = vectorizer.fit_transform(train_texts)
    
    # print(f"✓ Bag of Words creado: {X_train.shape[1]} features")
    
    if test_texts is not None:
        X_test = vectorizer.transform(test_texts)
        return X_train, X_test, vectorizer
    
    return X_train, vectorizer


def create_tfidf_features(
    train_texts: List[str],
    test_texts: Optional[List[str]] = None,
    max_features: int = MAX_FEATURES_TFIDF,
    min_df: int = MIN_DF,
    max_df: float = MAX_DF,
    ngram_range: Tuple[int, int] = (1, 1),
    stop_words: Optional[List[str]] = None
) -> Tuple:
    """
    Crea representaciones TF-IDF para textos de entrenamiento y test.
    
    Args:
        train_texts: Lista de textos de entrenamiento.
        test_texts: Lista de textos de test (opcional).
        max_features: Número máximo de features.
        min_df: Frecuencia mínima de documentos.
        max_df: Frecuencia máxima de documentos (proporción).
        ngram_range: Rango de n-gramas (min, max).
        stop_words: Lista de stopwords personalizada.
    
    Returns:
        Si test_texts es None: (X_train, vectorizer)
        Si test_texts no es None: (X_train, X_test, vectorizer)
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range,
        stop_words=stop_words
    )
    
    X_train = vectorizer.fit_transform(train_texts)
    
    # print(f"✓ TF-IDF creado: {X_train.shape[1]} features")
    
    if test_texts is not None:
        X_test = vectorizer.transform(test_texts)
        return X_train, X_test, vectorizer
    
    return X_train, vectorizer


def train_word2vec_model(
    texts: List[str],
    vector_size: int = W2V_VECTOR_SIZE,
    window: int = W2V_WINDOW,
    min_count: int = W2V_MIN_COUNT,
    workers: int = W2V_WORKERS,
    epochs: int = W2V_EPOCHS
) -> Word2Vec:
    """
    Entrena un modelo Word2Vec sobre una lista de textos.
    
    Args:
        texts: Lista de textos (ya preprocesados).
        vector_size: Dimensión de los vectores.
        window: Ventana de contexto.
        min_count: Frecuencia mínima de palabras.
        workers: Número de workers para entrenamiento paralelo.
        epochs: Número de épocas de entrenamiento.
    
    Returns:
        Modelo Word2Vec entrenado.
    """
    # Tokenizar textos
    sentences = [word_tokenize(text) for text in texts]
    
    # Entrenar modelo
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=epochs
    )
    
    print(f"✓ Modelo Word2Vec entrenado: {len(model.wv)} palabras en vocabulario")
    
    return model


def get_word2vec_embedding(text: str, model: Word2Vec) -> np.ndarray:
    """
    Obtiene el embedding promedio de un texto usando Word2Vec.
    
    Args:
        text: Texto a vectorizar.
        model: Modelo Word2Vec entrenado.
    
    Returns:
        Vector promedio del texto.
    """
    tokens = word_tokenize(text)
    
    # Obtener vectores de palabras que están en el vocabulario
    word_vectors = [model.wv[word] for word in tokens if word in model.wv]
    
    if len(word_vectors) == 0:
        # Si no hay palabras en el vocabulario, devolver vector de ceros
        return np.zeros(model.vector_size)
    
    # Promedio de vectores
    embedding = np.mean(word_vectors, axis=0)
    
    return embedding


def create_word2vec_features(
    texts: List[str],
    model: Word2Vec
) -> np.ndarray:
    """
    Crea representaciones Word2Vec (embeddings promedio) para una lista de textos.
    
    Args:
        texts: Lista de textos.
        model: Modelo Word2Vec entrenado.
    
    Returns:
        Matriz de embeddings (n_samples, vector_size).
    """
    embeddings = np.array([get_word2vec_embedding(text, model) for text in texts])
    
    print(f"✓ Embeddings Word2Vec creados: {embeddings.shape}")
    
    return embeddings


def get_top_tfidf_words(
    vectorizer: TfidfVectorizer,
    X_tfidf,
    n_top: int = 20
) -> pd.DataFrame:
    """
    Obtiene las palabras con mayor peso TF-IDF promedio.
    
    Args:
        vectorizer: Vectorizador TF-IDF ajustado.
        X_tfidf: Matriz TF-IDF.
        n_top: Número de palabras top a retornar.
    
    Returns:
        DataFrame con palabras y sus pesos promedio.
    """
    # Calcular peso promedio por término
    mean_tfidf = np.asarray(X_tfidf.mean(axis=0)).flatten()
    
    # Obtener nombres de features
    feature_names = vectorizer.get_feature_names_out()
    
    # Crear DataFrame
    df = pd.DataFrame({
        'word': feature_names,
        'tfidf_score': mean_tfidf
    })
    
    # Ordenar por score descendente
    df = df.sort_values('tfidf_score', ascending=False).head(n_top)
    
    return df.reset_index(drop=True)


def get_vocabulary_size(vectorizer) -> int:
    """
    Obtiene el tamaño del vocabulario de un vectorizador.
    
    Args:
        vectorizer: CountVectorizer o TfidfVectorizer ajustado.
    
    Returns:
        Tamaño del vocabulario.
    """
    return len(vectorizer.vocabulary_)


def analyze_word_analogies(model: Word2Vec, word1: str, word2: str, word3: str) -> List[Tuple[str, float]]:
    """
    PREPARADO para futuro: Analiza analogías entre palabras usando Word2Vec.
    Ejemplo: "good" es a "better" como "bad" es a "?"
    
    Según consigna TP3: preparado para agregar análisis de analogías con embeddings.
    
    Args:
        model: Modelo Word2Vec entrenado.
        word1: Primera palabra de la analogía.
        word2: Segunda palabra de la analogía.
        word3: Tercera palabra de la analogía.
    
    Returns:
        Lista de tuplas (palabra, similitud) con las palabras más similares.
    """
    try:
        # word1 : word2 :: word3 : ?
        results = model.wv.most_similar(positive=[word2, word3], negative=[word1], topn=5)
        return results
    except KeyError as e:
        print(f"Error: Palabra no encontrada en el vocabulario - {e}")
        return []


def get_similar_words(model: Word2Vec, word: str, top_n: int = 10) -> List[Tuple[str, float]]:
    """
    Encuentra las palabras más similares a una palabra dada usando Word2Vec.
    
    Args:
        model: Modelo Word2Vec entrenado.
        word: Palabra de consulta.
        top_n: Número de palabras similares a retornar.
    
    Returns:
        Lista de tuplas (palabra, similitud).
    """
    try:
        similar = model.wv.most_similar(word, topn=top_n)
        return similar
    except KeyError:
        print(f"Palabra '{word}' no encontrada en el vocabulario")
        return []
