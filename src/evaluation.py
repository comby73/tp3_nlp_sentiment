"""
evaluation.py
Funciones para evaluar modelos clasificadores de sentimiento.

Según consigna TP3:
- Métricas clásicas: accuracy, precision, recall, f1-score, matriz de confusión.
- Métricas especiales obligatorias (al menos UNA):
  1) Similitud del coseno entre embeddings de textos.
  2) PMI (Pointwise Mutual Information) entre palabras y clases de polaridad.
- Funciones para comparar múltiples modelos entre sí.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import math


def calculate_accuracy(y_true, y_pred) -> float:
    """
    Calcula el accuracy.
    
    Args:
        y_true: Etiquetas verdaderas.
        y_pred: Predicciones.
    
    Returns:
        Accuracy score.
    """
    return accuracy_score(y_true, y_pred)


def calculate_precision(y_true, y_pred, average: str = 'weighted') -> float:
    """
    Calcula la precision.
    
    Args:
        y_true: Etiquetas verdaderas.
        y_pred: Predicciones.
        average: Tipo de promedio ('micro', 'macro', 'weighted').
    
    Returns:
        Precision score.
    """
    return precision_score(y_true, y_pred, average=average, zero_division=0)


def calculate_recall(y_true, y_pred, average: str = 'weighted') -> float:
    """
    Calcula el recall.
    
    Args:
        y_true: Etiquetas verdaderas.
        y_pred: Predicciones.
        average: Tipo de promedio ('micro', 'macro', 'weighted').
    
    Returns:
        Recall score.
    """
    return recall_score(y_true, y_pred, average=average, zero_division=0)


def calculate_f1(y_true, y_pred, average: str = 'weighted') -> float:
    """
    Calcula el F1-score.
    
    Args:
        y_true: Etiquetas verdaderas.
        y_pred: Predicciones.
        average: Tipo de promedio ('micro', 'macro', 'weighted').
    
    Returns:
        F1 score.
    """
    return f1_score(y_true, y_pred, average=average, zero_division=0)


def get_confusion_matrix(y_true, y_pred) -> np.ndarray:
    """
    Calcula la matriz de confusión.
    
    Args:
        y_true: Etiquetas verdaderas.
        y_pred: Predicciones.
    
    Returns:
        Matriz de confusión.
    """
    return confusion_matrix(y_true, y_pred)


def get_classification_report(y_true, y_pred) -> str:
    """
    Genera un reporte de clasificación completo.
    
    Args:
        y_true: Etiquetas verdaderas.
        y_pred: Predicciones.
    
    Returns:
        String con el reporte.
    """
    return classification_report(y_true, y_pred)


def evaluate_model(y_true, y_pred) -> Dict[str, float]:
    """
    Evalúa un modelo con múltiples métricas.
    
    Args:
        y_true: Etiquetas verdaderas.
        y_pred: Predicciones.
    
    Returns:
        Diccionario con las métricas.
    """
    metrics = {
        'accuracy': calculate_accuracy(y_true, y_pred),
        'precision': calculate_precision(y_true, y_pred),
        'recall': calculate_recall(y_true, y_pred),
        'f1_score': calculate_f1(y_true, y_pred)
    }
    
    return metrics


def print_evaluation_metrics(y_true, y_pred, model_name: str = "Modelo") -> None:
    """
    Imprime las métricas de evaluación de forma legible.
    
    Args:
        y_true: Etiquetas verdaderas.
        y_pred: Predicciones.
        model_name: Nombre del modelo para el título.
    """
    metrics = evaluate_model(y_true, y_pred)
    
    print(f"\n{'='*50}")
    print(f"Evaluación: {model_name}")
    print(f"{'='*50}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print(f"{'='*50}\n")


def calculate_cosine_similarity(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """
    Calcula la similitud del coseno entre dos vectores.
    
    Args:
        vector1: Primer vector.
        vector2: Segundo vector.
    
    Returns:
        Similitud del coseno (valor entre -1 y 1).
    """
    # Asegurar que son 2D para sklearn
    v1 = vector1.reshape(1, -1)
    v2 = vector2.reshape(1, -1)
    
    similarity = cosine_similarity(v1, v2)[0][0]
    
    return similarity


def calculate_pairwise_cosine_similarity(vectors: np.ndarray) -> np.ndarray:
    """
    Calcula la matriz de similitud del coseno entre múltiples vectores.
    
    Args:
        vectors: Matriz de vectores (n_samples, n_features).
    
    Returns:
        Matriz de similitud (n_samples, n_samples).
    """
    return cosine_similarity(vectors)


def calculate_pmi(
    df: pd.DataFrame,
    text_column: str,
    label_column: str,
    min_freq: int = 5
) -> pd.DataFrame:
    """
    Calcula el PMI (Pointwise Mutual Information) entre palabras y clases.
    
    PMI(palabra, clase) = log2(P(palabra, clase) / (P(palabra) * P(clase)))
    
    Un PMI alto indica que la palabra es muy informativa para esa clase.
    
    Args:
        df: DataFrame con textos y etiquetas.
        text_column: Nombre de la columna con el texto procesado.
        label_column: Nombre de la columna con las etiquetas.
        min_freq: Frecuencia mínima de palabras para considerar.
    
    Returns:
        DataFrame con palabras, clases y sus valores de PMI.
    """
    # Contar documentos totales
    n_total = len(df)
    
    # Contar documentos por clase
    class_counts = df[label_column].value_counts().to_dict()
    
    # Crear diccionario de frecuencias de palabras por clase
    word_class_freq = {}
    word_total_freq = Counter()
    
    for label in df[label_column].unique():
        # Textos de esta clase
        texts = df[df[label_column] == label][text_column]
        
        # Contar palabras
        for text in texts:
            words = text.split()
            for word in words:
                word_total_freq[word] += 1
                
                key = (word, label)
                if key not in word_class_freq:
                    word_class_freq[key] = 0
                word_class_freq[key] += 1
    
    # Filtrar palabras por frecuencia mínima
    word_total_freq = {w: c for w, c in word_total_freq.items() if c >= min_freq}
    
    # Calcular PMI
    pmi_results = []
    
    for (word, label), freq in word_class_freq.items():
        if word not in word_total_freq:
            continue
        
        # P(palabra, clase)
        p_word_class = freq / n_total
        
        # P(palabra)
        p_word = word_total_freq[word] / n_total
        
        # P(clase)
        p_class = class_counts[label] / n_total
        
        # PMI
        pmi = math.log2(p_word_class / (p_word * p_class)) if (p_word * p_class) > 0 else 0
        
        pmi_results.append({
            'word': word,
            'class': label,
            'pmi': pmi,
            'frequency': freq
        })
    
    # Crear DataFrame
    pmi_df = pd.DataFrame(pmi_results)
    pmi_df = pmi_df.sort_values('pmi', ascending=False)
    
    return pmi_df.reset_index(drop=True)


def get_top_pmi_words_by_class(
    pmi_df: pd.DataFrame,
    n_top: int = 20
) -> Dict[int, pd.DataFrame]:
    """
    Obtiene las palabras con mayor PMI para cada clase.
    
    Args:
        pmi_df: DataFrame con resultados de PMI.
        n_top: Número de palabras top por clase.
    
    Returns:
        Diccionario con clase -> DataFrame de palabras top.
    """
    results = {}
    
    for class_label in pmi_df['class'].unique():
        class_pmi = pmi_df[pmi_df['class'] == class_label].head(n_top)
        results[class_label] = class_pmi
    
    return results


def calculate_average_similarity_by_class(
    embeddings: np.ndarray,
    labels: np.ndarray
) -> Dict[int, float]:
    """
    MÉTRICA ESPECIAL OBLIGATORIA (opción 1): Similitud del coseno.
    
    Calcula la similitud promedio intra-clase usando embeddings de Word2Vec o similares.
    Útil para entender qué tan cohesivos son los tweets de cada clase de sentimiento.
    
    Args:
        embeddings: Matriz de embeddings (n_samples, n_features).
        labels: Etiquetas de clase (polaridad).
    
    Returns:
        Diccionario con clase -> similitud del coseno promedio intra-clase.
    """
    results = {}
    
    for label in np.unique(labels):
        # Embeddings de esta clase
        class_embeddings = embeddings[labels == label]
        
        if len(class_embeddings) < 2:
            results[label] = 0.0
            continue
        
        # Calcular similitud intra-clase
        sim_matrix = cosine_similarity(class_embeddings)
        
        # Tomar triángulo superior (sin diagonal)
        mask = np.triu(np.ones_like(sim_matrix), k=1).astype(bool)
        similarities = sim_matrix[mask]
        
        avg_similarity = np.mean(similarities)
        results[label] = avg_similarity
    
    return results


def compare_models(
    models_results: Dict[str, Dict[str, float]]
) -> pd.DataFrame:
    """
    Compara múltiples modelos según sus métricas de evaluación.
    Según consigna TP3: debe compararse al menos 2 enfoques diferentes.
    
    Args:
        models_results: Diccionario con {nombre_modelo: {metrica: valor}}.
        
    Returns:
        DataFrame ordenado por F1-score descendente con comparación de modelos.
    """
    df_comparison = pd.DataFrame(models_results).T
    
    # Ordenar por F1-score (o accuracy si F1 no está)
    sort_column = 'f1_score' if 'f1_score' in df_comparison.columns else 'accuracy'
    df_comparison = df_comparison.sort_values(sort_column, ascending=False)
    
    return df_comparison


def get_model_comparison_summary(
    models_results: Dict[str, Dict[str, float]]
) -> str:
    """
    Genera un resumen textual comparando modelos.
    
    Args:
        models_results: Diccionario con {nombre_modelo: {metrica: valor}}.
        
    Returns:
        String con resumen de comparación.
    """
    df = compare_models(models_results)
    
    best_model = df.index[0]
    best_f1 = df.iloc[0]['f1_score'] if 'f1_score' in df.columns else df.iloc[0]['accuracy']
    
    summary = f"""
    {'='*80}
    COMPARACIÓN DE MODELOS - RESUMEN
    {'='*80}
    
    Mejor modelo: {best_model}
    F1-Score: {best_f1:.4f}
    
    Ranking completo:
    {df.to_string()}
    {'='*80}
    """
    
    return summary
