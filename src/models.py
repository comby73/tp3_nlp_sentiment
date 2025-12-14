"""
models.py
Funciones para entrenar, guardar y cargar modelos de clasificación de sentimiento.

Según consigna TP3:
- Debe haber al menos UN clasificador de sentimiento entrenado.
- Se deben comparar al menos DOS enfoques:
  a) Un modelo entrenado con BoW/TF-IDF (ej: LogisticRegression, Naive Bayes, SVM).
  b) Otro modelo o enfoque base (otro algoritmo ML, o modelo pre-entrenado).
  
Este módulo NO debe hacer preprocesamiento ni generar features, solo entrenar modelos.
"""

import numpy as np
from typing import Any, Dict
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import joblib

from .config import RANDOM_SEED, PROCESSED_DATA_DIR
import time
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

def train_and_evaluate(model, model_name, X_train, y_train, X_test, y_test):
    """
    Entrena un modelo y retorna métricas de evaluación en ambos conjuntos (train y test).
    Evaluar en train permite detectar overfitting comparando con test.
    """
    print(f"\n{'='*60}")
    print(f"MODELO: {model_name}")
    print(f"{'='*60}")
    
    # Entrenar
    print("Entrenando...")
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    print(f"✓ Entrenado en {train_time:.2f} segundos")
    
    # Predecir en TRAIN
    print("Prediciendo en conjunto de TRAIN...")
    start_time = time.time()
    y_pred_train = model.predict(X_train)
    pred_time_train = time.time() - start_time
    print(f"✓ Predicción en train completada en {pred_time_train:.2f} segundos")
    
    # Predecir en TEST
    print("Prediciendo en conjunto de TEST...")
    start_time = time.time()
    y_pred_test = model.predict(X_test)
    pred_time_test = time.time() - start_time
    print(f"✓ Predicción en test completada en {pred_time_test:.2f} segundos")
    
    # Calcular métricas en TRAIN
    train_accuracy = accuracy_score(y_train, y_pred_train)
    train_precision = precision_score(y_train, y_pred_train, pos_label=1)
    train_recall = recall_score(y_train, y_pred_train, pos_label=1)
    train_f1 = f1_score(y_train, y_pred_train, pos_label=1)
    
    # Calcular métricas en TEST
    test_accuracy = accuracy_score(y_test, y_pred_test)
    test_precision = precision_score(y_test, y_pred_test, pos_label=1)
    test_recall = recall_score(y_test, y_pred_test, pos_label=1)
    test_f1 = f1_score(y_test, y_pred_test, pos_label=1)
    
    print("\n📊 MÉTRICAS EN TRAIN:")
    print(f"  Accuracy:  {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"  Precision: {train_precision:.4f}")
    print(f"  Recall:    {train_recall:.4f}")
    print(f"  F1-Score:  {train_f1:.4f}")
    
    print("\n📊 MÉTRICAS EN TEST:")
    print(f"  Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall:    {test_recall:.4f}")
    print(f"  F1-Score:  {test_f1:.4f}")
    
    # Análisis de overfitting
    f1_diff = train_f1 - test_f1
    print("\n🔍 ANÁLISIS DE GENERALIZACIÓN:")
    print(f"  Diferencia F1 (Train - Test): {f1_diff:+.4f}")
    if f1_diff > 0.05:
        print(f"  ⚠️ Posible overfitting: Train F1 > Test F1 en {f1_diff:.4f}")
    elif f1_diff < -0.05:
        print("  ✅ Test superior a Train (buena generalización)")
    else:
        print("  ✅ Modelo generaliza bien (diferencia < 5%)")
    
    # Retornar resultados
    # Mantenemos compatibilidad con código existente usando métricas de test
    return {
        'model': model,
        'model_name': model_name,
        'y_pred': y_pred_test,  # Compatibilidad: y_pred es de test
        'y_pred_train': y_pred_train,
        'y_pred_test': y_pred_test,
        # Métricas de TEST (para compatibilidad con comparaciones existentes)
        'accuracy': test_accuracy,
        'precision': test_precision,
        'recall': test_recall,
        'f1': test_f1,
        # Métricas de TRAIN
        'train_accuracy': train_accuracy,
        'train_precision': train_precision,
        'train_recall': train_recall,
        'train_f1': train_f1,
        # Métricas de TEST (con prefijo explícito)
        'test_accuracy': test_accuracy,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        # Tiempos
        'train_time': train_time,
        'pred_time': pred_time_test,
        'pred_time_train': pred_time_train,
        'pred_time_test': pred_time_test
    }



def train_logistic_regression(
    X_train, 
    y_train,
    max_iter: int = 1000,
    random_state: int = RANDOM_SEED
) -> LogisticRegression:
    """
    Entrena un modelo de Regresión Logística.
    
    Args:
        X_train: Features de entrenamiento.
        y_train: Etiquetas de entrenamiento.
        max_iter: Número máximo de iteraciones.
        random_state: Semilla aleatoria.
    
    Returns:
        Modelo entrenado.
    """
    print("Entrenando Regresión Logística...")
    
    model = LogisticRegression(
        max_iter=max_iter,
        random_state=random_state,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    print("✓ Regresión Logística entrenada")
    
    return model


def train_naive_bayes(X_train, y_train) -> MultinomialNB:
    """
    Entrena un modelo Naive Bayes (Multinomial).
    
    Args:
        X_train: Features de entrenamiento.
        y_train: Etiquetas de entrenamiento.
    
    Returns:
        Modelo entrenado.
    """
    print("Entrenando Naive Bayes...")
    
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    print("✓ Naive Bayes entrenado")
    
    return model


def train_random_forest(
    X_train,
    y_train,
    n_estimators: int = 100,
    max_depth: int = None,
    random_state: int = RANDOM_SEED
) -> RandomForestClassifier:
    """
    Entrena un modelo Random Forest.
    
    Args:
        X_train: Features de entrenamiento.
        y_train: Etiquetas de entrenamiento.
        n_estimators: Número de árboles.
        max_depth: Profundidad máxima de los árboles.
        random_state: Semilla aleatoria.
    
    Returns:
        Modelo entrenado.
    """
    print("Entrenando Random Forest...")
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    print("✓ Random Forest entrenado")
    
    return model


def train_svm(
    X_train,
    y_train,
    kernel: str = 'linear',
    C: float = 1.0,
    random_state: int = RANDOM_SEED
) -> SVC:
    """
    Entrena un modelo SVM (Support Vector Machine).
    
    Args:
        X_train: Features de entrenamiento.
        y_train: Etiquetas de entrenamiento.
        kernel: Tipo de kernel ('linear', 'rbf', 'poly').
        C: Parámetro de regularización.
        random_state: Semilla aleatoria.
    
    Returns:
        Modelo entrenado.
    """
    print(f"Entrenando SVM con kernel {kernel}...")
    
    model = SVC(
        kernel=kernel,
        C=C,
        random_state=random_state
    )
    
    model.fit(X_train, y_train)
    
    print("✓ SVM entrenado")
    
    return model


def save_model(model: Any, filename: str) -> None:
    """
    Guarda un modelo entrenado usando joblib.
    
    Args:
        model: Modelo a guardar.
        filename: Nombre del archivo (con extensión .joblib o .pkl).
    """
    file_path = PROCESSED_DATA_DIR / filename
    
    joblib.dump(model, file_path)
    
    print(f"✓ Modelo guardado en: {file_path}")


def load_model(filename: str) -> Any:
    """
    Carga un modelo guardado.
    
    Args:
        filename: Nombre del archivo del modelo.
    
    Returns:
        Modelo cargado.
    """
    file_path = PROCESSED_DATA_DIR / filename
    
    if not file_path.exists():
        raise FileNotFoundError(f"No se encontró el modelo: {file_path}")
    
    model = joblib.load(file_path)
    
    print(f"✓ Modelo cargado desde: {file_path}")
    
    return model


def predict(model: Any, X) -> np.ndarray:
    """
    Realiza predicciones con un modelo.
    
    Args:
        model: Modelo entrenado.
        X: Features para predicción.
    
    Returns:
        Array de predicciones.
    """
    predictions = model.predict(X)
    return predictions


def predict_proba(model: Any, X) -> np.ndarray:
    """
    Obtiene probabilidades de predicción (si el modelo lo soporta).
    
    Args:
        model: Modelo entrenado.
        X: Features para predicción.
    
    Returns:
        Array de probabilidades.
    """
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X)
        return probabilities
    else:
        raise AttributeError("El modelo no tiene método predict_proba")


def get_model_params(model: Any) -> Dict:
    """
    Obtiene los hiperparámetros de un modelo.
    
    Args:
        model: Modelo de scikit-learn.
    
    Returns:
        Diccionario con los parámetros del modelo.
    """
    return model.get_params()


def train_decision_tree(
    X_train,
    y_train,
    max_depth: int = 10,
    random_state: int = RANDOM_SEED
):
    """
    Entrena un Árbol de Decisión (útil como modelo base alternativo).
    
    Args:
        X_train: Features de entrenamiento.
        y_train: Etiquetas de entrenamiento.
        max_depth: Profundidad máxima del árbol.
        random_state: Semilla aleatoria.
    
    Returns:
        Modelo entrenado.
    """
    from sklearn.tree import DecisionTreeClassifier
    
    print("Entrenando Árbol de Decisión...")
    
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=random_state
    )
    
    model.fit(X_train, y_train)
    
    print("✓ Árbol de Decisión entrenado")
    
    return model
