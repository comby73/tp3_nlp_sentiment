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
    Entrena un modelo y retorna métricas de evaluación.
    """
    print(f"\\n{'='*60}")
    print(f"MODELO: {model_name}")
    print(f"{'='*60}")
    
    # Entrenar
    print("Entrenando...")
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    print(f"✓ Entrenado en {train_time:.2f} segundos")
    
    # Predecir
    print("Prediciendo...")
    start_time = time.time()
    y_pred = model.predict(X_test)
    pred_time = time.time() - start_time
    print(f"✓ Predicción completada en {pred_time:.2f} segundos")
    
    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1)
    recall = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    
    print(f"\\n📊 MÉTRICAS:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    # Retornar resultados
    return {
        'model': model,
        'model_name': model_name,
        'y_pred': y_pred,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'train_time': train_time,
        'pred_time': pred_time
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
