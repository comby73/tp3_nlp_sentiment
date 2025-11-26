"""
Pipeline de Predicción de Sentimiento en Producción
====================================================

Este script permite predecir el sentimiento de tweets nuevos usando
el modelo entrenado. Incluye todo el preprocesamiento necesario.

Uso:
    python predict_sentiment.py "I love this product!"
    python predict_sentiment.py --file tweets.txt
"""

import pandas as pd
import numpy as np
import pickle
import re
import argparse
from pathlib import Path
import sys

# Agregar directorio actual al path para poder importar src si se ejecuta desde root
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from src.preprocessing import (
    clean_text,
    extract_hashtags,
    extract_mentions,
    extract_urls,
    calculate_uppercase_ratio
)

# ============================================================================
# 1. FUNCIONES DE PREPROCESAMIENTO (mismas del notebook 02)
# ============================================================================

def preprocess_tweet(text):
    """
    Preprocesa un tweet usando la función centralizada de src.
    """
    return clean_text(text)


def extract_features(text):
    """
    Extrae 7 features numéricas del texto original usando helpers de src.
    """
    if pd.isna(text) or not text:
        return {
            'length': 0,
            'num_words': 0,
            'num_hashtags': 0,
            'num_mentions': 0,
            'num_urls': 0,
            'num_uppercase': 0,
            'pct_uppercase': 0.0
        }
    
    return {
        'length': len(text),
        'num_words': len(text.split()),
        'num_hashtags': len(extract_hashtags(text)),
        'num_mentions': len(extract_mentions(text)),
        'num_urls': len(extract_urls(text)),
        'num_uppercase': sum(1 for c in text if c.isupper()),
        'pct_uppercase': calculate_uppercase_ratio(text) * 100
    }


# ============================================================================
# 2. CLASE PREDICTORA (encapsula todo el pipeline)
# ============================================================================

class SentimentPredictor:
    """
    Predictor de sentimiento en tweets.
    Carga el modelo y vectorizador entrenados.
    """
    
    def __init__(self, model_dir='models', data_dir='data/vectorized'):
        """
        Inicializa el predictor cargando modelo y vectorizador.
        """
        # Resolver rutas relativas a la ubicación de este script
        base_dir = Path(__file__).resolve().parent
        self.model_dir = base_dir / model_dir
        self.data_dir = base_dir / data_dir
        
        # Cargar métricas para saber cuál es el mejor modelo
        metrics_path = self.model_dir / 'model_metrics.pkl'
        with open(metrics_path, 'rb') as f:
            metrics = pickle.load(f)
        
        self.model_name = metrics['best_model_name']
        
        # Cargar modelo
        model_filename = f"best_model_{self.model_name.replace(' ', '_').lower()}.pkl"
        model_path = self.model_dir / model_filename
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # Cargar vectorizador TF-IDF
        vectorizer_path = self.data_dir / 'tfidf_vectorizer.pkl'
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        print(f"✓ Modelo cargado: {self.model_name}")
        print(f"✓ Vectorizador cargado")
    
    def preprocess_single(self, text):
        """
        Preprocesa un solo tweet y retorna texto limpio + features.
        """
        # Extraer features numéricas DEL TEXTO ORIGINAL
        features_dict = extract_features(text)
        
        # Limpiar texto
        text_clean = preprocess_tweet(text)
        
        return text_clean, features_dict
    
    def predict_single(self, text):
        """
        Predice el sentimiento de un solo tweet.
        
        Returns:
            dict: {
                'text': texto original,
                'text_clean': texto limpio,
                'sentiment': 'Positivo' o 'Negativo',
                'confidence': probabilidad (si disponible),
                'features': features numéricas extraídas
            }
        """
        # Preprocesar
        text_clean, features_dict = self.preprocess_single(text)
        
        if not text_clean:
            return {
                'text': text,
                'text_clean': '',
                'sentiment': 'Neutral',
                'confidence': None,
                'features': features_dict,
                'error': 'Texto vacío después de limpieza'
            }
        
        # Vectorizar texto
        X_text = self.vectorizer.transform([text_clean])
        
        # Agregar features numéricas
        from scipy.sparse import hstack, csr_matrix
        features_array = np.array([[
            features_dict['length'],
            features_dict['num_words'],
            features_dict['num_hashtags'],
            features_dict['num_mentions'],
            features_dict['num_urls'],
            features_dict['num_uppercase'],
            features_dict['pct_uppercase']
        ]])
        
        # Combinar
        X_final = hstack([X_text, csr_matrix(features_array)])
        
        # Predecir
        prediction = self.model.predict(X_final)[0]
        sentiment = 'Positivo' if prediction == 1 else 'Negativo'
        
        # Probabilidad (si el modelo la soporta)
        confidence = None
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(X_final)[0]
            confidence = float(proba[prediction])
        elif hasattr(self.model, 'decision_function'):
            score = self.model.decision_function(X_final)[0]
            # Convertir a probabilidad aproximada
            confidence = float(1 / (1 + np.exp(-score)))
        
        return {
            'text': text,
            'text_clean': text_clean,
            'sentiment': sentiment,
            'confidence': confidence,
            'features': features_dict
        }
    
    def predict_batch(self, texts):
        """
        Predice el sentimiento de múltiples tweets.
        
        Args:
            texts: lista de strings
        
        Returns:
            list de dicts con resultados
        """
        results = []
        for text in texts:
            result = self.predict_single(text)
            results.append(result)
        return results


# ============================================================================
# 3. FUNCIÓN PRINCIPAL (CLI)
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Predictor de Sentimiento en Tweets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python predict_sentiment.py "I love this product!"
  python predict_sentiment.py --file tweets.txt
  python predict_sentiment.py --text "Great day!" --verbose
        """
    )
    
    parser.add_argument(
        'text',
        nargs='?',
        help='Tweet a clasificar (entre comillas si tiene espacios)'
    )
    
    parser.add_argument(
        '--file', '-f',
        help='Archivo con tweets (uno por línea)'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Archivo de salida CSV (opcional)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Mostrar información detallada'
    )
    
    args = parser.parse_args()
    
    # Validar argumentos
    if not args.text and not args.file:
        parser.error("Debe proporcionar --text o --file")
    
    # Cargar predictor
    print("Cargando modelo...")
    predictor = SentimentPredictor()
    print()
    
    # Predecir
    results = []
    
    if args.text:
        # Predicción única
        result = predictor.predict_single(args.text)
        results = [result]
        
    elif args.file:
        # Predicción batch desde archivo
        with open(args.file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        print(f"Procesando {len(texts)} tweets...")
        results = predictor.predict_batch(texts)
    
    # Mostrar resultados
    print("="*80)
    print("RESULTADOS")
    print("="*80)
    
    for i, result in enumerate(results, 1):
        if 'error' in result:
            print(f"\n[{i}] ⚠️ ERROR: {result['error']}")
            print(f"    Texto: {result['text'][:80]}...")
            continue
        
        emoji = "😊" if result['sentiment'] == 'Positivo' else "😞"
        print(f"\n[{i}] {emoji} {result['sentiment']}", end='')
        
        if result['confidence']:
            print(f" (confianza: {result['confidence']:.2%})", end='')
        
        print()
        print(f"    Original: {result['text'][:80]}...")
        
        if args.verbose:
            print(f"    Limpio:   {result['text_clean'][:80]}...")
            print(f"    Features: words={result['features']['num_words']}, "
                  f"hashtags={result['features']['num_hashtags']}, "
                  f"uppercase={result['features']['num_uppercase']}")
    
    # Guardar a CSV si se especificó
    if args.output:
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False)
        print(f"\n✓ Resultados guardados en: {args.output}")
    
    # Resumen
    if len(results) > 1:
        sentiments = [r['sentiment'] for r in results if 'error' not in r]
        print(f"\n{'='*80}")
        print(f"RESUMEN: {len(sentiments)} tweets clasificados")
        print(f"  Positivos: {sentiments.count('Positivo')} ({sentiments.count('Positivo')/len(sentiments)*100:.1f}%)")
        print(f"  Negativos: {sentiments.count('Negativo')} ({sentiments.count('Negativo')/len(sentiments)*100:.1f}%)")


# ============================================================================
# 4. EJEMPLOS DE USO PROGRAMÁTICO
# ============================================================================

def example_usage():
    """
    Ejemplos de cómo usar el predictor desde código Python.
    """
    # Crear predictor
    predictor = SentimentPredictor()
    
    # Ejemplo 1: Tweet único
    result = predictor.predict_single("I love this amazing product!")
    print(f"Sentimiento: {result['sentiment']}")
    print(f"Confianza: {result['confidence']:.2%}")
    
    # Ejemplo 2: Batch de tweets
    tweets = [
        "This is the worst experience ever!",
        "Amazing! So happy with this!",
        "Meh, it's okay I guess"
    ]
    
    results = predictor.predict_batch(tweets)
    for tweet, result in zip(tweets, results):
        print(f"{tweet} → {result['sentiment']}")


if __name__ == '__main__':
    main()
