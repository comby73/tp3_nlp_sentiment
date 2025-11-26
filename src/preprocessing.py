"""
preprocessing.py
Funciones puras de limpieza y normalización de texto.

Según consigna TP3:
- Solo limpieza de texto: URLs, menciones, hashtags, puntuación, stopwords, etc.
- NO genera features numéricas (eso va en features.py).
- NO entrena modelos (eso va en models.py).
- Devuelve texto limpio listo para vectorización.
"""

import re
import string
from typing import List
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

from .config import STOP_WORDS_LANGUAGE, MIN_WORD_LENGTH

# Asegurar que los recursos de NLTK están disponibles
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)


def extract_urls(text: str) -> List[str]:
    """
    Extrae todas las URLs de un texto.
    
    Args:
        text: Texto del que extraer URLs.
    
    Returns:
        Lista de URLs encontradas.
    """
    url_pattern = r'http\S+|www\S+|https\S+'
    return re.findall(url_pattern, str(text))


def extract_mentions(text: str) -> List[str]:
    """
    Extrae todas las menciones (@usuario) de un texto.
    
    Args:
        text: Texto del que extraer menciones.
    
    Returns:
        Lista de menciones encontradas (sin el @).
    """
    mention_pattern = r'@(\w+)'
    return re.findall(mention_pattern, str(text))


def extract_hashtags(text: str) -> List[str]:
    """
    Extrae todos los hashtags (#tag) de un texto.
    
    Args:
        text: Texto del que extraer hashtags.
    
    Returns:
        Lista de hashtags encontrados (sin el #).
    """
    hashtag_pattern = r'#(\w+)'
    return re.findall(hashtag_pattern, str(text))


def detect_intensified_words(text: str) -> int:
    """
    Detecta palabras intensificadas con letras repetidas.
    Por ejemplo: 'loooove', 'haaaate', 'soooo'.
    
    Args:
        text: Texto a analizar.
    
    Returns:
        Cantidad de palabras intensificadas encontradas.
    """
    # Patrón: letra repetida 3 o más veces
    intensified_pattern = r'\b\w*([a-z])\1{2,}\w*\b'
    matches = re.findall(intensified_pattern, str(text).lower())
    return len(matches)


def calculate_uppercase_ratio(text: str) -> float:
    """
    Calcula la proporción de caracteres en mayúsculas en el texto.
    
    Args:
        text: Texto a analizar.
    
    Returns:
        Ratio de mayúsculas (0.0 a 1.0).
    """
    if not text or len(text) == 0:
        return 0.0
    
    # Solo contar letras
    letters = [c for c in str(text) if c.isalpha()]
    if len(letters) == 0:
        return 0.0
    
    uppercase_count = sum(1 for c in letters if c.isupper())
    return uppercase_count / len(letters)


def clean_text(text: str) -> str:
    """
    Limpia un texto eliminando URLs, menciones, hashtags,
    caracteres especiales y convirtiendo a minúsculas.
    
    Args:
        text: Texto a limpiar.
    
    Returns:
        Texto limpio.
    """
    if pd.isna(text):
        return ""
    
    # Convertir a string por si acaso
    text = str(text)
    
    # Convertir a minúsculas
    text = text.lower()
    
    # Eliminar URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Eliminar menciones (@usuario)
    text = re.sub(r'@\w+', '', text)
    
    # Eliminar hashtags (solo el símbolo #, mantener el texto)
    text = re.sub(r'#', '', text)
    
    # Eliminar números
    text = re.sub(r'\d+', '', text)
    
    # Eliminar puntuación
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Eliminar espacios múltiples
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def tokenize_text(text: str) -> List[str]:
    """
    Tokeniza un texto en palabras individuales.
    
    Args:
        text: Texto a tokenizar.
    
    Returns:
        Lista de tokens.
    """
    if not text:
        return []
    
    tokens = word_tokenize(text)
    return tokens


def remove_stopwords(tokens: List[str], language: str = STOP_WORDS_LANGUAGE) -> List[str]:
    """
    Elimina las stopwords de una lista de tokens.
    
    Args:
        tokens: Lista de tokens.
        language: Idioma de las stopwords.
    
    Returns:
        Lista de tokens sin stopwords.
    """
    stop_words = set(stopwords.words(language))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return filtered_tokens


def apply_stemming(tokens: List[str]) -> List[str]:
    """
    Aplica stemming (reducción a raíz) a una lista de tokens.
    
    Args:
        tokens: Lista de tokens.
    
    Returns:
        Lista de tokens con stemming aplicado.
    """
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return stemmed_tokens


def apply_lemmatization(tokens: List[str]) -> List[str]:
    """
    Aplica lematización a una lista de tokens.
    
    Args:
        tokens: Lista de tokens.
    
    Returns:
        Lista de tokens lematizados.
    """
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens


def filter_short_tokens(tokens: List[str], min_length: int = MIN_WORD_LENGTH) -> List[str]:
    """
    Filtra tokens que sean muy cortos.
    
    Args:
        tokens: Lista de tokens.
        min_length: Longitud mínima de los tokens.
    
    Returns:
        Lista de tokens filtrados.
    """
    return [token for token in tokens if len(token) >= min_length]


def preprocess_text(
    text: str, 
    remove_stops: bool = True,
    use_stemming: bool = False,
    use_lemmatization: bool = False
) -> str:
    """
    Pipeline completo de preprocesamiento de texto.
    
    Args:
        text: Texto a preprocesar.
        remove_stops: Si se eliminan stopwords.
        use_stemming: Si se aplica stemming.
        use_lemmatization: Si se aplica lematización.
    
    Returns:
        Texto preprocesado como string.
    """
    # Limpiar texto
    text = clean_text(text)
    
    # Tokenizar
    tokens = tokenize_text(text)
    
    # Eliminar stopwords
    if remove_stops:
        tokens = remove_stopwords(tokens)
    
    # Filtrar tokens cortos
    tokens = filter_short_tokens(tokens)
    
    # Aplicar stemming o lematización
    if use_stemming:
        tokens = apply_stemming(tokens)
    elif use_lemmatization:
        tokens = apply_lemmatization(tokens)
    
    # Reconstruir texto
    processed_text = ' '.join(tokens)
    
    return processed_text


def preprocess_dataframe(
    df: pd.DataFrame,
    text_column: str = 'text',
    remove_stops: bool = True,
    use_stemming: bool = False,
    use_lemmatization: bool = False
) -> pd.DataFrame:
    """
    Aplica preprocesamiento a una columna de texto en un DataFrame.
    
    Args:
        df: DataFrame con los datos.
        text_column: Nombre de la columna con el texto.
        remove_stops: Si se eliminan stopwords.
        use_stemming: Si se aplica stemming.
        use_lemmatization: Si se aplica lematización.
    
    Returns:
        DataFrame con una nueva columna 'text_processed'.
    """
    df_copy = df.copy()
    
    print("Preprocesando textos...")
    df_copy['text_processed'] = df_copy[text_column].apply(
        lambda x: preprocess_text(
            x, 
            remove_stops=remove_stops,
            use_stemming=use_stemming,
            use_lemmatization=use_lemmatization
        )
    )
    
    print("✓ Preprocesamiento completado")
    return df_copy
