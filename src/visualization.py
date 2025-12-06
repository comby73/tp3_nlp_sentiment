"""
visualization.py
Funciones para generar gráficos y visualizaciones.

Según consigna TP3:
- Gráficos de distribución de polaridad, longitud de tweets, wordclouds.
- Matrices de confusión de modelos.
- Visualizaciones de embeddings (PCA, UMAP).
- Preparado para: análisis de hashtags, menciones, usuarios.
- TODAS las figuras se guardan en reports/figuras/.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from typing import Optional, List, Dict
from pathlib import Path
import plotly.express as px
from sklearn.decomposition import PCA
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

from src.config import POLARITY_MAP

# Configuración de estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Colores por polaridad
POLARITY_COLORS = {
    0: '#FF4444',    # Rojo para negativo
    2: '#888888',    # Gris para neutral
    4: '#44AA44'     # Verde para positivo
}


def save_figure(fig, save_path: Path, dpi: int = 300) -> None:
    """
    Guarda una figura en el path especificado.
    
    Args:
        fig: Figura de matplotlib a guardar.
        save_path: Ruta donde guardar la figura.
        dpi: Resolución de la imagen.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"✓ Figura guardada en: {save_path}")


def plot_wordcloud_by_polarity(
    df: pd.DataFrame,
    text_column: str = 'text',
    polarity_column: str = 'polarity',
    save_path: Optional[Path] = None
) -> None:
    """
    Genera wordclouds separados por polaridad.
    
    Args:
        df: DataFrame con los datos.
        text_column: Nombre de la columna con el texto.
        polarity_column: Nombre de la columna de polaridad.
        save_path: Ruta base donde guardar las figuras (opcional).
    """
    polarities = sorted(df[polarity_column].unique())
    n_polarities = len(polarities)
    
    fig, axes = plt.subplots(1, n_polarities, figsize=(7 * n_polarities, 6))
    if n_polarities == 1:
        axes = [axes]
    
    for idx, polarity in enumerate(polarities):
        pol_name = POLARITY_MAP.get(polarity, str(polarity))
        texts = df[df[polarity_column] == polarity][text_column]
        
        # Combinar todos los textos
        all_text = ' '.join(texts.astype(str))
        
        # Generar wordcloud
        color = POLARITY_COLORS.get(polarity, '#000000')
        if polarity == 0:
            cmap = 'Reds'
        elif polarity == 2:
            cmap = 'Greys'
        else:
            cmap = 'Greens'
        
        wordcloud = WordCloud(
            width=800,
            height=600,
            background_color='white',
            colormap=cmap,
            max_words=100,
            relative_scaling=0.5
        ).generate(all_text)
        
        axes[idx].imshow(wordcloud, interpolation='bilinear')
        axes[idx].set_title(f'WordCloud - {pol_name.upper()}',
                            fontsize=14, fontweight='bold', color=color)
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    plt.show()


def plot_temporal_heatmap(
    df: pd.DataFrame,
    hour_column: str = 'hour',
    day_column: str = 'day_of_week',
    polarity_column: str = 'polarity',
    save_path: Optional[Path] = None
) -> None:
    """
    Genera heatmap de actividad temporal por polaridad.
    
    Args:
        df: DataFrame con los datos.
        hour_column: Nombre de la columna de hora.
        day_column: Nombre de la columna de día de la semana.
        polarity_column: Nombre de la columna de polaridad.
        save_path: Ruta donde guardar la figura (opcional).
    """
    polarities = sorted(df[polarity_column].unique())
    n_polarities = len(polarities)
    
    fig, axes = plt.subplots(1, n_polarities, figsize=(8 * n_polarities, 6))
    if n_polarities == 1:
        axes = [axes]
    
    day_names = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
    
    for idx, polarity in enumerate(polarities):
        pol_name = POLARITY_MAP.get(polarity, str(polarity))
        subset = df[df[polarity_column] == polarity]
        
        # Crear tabla pivote
        pivot_data = subset.groupby([day_column, hour_column]).size()
        pivot_data = pivot_data.reset_index(name='count')
        pivot_table = pivot_data.pivot(
            index=day_column, columns=hour_column, values='count'
        ).fillna(0)
        
        # Asegurar que tenemos todos los días
        for day in range(7):
            if day not in pivot_table.index:
                pivot_table.loc[day] = 0
        pivot_table = pivot_table.sort_index()
        
        # Crear heatmap
        color = POLARITY_COLORS.get(polarity, '#000000')
        if polarity == 0:
            cmap = 'Reds'
        elif polarity == 2:
            cmap = 'Greys'
        else:
            cmap = 'Greens'
        
        sns.heatmap(pivot_table, cmap=cmap, ax=axes[idx],
                    cbar_kws={'label': 'Tweets'}, linewidths=0.5)
        axes[idx].set_xlabel('Hora del día', fontweight='bold')
        axes[idx].set_ylabel('Día de la semana', fontweight='bold')
        axes[idx].set_title(f'Heatmap Temporal - {pol_name.upper()}',
                            fontsize=14, fontweight='bold', color=color)
        
        # Convertir etiquetas de días a nombres
        y_labels = []
        for label in axes[idx].get_yticklabels():
            text = label.get_text()
            if text.replace('.', '').isdigit():
                y_labels.append(day_names[int(float(text))])
            else:
                y_labels.append(text)
        axes[idx].set_yticklabels(y_labels, rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    plt.show()


def plot_polarity_distribution(
    df: pd.DataFrame,
    polarity_column: str = 'polarity',
    save_path: Optional[Path] = None
) -> None:
    """
    Grafica la distribución de polaridades en el dataset.
    
    Args:
        df: DataFrame con los datos.
        polarity_column: Nombre de la columna de polaridad.
        save_path: Ruta donde guardar la figura (opcional).
    """
    plt.figure(figsize=(10, 6))
    
    counts = df[polarity_column].value_counts().sort_index()
    labels = [POLARITY_MAP.get(p, str(p)) for p in counts.index]
    
    plt.bar(range(len(counts)), counts.values, color=['red', 'gray', 'green'])
    plt.xticks(range(len(counts)), labels)
    plt.xlabel('Polaridad')
    plt.ylabel('Cantidad de tweets')
    plt.title('Distribución de Polaridad en el Dataset')
    
    for i, v in enumerate(counts.values):
        plt.text(i, v + max(counts.values) * 0.01, str(v), ha='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico guardado en: {save_path}")
    
    plt.show()


def generate_wordcloud(
    texts: List[str],
    title: str = "WordCloud",
    max_words: int = 100,
    background_color: str = 'white',
    colormap: str = 'viridis',
    save_path: Optional[Path] = None
) -> None:
    """
    Genera un wordcloud a partir de una lista de textos.
    
    Args:
        texts: Lista de textos.
        title: Título del gráfico.
        max_words: Número máximo de palabras en el wordcloud.
        background_color: Color de fondo.
        colormap: Mapa de colores.
        save_path: Ruta donde guardar la figura (opcional).
    """
    # Unir todos los textos
    all_text = ' '.join(texts)
    
    # Crear wordcloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        max_words=max_words,
        background_color=background_color,
        colormap=colormap
    ).generate(all_text)
    
    # Graficar
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ WordCloud guardado en: {save_path}")
    
    plt.show()


def plot_wordclouds_by_polarity(
    df: pd.DataFrame,
    text_column: str = 'text_processed',
    polarity_column: str = 'polarity',
    save_dir: Optional[Path] = None
) -> None:
    """
    Genera wordclouds separados por cada clase de polaridad.
    
    Args:
        df: DataFrame con los datos.
        text_column: Nombre de la columna con el texto.
        polarity_column: Nombre de la columna de polaridad.
        save_dir: Directorio donde guardar las figuras (opcional).
    """
    polarities = df[polarity_column].unique()
    
    for polarity in sorted(polarities):
        # Filtrar textos por polaridad
        texts = df[df[polarity_column] == polarity][text_column].tolist()
        
        # Nombre de la polaridad
        pol_name = POLARITY_MAP.get(polarity, str(polarity))
        title = f"WordCloud - {pol_name.capitalize()}"
        
        # Path para guardar
        save_path = None
        if save_dir:
            save_path = save_dir / f"wordcloud_{pol_name}.png"
        
        # Mapa de colores según polaridad
        colormap = {0: 'Reds', 2: 'Greys', 4: 'Greens'}.get(polarity, 'viridis')
        
        # Generar wordcloud
        generate_wordcloud(
            texts=texts,
            title=title,
            colormap=colormap,
            save_path=save_path
        )


def plot_text_length_distribution(
    df: pd.DataFrame,
    text_column: str = 'text',
    polarity_column: str = 'polarity',
    save_path: Optional[Path] = None
) -> None:
    """
    Grafica la distribución de longitud de textos por polaridad.
    
    Args:
        df: DataFrame con los datos.
        text_column: Nombre de la columna con el texto.
        polarity_column: Nombre de la columna de polaridad.
        save_path: Ruta donde guardar la figura (opcional).
    """
    df_plot = df.copy()
    df_plot['text_length'] = df_plot[text_column].str.len()
    df_plot['polarity_label'] = df_plot[polarity_column].map(POLARITY_MAP)
    
    plt.figure(figsize=(12, 6))
    
    for polarity in sorted(df[polarity_column].unique()):
        pol_data = df_plot[df_plot[polarity_column] == polarity]['text_length']
        plt.hist(pol_data, bins=50, alpha=0.5, label=POLARITY_MAP.get(polarity, str(polarity)))
    
    plt.xlabel('Longitud del texto (caracteres)')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de Longitud de Texto por Polaridad')
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico guardado en: {save_path}")
    
    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: List[str],
    title: str = "Matriz de Confusión",
    save_path: Optional[Path] = None
) -> None:
    """
    Grafica una matriz de confusión.
    
    Args:
        cm: Matriz de confusión.
        labels: Etiquetas de las clases.
        title: Título del gráfico.
        save_path: Ruta donde guardar la figura (opcional).
    """
    plt.figure(figsize=(8, 6))
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar=True
    )
    
    plt.xlabel('Predicción')
    plt.ylabel('Valor Real')
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Matriz de confusión guardada en: {save_path}")
    
    plt.show()


def plot_top_words(
    word_scores_df: pd.DataFrame,
    title: str = "Top Palabras",
    n_top: int = 20,
    save_path: Optional[Path] = None
) -> None:
    """
    Grafica las palabras con mayor score (TF-IDF, PMI, etc.).
    
    Args:
        word_scores_df: DataFrame con columnas 'word' y un score.
        title: Título del gráfico.
        n_top: Número de palabras a mostrar.
        save_path: Ruta donde guardar la figura (opcional).
    """
    df_plot = word_scores_df.head(n_top)
    score_col = [col for col in df_plot.columns if col != 'word'][0]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(df_plot)), df_plot[score_col].values, color='steelblue')
    plt.yticks(range(len(df_plot)), df_plot['word'].values)
    plt.xlabel(score_col.upper())
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico guardado en: {save_path}")
    
    plt.show()


def plot_embeddings_2d(
    embeddings: np.ndarray,
    labels: np.ndarray,
    method: str = 'pca',
    title: str = "Visualización de Embeddings",
    save_path: Optional[Path] = None
) -> None:
    """
    Reduce embeddings a 2D y los grafica coloreados por clase.
    
    Args:
        embeddings: Matriz de embeddings (n_samples, n_features).
        labels: Etiquetas de clase.
        method: Método de reducción ('pca' o 'umap').
        title: Título del gráfico.
        save_path: Ruta donde guardar la figura (opcional).
    """
    if method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings)
    elif method == 'umap':
        if not UMAP_AVAILABLE:
            print("⚠️  UMAP no está instalado. Usando PCA en su lugar.")
            reducer = PCA(n_components=2, random_state=42)
            embeddings_2d = reducer.fit_transform(embeddings)
        else:
            reducer = umap.UMAP(n_components=2, random_state=42)
            embeddings_2d = reducer.fit_transform(embeddings)
    else:
        raise ValueError("method debe ser 'pca' o 'umap'")
    
    # Crear DataFrame para plotly
    df_plot = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'label': labels
    })
    
    df_plot['label_name'] = df_plot['label'].map(POLARITY_MAP)
    
    # Graficar con plotly
    fig = px.scatter(
        df_plot,
        x='x',
        y='y',
        color='label_name',
        title=title,
        labels={'x': f'{method.upper()} 1', 'y': f'{method.upper()} 2'},
        opacity=0.6
    )
    
    if save_path:
        fig.write_html(save_path)
        print(f"✓ Visualización guardada en: {save_path}")
    
    fig.show()


def plot_metrics_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    save_path: Optional[Path] = None
) -> None:
    """
    Grafica una comparación de métricas entre múltiples modelos.
    IMPORTANTE para consigna TP3: visualizar comparación de al menos 2 enfoques.
    
    Args:
        metrics_dict: Diccionario {modelo_nombre: {metrica: valor}}.
        save_path: Ruta donde guardar la figura (opcional).
    """
    df_metrics = pd.DataFrame(metrics_dict).T
    
    df_metrics.plot(kind='bar', figsize=(12, 6))
    plt.xlabel('Modelo')
    plt.ylabel('Score')
    plt.title('Comparación de Métricas entre Modelos')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Métrica')
    plt.ylim([0, 1])
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Comparación guardada en: {save_path}")
    
    plt.show()


def plot_hashtag_frequency(
    df: pd.DataFrame,
    text_column: str = 'text',
    top_n: int = 20,
    save_path: Optional[Path] = None
) -> None:
    """
    Analiza y grafica los hashtags más frecuentes.
    Preparado para análisis adicional según consigna TP3.
    
    Args:
        df: DataFrame con los tweets.
        text_column: Columna con el texto.
        top_n: Cantidad de hashtags top a mostrar.
        save_path: Ruta donde guardar la figura (opcional).
    """
    import re
    from collections import Counter
    
    # Extraer todos los hashtags
    all_hashtags = []
    for text in df[text_column]:
        if pd.notna(text):
            hashtags = re.findall(r'#(\w+)', str(text).lower())
            all_hashtags.extend(hashtags)
    
    # Contar frecuencias
    hashtag_counts = Counter(all_hashtags).most_common(top_n)
    
    if not hashtag_counts:
        print("No se encontraron hashtags en los tweets")
        return
    
    # Graficar
    tags, counts = zip(*hashtag_counts)
    
    plt.figure(figsize=(12, 6))
    plt.barh(range(len(tags)), counts, color='steelblue')
    plt.yticks(range(len(tags)), [f'#{tag}' for tag in tags])
    plt.xlabel('Frecuencia')
    plt.title(f'Top {top_n} Hashtags Más Frecuentes')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico de hashtags guardado en: {save_path}")
    
    plt.show()


def plot_mentions_frequency(
    df: pd.DataFrame,
    text_column: str = 'text',
    top_n: int = 20,
    save_path: Optional[Path] = None
) -> None:
    """
    Analiza y grafica las menciones (@usuario) más frecuentes.
    Preparado para análisis adicional según consigna TP3.
    
    Args:
        df: DataFrame con los tweets.
        text_column: Columna con el texto.
        top_n: Cantidad de menciones top a mostrar.
        save_path: Ruta donde guardar la figura (opcional).
    """
    import re
    from collections import Counter
    
    # Extraer todas las menciones
    all_mentions = []
    for text in df[text_column]:
        if pd.notna(text):
            mentions = re.findall(r'@(\w+)', str(text).lower())
            all_mentions.extend(mentions)
    
    # Contar frecuencias
    mention_counts = Counter(all_mentions).most_common(top_n)
    
    if not mention_counts:
        print("No se encontraron menciones en los tweets")
        return
    
    # Graficar
    users, counts = zip(*mention_counts)
    
    plt.figure(figsize=(12, 6))
    plt.barh(range(len(users)), counts, color='coral')
    plt.yticks(range(len(users)), [f'@{user}' for user in users])
    plt.xlabel('Frecuencia')
    plt.title(f'Top {top_n} Usuarios Más Mencionados')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico de menciones guardado en: {save_path}")
    
    plt.show()
