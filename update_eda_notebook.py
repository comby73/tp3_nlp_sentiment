import json

notebook_path = r'd:\Diplomatura en ia\trabajo practico 3 -Omar Gonzalez\tp3_nlp_sentiment\notebooks\01_eda.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Define the new cells
new_markdown_cell = {
   "cell_type": "markdown",
   "id": "temp_analysis_md",
   "metadata": {},
   "source": [
    "### 3.1 Análisis Temporal (Opcional)\n",
    "\n",
    "Analizamos si existe alguna relación entre la hora del día o el día de la semana y el sentimiento de los tweets."
   ]
}

new_code_cell = {
   "cell_type": "code",
   "execution_count": None,
   "id": "temp_analysis_code",
   "metadata": {},
   "outputs": [],
   "source": [
    "# ============================================\n",
    "# ANÁLISIS TEMPORAL\n",
    "# ============================================\n",
    "\n",
    "# Convertir fecha a datetime (si no se hizo antes)\n",
    "# El formato es 'Mon Apr 06 22:19:45 PDT 2009'\n",
    "# Usamos errors='coerce' para evitar fallos con formatos raros\n",
    "if 'datetime' not in df_train.columns:\n",
    "    # Optimización: convertir solo una muestra si es muy lento, pero aquí lo hacemos completo\n",
    "    # pd.to_datetime es lento con formatos no estándar, especificamos formato para acelerar\n",
    "    # Nota: PDT (zona horaria) puede causar problemas, lo removemos simple\n",
    "    df_train['date_clean'] = df_train['date'].str.replace(' PDT', '', regex=False)\n",
    "    df_train['datetime'] = pd.to_datetime(df_train['date_clean'], format='%a %b %d %H:%M:%S %Y')\n",
    "\n",
    "# Extraer componentes temporales\n",
    "df_train['hour'] = df_train['datetime'].dt.hour\n",
    "df_train['day_name'] = df_train['datetime'].dt.day_name()\n",
    "\n",
    "# Mapear polaridad para visualización\n",
    "df_train['sentiment_label'] = df_train['polarity'].map({0: 'Negativo', 4: 'Positivo'})\n",
    "\n",
    "# 1. ANÁLISIS POR HORA DEL DÍA\n",
    "hourly_sentiment = df_train.groupby(['hour', 'sentiment_label']).size().unstack(fill_value=0)\n",
    "hourly_pct = hourly_sentiment.div(hourly_sentiment.sum(axis=1), axis=0) * 100\n",
    "\n",
    "plt.figure(figsize=(12, 5))\n",
    "hourly_pct.plot(kind='bar', stacked=True, color=['#e74c3c', '#2ecc71'], ax=plt.gca())\n",
    "plt.title('Distribución de Sentimiento por Hora del Día')\n",
    "plt.xlabel('Hora (0-23)')\n",
    "plt.ylabel('Porcentaje (%)')\n",
    "plt.legend(['Negativo', 'Positivo'], loc='upper right')\n",
    "plt.xticks(rotation=0)\n",
    "plt.tight_layout()\n",
    "plt.savefig('../reports/figuras/sentiment_by_hour.png')\n",
    "plt.show()\n",
    "\n",
    "# 2. ANÁLISIS POR DÍA DE LA SEMANA\n",
    "daily_sentiment = df_train.groupby(['day_name', 'sentiment_label']).size().unstack(fill_value=0)\n",
    "# Ordenar días\n",
    "day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']\n",
    "daily_sentiment = daily_sentiment.reindex(day_order)\n",
    "daily_pct = daily_sentiment.div(daily_sentiment.sum(axis=1), axis=0) * 100\n",
    "\n",
    "plt.figure(figsize=(10, 5))\n",
    "daily_pct.plot(kind='bar', stacked=True, color=['#e74c3c', '#2ecc71'], ax=plt.gca())\n",
    "plt.title('Distribución de Sentimiento por Día de la Semana')\n",
    "plt.xlabel('Día')\n",
    "plt.ylabel('Porcentaje (%)')\n",
    "plt.legend(['Negativo', 'Positivo'], loc='upper right')\n",
    "plt.xticks(rotation=45)\n",
    "plt.tight_layout()\n",
    "plt.savefig('../reports/figuras/sentiment_by_day.png')\n",
    "plt.show()\n",
    "\n",
    "# 3. CONCLUSIONES RÁPIDAS\n",
    "print(\"📊 HALLAZGOS TEMPORALES\")\n",
    "print(\"=\"*50)\n",
    "\n",
    "# Hora más negativa\n",
    "neg_by_hour = hourly_pct['Negativo']\n",
    "most_negative_hour = neg_by_hour.idxmax()\n",
    "print(f\"Hora más negativa: {most_negative_hour}:00 ({neg_by_hour.max():.1f}% negativos)\")\n",
    "\n",
    "# Hora más positiva\n",
    "most_positive_hour = neg_by_hour.idxmin()\n",
    "print(f\"Hora más positiva: {most_positive_hour}:00 ({100-neg_by_hour.min():.1f}% positivos)\")\n",
    "\n",
    "# Día más negativo\n",
    "neg_by_day = daily_pct['Negativo']\n",
    "most_negative_day = neg_by_day.idxmax()\n",
    "print(f\"Día más negativo: {most_negative_day} ({neg_by_day.max():.1f}% negativos)\")"
   ]
}

# Find insertion index (after the descriptive statistics cell)
insert_index = -1
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code' and 'ESTADÍSTICAS DESCRIPTIVAS DE VARIABLES NUMÉRICAS' in "".join(cell['source']):
        insert_index = i + 1
        break

if insert_index != -1:
    notebook['cells'].insert(insert_index, new_code_cell)
    notebook['cells'].insert(insert_index, new_markdown_cell)
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    print("Notebook updated successfully.")
else:
    print("Could not find insertion point.")
