import json

notebook_path = r'd:\Diplomatura en ia\trabajo practico 3 -Omar Gonzalez\tp3_nlp_sentiment\notebooks\01_eda.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Define the new cell with feature calculations
feature_calc_cell = {
   "cell_type": "code",
   "execution_count": None,
   "id": "feature_calc_pre_stats",
   "metadata": {},
   "outputs": [],
   "source": [
    "# ============================================\n",
    "# CÁLCULO DE FEATURES PARA ESTADÍSTICAS\n",
    "# ============================================\n",
    "print(\"🔄 Calculando features necesarias para el análisis...\")\n",
    "\n",
    "# 1. Longitudes\n",
    "df_train['text_length'] = df_train['text'].str.len()\n",
    "df_train['word_count'] = df_train['text'].str.split().str.len()\n",
    "\n",
    "# 2. Temporales\n",
    "if 'datetime' not in df_train.columns:\n",
    "    # Optimización: convertir fecha\n",
    "    df_train['date_clean'] = df_train['date'].str.replace(' PDT', '', regex=False)\n",
    "    df_train['datetime'] = pd.to_datetime(df_train['date_clean'], format='%a %b %d %H:%M:%S %Y')\n",
    "df_train['hour'] = df_train['datetime'].dt.hour\n",
    "\n",
    "# 3. Contenido (URLs, Mentions, Hashtags, etc.)\n",
    "# Usamos vectorización simple o apply donde sea necesario\n",
    "print(\"   → Extrayendo URLs, menciones y hashtags...\")\n",
    "df_train['n_urls'] = df_train['text'].apply(lambda x: len(extract_urls(x)))\n",
    "df_train['n_mentions'] = df_train['text'].apply(lambda x: len(extract_mentions(x)))\n",
    "df_train['n_hashtags'] = df_train['text'].apply(lambda x: len(extract_hashtags(x)))\n",
    "\n",
    "print(\"   → Calculando ratios y palabras intensificadas...\")\n",
    "df_train['uppercase_ratio'] = df_train['text'].apply(calculate_uppercase_ratio)\n",
    "df_train['n_intensified'] = df_train['text'].apply(detect_intensified_words)\n",
    "\n",
    "print(\"✅ Todas las features calculadas exitosamente\")"
   ]
}

# Find the index of the statistics cell
insert_index = -1
for i, cell in enumerate(notebook['cells']):
    # Look for the cell that caused the error
    if cell['cell_type'] == 'code' and 'ESTADÍSTICAS DESCRIPTIVAS DE VARIABLES NUMÉRICAS' in "".join(cell['source']):
        insert_index = i
        break

if insert_index != -1:
    # Insert the new cell BEFORE the statistics cell
    notebook['cells'].insert(insert_index, feature_calc_cell)
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    print("Notebook updated successfully: Feature calculation cell inserted.")
else:
    print("Could not find the statistics cell to insert before.")
