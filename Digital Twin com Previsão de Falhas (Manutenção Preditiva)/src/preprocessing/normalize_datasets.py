# src/preprocessing/normalize_datasets.py

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import csv

from paths import Paths

ARQUIVO_TRAIN_FD001 = Paths.RAW_DIR / 'train_processed_copper' / 'train_FD001.csv'
ARQUIVO_TRAIN_FD002 = Paths.RAW_DIR / 'train_processed_copper' / 'train_FD002.csv'
ARQUIVO_TRAIN_FD003 = Paths.RAW_DIR / 'train_processed_copper' / 'train_FD003.csv'
ARQUIVO_TRAIN_FD004 = Paths.RAW_DIR / 'train_processed_copper' / 'train_FD004.csv'

ARQUIVO_SAIDA_FD001 = Paths.RAW_DIR / 'train_processed_silver' / 'train_FD001_normalized.csv'
ARQUIVO_SAIDA_FD002 = Paths.RAW_DIR / 'train_processed_silver' / 'train_FD002_normalized.csv'
ARQUIVO_SAIDA_FD003 = Paths.RAW_DIR / 'train_processed_silver' / 'train_FD003_normalized.csv'
ARQUIVO_SAIDA_FD004 = Paths.RAW_DIR / 'train_processed_silver' / 'train_FD004_normalized.csv'

# Carrega arquivos
df = pd.read_csv(ARQUIVO_TRAIN_FD004, sep=";", header=None, engine='python')

Database = df.values

# Separar identificadores (ID e ciclo)
df_identifiers = df.iloc[:, :2]         # Colunas 0 e 1
df_features = df.iloc[:, 2:]            # Colunas 2 em diante (features numéricas)

# Normalizar features com Min-Max entre 0 e 1
scaler = MinMaxScaler()
normalized_features = scaler.fit_transform(df_features)

# Juntar identificadores e features normalizadas
df_normalized = pd.concat([
    df_identifiers,
    pd.DataFrame(normalized_features, columns=df_features.columns)
], axis=1)

# Salvar como novo CSV
df_normalized.to_csv(ARQUIVO_SAIDA_FD004, index=False, header=False, sep=";")

print(f"✅ Dados normalizados salvos em: {ARQUIVO_SAIDA_FD001}")