# src/preprocessing/prepare_datasets.py

import pandas as pd
from pathlib import Path
import csv

from paths import Paths

ARQUIVO_TRAIN_FD001 = Paths.RAW_DIR / 'CMAPSSData' / 'train_FD001.txt'
ARQUIVO_TRAIN_FD002 = Paths.RAW_DIR / 'CMAPSSData' / 'train_FD002.txt'
ARQUIVO_TRAIN_FD003 = Paths.RAW_DIR / 'CMAPSSData' / 'train_FD003.txt'
ARQUIVO_TRAIN_FD004 = Paths.RAW_DIR / 'CMAPSSData' / 'train_FD004.txt'

ARQUIVO_SAIDA_FD001 = Paths.PROCESSED_DIR / 'train_FD001.csv'
ARQUIVO_SAIDA_FD002 = Paths.PROCESSED_DIR / 'train_FD002.csv'
ARQUIVO_SAIDA_FD003 = Paths.PROCESSED_DIR / 'train_FD003.csv'
ARQUIVO_SAIDA_FD004 = Paths.PROCESSED_DIR / 'train_FD004.csv'

# Conversão com pandas
df = pd.read_csv(ARQUIVO_TRAIN_FD004, sep=r"\s+", header=None, engine='python')

# Exportar para CSV
df.to_csv(ARQUIVO_SAIDA_FD004, index=False)
print(f"Convertido com sucesso para: {ARQUIVO_SAIDA_FD004}")
