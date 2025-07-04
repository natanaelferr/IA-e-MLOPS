# src\simulation\rul_model_manual.py

import pandas as pd
import numpy as np
from paths import Paths
import os

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam


# Caminhos de entrada e saída
DATABASE_FD001 = Paths.RAW_DIR / 'train_processed_silver' / 'train_FD001_normalized.csv'
DATABASE_FD002 = Paths.RAW_DIR / 'train_processed_silver' / 'train_FD002_normalized.csv'
DATABASE_FD003 = Paths.RAW_DIR / 'train_processed_silver' / 'train_FD003_normalized.csv'
DATABASE_FD004 = Paths.RAW_DIR / 'train_processed_silver' / 'train_FD004_normalized.csv'

ARQUIVO_PARQUET_FD001 = Paths.PROCESSED_DIR / 'train_FD001.parquet'
ARQUIVO_PARQUET_FD002 = Paths.PROCESSED_DIR / 'train_FD002.parquet'
ARQUIVO_PARQUET_FD003 = Paths.PROCESSED_DIR / 'train_FD003.parquet'
ARQUIVO_PARQUET_FD004 = Paths.PROCESSED_DIR / 'train_FD004.parquet'

MODEL_OUTPUT = Paths.MODELS_DIR / "rul_model.keras"

# DATABASE HEADERS DEFINITION AND DATABASE LOADING
#column_names = ["engine_id", "time_in_cycles", "op_setting_1", "op_setting_2", "op_setting_3"] + \
#               [f"sensor_{i}" for i in range(1, 22)]  # Total 26 colunas

column_names = ['engine_id', 'cycle'] + [f'sensor_{i}' for i in range(1, 18)]

#Le parquet ou csv
#df = pd.read_csv(DATABASE_FD001, sep=",", header=None, names=column_names)
df = pd.read_parquet(ARQUIVO_PARQUET_FD001)
df.columns = ['engine_id', 'cycle'] + [f'sensor_{i}' for i in range(1, df.shape[1] - 1)]


# RUL GENERATION
rul_df = df.groupby('engine_id')['cycle'].max().reset_index()
rul_df.columns = ['engine_id', 'max_cycle']
df = df.merge(rul_df, on='engine_id')
df['RUL'] = df['max_cycle'] - df['cycle']
df.drop(columns=['max_cycle'], inplace=True)

# ==== 3. Prepara os dados ====
X = df.iloc[:, 2:df.columns.get_loc('RUL')].values  # sensores (sem engine_id e cycle)
y = df['RUL'].values         # alvo: RUL

# ==== 4. Trata valores nulos ====
# Substitui NaNs por zero (ou outra estratégia, ex: média/mediana)
X = np.nan_to_num(X, nan=0.0)

# ==== 5. Split treino/teste ====
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==== 6. Modelo de Deep Learning ====
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1)
])

model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])

# ==== 7. Treinamento e Salvar o modelo ====
model.fit(X_train, y_train, epochs=30, batch_size=64, validation_split=0.2)
model.save(MODEL_OUTPUT)

# ==== 8. Avaliação ====
loss, mae = model.evaluate(X_test, y_test)
print(f"Loss (MSE): {loss:.4f} - MAE: {mae:.4f}")




