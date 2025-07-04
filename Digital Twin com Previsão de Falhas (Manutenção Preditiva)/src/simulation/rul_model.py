import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from paths import Paths
import os

# Caminhos de entrada e saída
input_csv_path = Paths.PROCESSED_DIR / "train_FD001.csv"
model_output_path = Paths.MODELS_DIR / "rul_model.keras"

# Leitura do CSV (espaço separado)
column_names = ["unit_number", "time_in_cycles", "op_setting_1", "op_setting_2", "op_setting_3"] + \
               [f"sensor_{i}" for i in range(1, 22)]  # Total 26 colunas
df = pd.read_csv(input_csv_path, sep="\s+", header=None, names=column_names)

# Geração da RUL
def generate_rul(df):
    rul = df.groupby("unit_number")["time_in_cycles"].transform("max") - df["time_in_cycles"]
    df["RUL"] = rul
    return df

df = generate_rul(df)

# Normalização dos sensores e settings
features = ["op_setting_1", "op_setting_2", "op_setting_3"] + [f"sensor_{i}" for i in range(1, 22)]
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

# Parâmetros de janelamento
window_size = 30
sequence_cols = features

# Criar janelas para LSTM
def create_sequences(df, window_size):
    sequence_data = []
    rul_labels = []
    for unit in df['unit_number'].unique():
        unit_data = df[df['unit_number'] == unit]
        unit_seq = unit_data[sequence_cols].values
        unit_rul = unit_data["RUL"].values
        for i in range(len(unit_seq) - window_size):
            sequence_data.append(unit_seq[i:i+window_size])
            rul_labels.append(unit_rul[i+window_size])
    return np.array(sequence_data), np.array(rul_labels)

X, y = create_sequences(df, window_size)

# Construção do modelo LSTM
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(window_size, len(sequence_cols))))
model.add(LSTM(32))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mae')

# Treinamento do modelo
checkpoint = ModelCheckpoint(model_output_path, save_best_only=True, monitor='loss', mode='min')
model.fit(X, y, epochs=10, batch_size=64, validation_split=0.2, callbacks=[checkpoint], verbose=1)

# Retorna os caminhos dos arquivos criados
model_output_path
