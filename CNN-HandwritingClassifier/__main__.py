######################################################
###             ML-kaggle-AZ-Handwritte            ###
######################################################


# Importar bibliotecas necessárias
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt



#DataSet para treinamento
DATASET = pd.read_csv("./A_Z-Handwritten-Data.csv")

# Separa rótulos (letras) e imagens
y = DATASET['0'].values
X = DATASET.drop('0', axis=1).values

# Normaliza os pixels (0–255 → 0–1)
X = X / 255.0

# Reshape para formato de imagem: [n, 28, 28, 1]
X = X.reshape(-1, 28, 28, 1)

# Codifica as labels como one-hot
y = to_categorical(y, num_classes=26)

# Split para treino/teste
# Divide uma parte do modelo para teste(20%) e outra para treinamento(80%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Cria o modelo CNN Com keras
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),  # Ajuda a evitar overfitting
    layers.Dense(26, activation='softmax')  # 26 letras
])

# Compila o modelo
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Treina o modelo
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=128,
    validation_data=(X_test, y_test)
)

# Salva o modelo treinado
model.save("modelo_letras.keras")


# Ajuda para visualizar o desempenho do modelo
plt.plot(history.history['accuracy'], label='Treinamento')
plt.plot(history.history['val_accuracy'], label='Validação')
plt.title("Acurácia por época")
plt.xlabel("Épocas")
plt.ylabel("Acurácia")
plt.legend()
plt.show()

# Avaliar o modelo em teste
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Acurácia final: {accuracy:.4f}")


