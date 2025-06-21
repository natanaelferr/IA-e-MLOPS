import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import string

# 1. Carrega o modelo já treinado
model = load_model("modelo_letras.keras")

# 2. Função para converter imagem 28x28 PNG para vetor de entrada
def preprocess_image(path):
    img = Image.open(path).convert("L")         # L = escala de cinza
    img = img.resize((28, 28))                  # garante tamanho correto
    img_array = np.array(img)                   # converte para array numpy
    img_array = img_array / 255.0               # normaliza (0 a 1)
    img_array = img_array.reshape(1, 28, 28, 1)  # reshape para formato de entrada do modelo
    return img_array

# 3. Caminho da imagem que você quer testar
image_path = "ImagemParaTestar/imagemTest.png"  # Substitua pelo caminho da sua imagem PNG

# 4. Pré-processamento
entrada = preprocess_image(image_path)

# 5. Faz a predição
pred = model.predict(entrada)
indice = np.argmax(pred)  # índice da classe com maior probabilidade

# 6. Converte índice para letra
letras = list(string.ascii_uppercase)  # ['A', 'B', ..., 'Z']
letra_predita = letras[indice]

# 7. Exibe resultado
print(f"A letra reconhecida é: {letra_predita}")
