## Atenção: Nesta board especificamente!
# Podem ocorrer erros de leitura ao escrever a letra,
# porque ao ser convertido para um downscale de 280 para 28,
# o algoritmo não está otimizado, não o modelo keras, mas o algoritmo de conversão,
# pode ser que em alguns conjuntos de pixels a resultante gerada esteja incorreta.

import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import tensorflow as tf

# Carregue seu modelo aqui
model = tf.keras.models.load_model("../modelo_letras.keras")  # Altere o nome se necessário

# Parâmetros
CANVAS_SIZE = 280  # Área grande para desenhar
IMG_SIZE = 28  # Tamanho para o modelo


class DrawApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Reconhecimento de Letras")

        self.canvas = tk.Canvas(master, width=CANVAS_SIZE, height=CANVAS_SIZE, bg='black')
        self.canvas.pack()

        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.paint)

        self.btn_frame = tk.Frame(master)
        self.btn_frame.pack()

        self.predict_btn = tk.Button(self.btn_frame, text="Reconhecer", command=self.predict)
        self.predict_btn.pack(side=tk.LEFT)

        self.clear_btn = tk.Button(self.btn_frame, text="Limpar", command=self.clear)
        self.clear_btn.pack(side=tk.LEFT)

        self.result_label = tk.Label(master, text="", font=("Helvetica", 24))
        self.result_label.pack()

    def paint(self, event):
        x1, y1 = (event.x - 10), (event.y - 10)
        x2, y2 = (event.x + 10), (event.y + 10)
        self.canvas.create_oval(x1, y1, x2, y2, fill="white", outline="white")
        self.draw.ellipse([x1, y1, x2, y2], fill=255)

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, CANVAS_SIZE, CANVAS_SIZE], fill=0)
        self.result_label.config(text="")

    def predict(self):
        # Pré-processamento
        image_resized = self.image.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
        image_inverted = ImageOps.invert(image_resized)
        image_np = np.array(image_inverted).astype('float32') / 255.0
        image_np = image_np.reshape(1, IMG_SIZE, IMG_SIZE, 1)

        # Predição
        prediction = model.predict(image_np)
        predicted_class = chr(np.argmax(prediction) + ord('A'))

        # Exibe resultado
        self.result_label.config(text=f"Letra: {predicted_class}")


# Executa a aplicação
if __name__ == "__main__":
    root = tk.Tk()
    app = DrawApp(root)
    root.mainloop()
