import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf

# Carregue seu modelo
model = tf.keras.models.load_model("modelo_letras.keras")

IMG_SIZE = 28
PIXEL_SIZE = 20  # Tamanho de cada bloco visível

class PixelBoard:
    def __init__(self, master):
        self.master = master
        master.title("IA: Reconhecimento de Letra Manuscrita")

        canvas_size = IMG_SIZE * PIXEL_SIZE
        self.canvas = tk.Canvas(master, width=canvas_size, height=canvas_size, bg='black')
        self.canvas.pack()

        # Imagem real 28x28
        self.image = Image.new("L", (IMG_SIZE, IMG_SIZE), color=0)
        self.draw = ImageDraw.Draw(self.image)

        # Estado visual da interface (para atualizar cor dos quadrados)
        self.rects = [[None for _ in range(IMG_SIZE)] for _ in range(IMG_SIZE)]

        # Cria grade de 28x28
        for i in range(IMG_SIZE):
            for j in range(IMG_SIZE):
                x1 = i * PIXEL_SIZE
                y1 = j * PIXEL_SIZE
                x2 = x1 + PIXEL_SIZE
                y2 = y1 + PIXEL_SIZE
                self.rects[i][j] = self.canvas.create_rectangle(x1, y1, x2, y2, fill="black", outline="gray")

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.paint)

        # Botões
        self.btn_frame = tk.Frame(master)
        self.btn_frame.pack()

        self.predict_btn = tk.Button(self.btn_frame, text="Reconhecer", command=self.predict)
        self.predict_btn.pack(side=tk.LEFT)

        self.clear_btn = tk.Button(self.btn_frame, text="Limpar", command=self.clear)
        self.clear_btn.pack(side=tk.LEFT)

        self.result_label = tk.Label(master, text="", font=("Helvetica", 24))
        self.result_label.pack()

    def paint(self, event):
        i = event.x // PIXEL_SIZE
        j = event.y // PIXEL_SIZE
        if 0 <= i < IMG_SIZE and 0 <= j < IMG_SIZE:
            # Pintar visualmente
            self.canvas.itemconfig(self.rects[i][j], fill="white")
            # Pintar no array real
            self.draw.point((i, j), fill=255)

    def clear(self):
        self.image = Image.new("L", (IMG_SIZE, IMG_SIZE), color=0)
        self.draw = ImageDraw.Draw(self.image)
        for i in range(IMG_SIZE):
            for j in range(IMG_SIZE):
                self.canvas.itemconfig(self.rects[i][j], fill="black")
        self.result_label.config(text="")

    def predict(self):
        image_np = np.array(self.image).astype("float32") / 255.0
        image_np = image_np.reshape(1, IMG_SIZE, IMG_SIZE, 1)
        prediction = model.predict(image_np)
        predicted_class = chr(np.argmax(prediction) + ord("A"))
        self.result_label.config(text=f"Letra: {predicted_class}")

# Executar
if __name__ == "__main__":
    root = tk.Tk()
    app = PixelBoard(root)
    root.mainloop()
