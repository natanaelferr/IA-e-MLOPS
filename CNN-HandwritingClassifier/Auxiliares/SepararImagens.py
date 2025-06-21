from PIL import Image
import os

# Caminho da imagem de entrada
caminho_imagem = '../DatasetToTest/Full Image.png'

# Diretório onde as imagens menores serão salvas
saida_dir = '../DatasetToTest/imagens_28x28'
os.makedirs(saida_dir, exist_ok=True)

# Tamanho dos blocos
largura_bloco = 28
altura_bloco = 28

# Abrir imagem original
imagem = Image.open(caminho_imagem)
largura_total, altura_total = imagem.size

# Iterar sobre a imagem em blocos
contador = 0
for y in range(0, altura_total, altura_bloco):
    for x in range(0, largura_total, largura_bloco):
        # Definir os limites do bloco
        caixa = (x, y, x + largura_bloco, y + altura_bloco)

        # Verificar se o bloco está dentro da imagem
        if caixa[2] <= largura_total and caixa[3] <= altura_total:
            bloco = imagem.crop(caixa)

            # Salvar o bloco com nome que indica a posição (linha, coluna)
            linha = y // altura_bloco
            coluna = x // largura_bloco
            nome_arquivo = f"img_l{linha:03}_c{coluna:03}.png"
            caminho_completo = os.path.join(saida_dir, nome_arquivo)
            bloco.save(caminho_completo)
            contador += 1

print(f"{contador} imagens 28x28 salvas em '{saida_dir}'.")
