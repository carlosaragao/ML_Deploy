import numpy as np
from PIL import Image, ImageOps


def preprocess_image(file_obj) -> np.ndarray:
    img = Image.open(file_obj).convert("L")
    img = img.resize((28, 28))

    arr = np.array(img, dtype=np.float32)

    # Normaliza para [0..1] antes de decidir inverter
    arr01 = arr / 255.0

    # Se a imagem está "clara" demais em média, normalmente é fundo branco + dígito preto
    # Então invertimos para ficar igual ao MNIST (fundo preto, dígito branco).
    if arr01.mean() > 0.5:
        img = ImageOps.invert(img)
        arr01 = np.array(img, dtype=np.float32) / 255.0

    return arr01.reshape(1, -1)