"""Gera um dataset FALSO só para testar se o pipeline roda. Resultados não têm significado."""
import os

import numpy as np
from PIL import Image

for split in ["train", "val"]:
    for cls in ["benign", "melanoma"]:
        d = f"data/{split}/{cls}"
        os.makedirs(d, exist_ok=True)
        for i in range(20):
            base = 90 if cls == "benign" else 160
            arr = np.random.randint(base - 40, base + 40, (256, 256, 3), dtype=np.uint8)
            Image.fromarray(arr).save(f"{d}/{i}.jpg")
print("Dataset falso criado em ./data")