"""
Calibra a verificação de imagem com fotos reais.
Uso:  python test_gate.py pasta_com_fotos [outra_pasta ...]
Ex.:  python test_gate.py fotos/lesoes fotos/desenhos fotos/ambientes fotos/pele_sem_lesao
"""
import glob
import os
import sys

from PIL import Image

from image_gate import check_lesion

for folder in sys.argv[1:]:
    print(f"\n== {folder}")
    for path in sorted(glob.glob(os.path.join(folder, "*"))):
        try:
            r = check_lesion(Image.open(path))
        except Exception:
            continue
        m = r["metrics"]
        print(f"{os.path.basename(path):30s} {r['status']:14s} pele={m.get('skin', 0):.2f} "
              f"contraste={m.get('contrast', 0):.0f} textura={m.get('texture', 0):.1f} "
              f"nitidez={m.get('sharp', 0):.2f} tinta={m.get('ink', 0):.0f} "
              f"cor={m.get('chroma', 0):.1f} brilho={m.get('lum', 0):.0f}")