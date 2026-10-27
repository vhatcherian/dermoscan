"""
Monta gate_data/ (skin / not_skin / tattoo) automaticamente, já dividido em train/val.

Fontes:
  skin     <- imagens do seu data/ (HAM10000) + suas fotos em gate_raw/skin/
  not_skin <- datasets públicos baixados pelo torchvision (texturas, flores, pets) + gate_raw/not_skin/
  tattoo   <- só o que você colocar em gate_raw/tattoo/ (baixe de Kaggle/Roboflow)

Uso:  python build_gate_data.py --force
"""
import argparse
import os
import random
import shutil

from PIL import Image
from torchvision import datasets

EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

parser = argparse.ArgumentParser()
parser.add_argument("--raw", default="gate_raw")
parser.add_argument("--out", default="gate_data")
parser.add_argument("--ham", default="data", help="pasta do HAM10000 já organizado (train/val)")
parser.add_argument("--max_per_source", type=int, default=300)
parser.add_argument("--val_size", type=float, default=0.2)
parser.add_argument("--force", action="store_true", help="apaga o gate_data existente")
args = parser.parse_args()
random.seed(42)

if os.path.exists(args.out):
    if not args.force:
        raise SystemExit(f"'{args.out}' já existe. Use --force para apagar e refazer.")
    shutil.rmtree(args.out)


def list_images(folder):
    """Lista imagens, ignorando arquivos ocultos (._*) e máscaras de anotação (annotations/trimaps)."""
    found = []
    for root, _, files in os.walk(folder):
        parts = {x.lower() for x in root.replace("\\", "/").split("/")}
        if parts & {"annotations", "trimaps", "__macosx"}:
            continue
        found += [os.path.join(root, f) for f in files
                  if f.lower().endswith(EXTS) and not f.startswith(".")]
    return found


def is_valid(path):
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def sample(paths, n=None):
    """Embaralha e devolve até n imagens que realmente abrem (n=None: todas as válidas)."""
    paths = list(paths)
    random.shuffle(paths)
    out = []
    for p in paths:
        if is_valid(p):
            out.append(p)
            if n is not None and len(out) >= n:
                break
    return out


# lista de (origem, caminho) por classe
items = {"skin": [], "not_skin": [], "tattoo": []}

# --- fotos do usuário (todas entram, sem limite) ---
for cls in items:
    mine = sample(list_images(os.path.join(args.raw, cls)))
    items[cls] += [("user", p) for p in mine]
    print(f"{cls}: {len(mine)} fotos suas em {args.raw}/{cls}")

# --- skin: HAM10000 ---
ham = list_images(args.ham)
ham = sample(ham, args.max_per_source)
items["skin"] += [("ham", p) for p in ham]
print(f"skin: {len(ham)} do HAM10000 ({args.ham}/)")

# --- not_skin: datasets públicos ---
sources = {
    "dtd": lambda r: datasets.DTD(r, split="train", download=True),
    "flowers": lambda r: datasets.Flowers102(r, split="train", download=True),
    "pets": lambda r: datasets.OxfordIIITPet(r, split="trainval", download=True),
}
for name, fn in sources.items():
    root = os.path.join(args.raw, "_downloads", name)
    try:
        fn(root)
        imgs = sample(list_images(root), args.max_per_source)
        items["not_skin"] += [(name, p) for p in imgs]
        print(f"not_skin: {len(imgs)} de {name}")
    except Exception as e:
        print(f"AVISO: não consegui baixar '{name}': {e}")

# --- divide e copia ---
print()
for cls, lst in items.items():
    random.shuffle(lst)
    n_val = max(1, int(len(lst) * args.val_size)) if len(lst) > 1 else 0
    for i, (src_name, path) in enumerate(lst):
        split = "val" if i < n_val else "train"
        dest = os.path.join(args.out, split, cls)
        os.makedirs(dest, exist_ok=True)
        ext = os.path.splitext(path)[1].lower()
        shutil.copy(path, os.path.join(dest, f"{src_name}_{i}{ext}"))
    print(f"{cls}: {len(lst) - n_val} train, {n_val} val")
    if len(lst) < 100:
        print(f"  ⚠ poucas imagens em '{cls}' — o ideal é 100 a 300 ou mais.")

for cls in items:  # garante que as pastas existam, mesmo vazias
    for split in ("train", "val"):
        os.makedirs(os.path.join(args.out, split, cls), exist_ok=True)