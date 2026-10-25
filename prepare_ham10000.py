"""
Organiza o HAM10000 na estrutura que o train.py espera.

Uso:
  python prepare_ham10000.py --csv HAM10000_metadata.csv --images pasta_com_imagens --out data

--images pode ser a pasta raiz; as subpastas (part_1, part_2) são varridas automaticamente.
A divisão treino/validação é feita por lesion_id (a mesma lesão nunca fica nos dois lados).
"""
import argparse
import csv
import os
import shutil

from sklearn.model_selection import GroupShuffleSplit

parser = argparse.ArgumentParser()
parser.add_argument("--csv", required=True)
parser.add_argument("--images", required=True)
parser.add_argument("--out", default="data")
parser.add_argument("--val_size", type=float, default=0.2)
args = parser.parse_args()

# Mapa image_id -> caminho do arquivo
paths = {}
for root, _, files in os.walk(args.images):
    for f in files:
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            paths[os.path.splitext(f)[0]] = os.path.join(root, f)

rows = []
with open(args.csv, newline="", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        if r["image_id"] in paths:
            rows.append((r["image_id"], r["lesion_id"], "melanoma" if r["dx"] == "mel" else "benign"))

print(f"{len(rows)} imagens encontradas de {sum(1 for _ in open(args.csv)) - 1} no CSV")

groups = [r[1] for r in rows]
splitter = GroupShuffleSplit(n_splits=1, test_size=args.val_size, random_state=42)
train_idx, val_idx = next(splitter.split(rows, groups=groups))

counts = {}
for split, idxs in [("train", train_idx), ("val", val_idx)]:
    for i in idxs:
        image_id, _, label = rows[i]
        dest = os.path.join(args.out, split, label)
        os.makedirs(dest, exist_ok=True)
        shutil.copy(paths[image_id], os.path.join(dest, image_id + ".jpg"))
        counts[(split, label)] = counts.get((split, label), 0) + 1

for k in sorted(counts):
    print(k, counts[k])