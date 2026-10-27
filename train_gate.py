"""
Treina o porteiro (pele / não-pele / tatuagem) com MobileNetV2.

Estrutura de pastas (mínimo ~300 imagens por classe; mais é melhor):

gate_data/
  train/skin/*.jpg        fotos de pele COM e SEM lesões (inclua fotos de webcam/celular, tons de pele variados)
  train/not_skin/*.jpg    objetos, ambientes, rostos distantes... inclua coisas COR DE PELE (madeira, papelão, areia, parede bege)
  train/tattoo/*.jpg      pele tatuada
  val/... (mesmas 3 pastas)

Separe treino/validação por pessoa/fonte, não aleatoriamente por imagem.

Uso:  python train_gate.py --data gate_data --epochs 8
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms

from model_utils import get_image_transforms
from skin_gate import GATE_PATH, build_gate_model

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="gate_data")
parser.add_argument("--epochs", type=int, default=8)
parser.add_argument("--batch", type=int, default=32)
parser.add_argument("--lr", type=float, default=1e-4)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(25),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3),
    transforms.ToTensor(), norm,
])

train_ds = datasets.ImageFolder(os.path.join(args.data, "train"), train_tf)
val_ds = datasets.ImageFolder(os.path.join(args.data, "val"), get_image_transforms())
assert set(train_ds.class_to_idx) == {"not_skin", "skin", "tattoo"}, train_ds.class_to_idx
print("Classes:", train_ds.class_to_idx)

counts = np.bincount(train_ds.targets)
sampler = WeightedRandomSampler([1.0 / counts[t] for t in train_ds.targets],
                                num_samples=len(train_ds), replacement=True)
train_dl = DataLoader(train_ds, batch_size=args.batch, sampler=sampler, num_workers=0)
val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

model = build_gate_model(3)
# começa dos pesos do ImageNet, trocando só o classificador
pre = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
state = {k: v for k, v in pre.state_dict().items() if not k.startswith("classifier")}
model.load_state_dict(state, strict=False)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
os.makedirs(os.path.dirname(GATE_PATH), exist_ok=True)
best = 0.0

for epoch in range(1, args.epochs + 1):
    model.train()
    for x, y in train_dl:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        criterion(model(x), y).backward()
        optimizer.step()

    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in val_dl:
            preds.extend(model(x.to(device)).argmax(1).cpu().numpy())
            labels.extend(y.numpy())
    preds, labels = np.array(preds), np.array(labels)
    acc = float((preds == labels).mean())
    recalls = {n: float((preds[labels == i] == i).mean()) if (labels == i).any() else float("nan")
               for n, i in train_ds.class_to_idx.items()}
    print(f"Época {epoch}/{args.epochs} | acc {acc:.2%} | recall por classe: "
          + ", ".join(f"{n} {r:.0%}" for n, r in recalls.items()))

    if acc > best:
        best = acc
        torch.save({"state_dict": model.state_dict(), "class_to_idx": train_ds.class_to_idx,
                    "val_acc": acc}, GATE_PATH)
        print(f"  ✔ modelo salvo em {GATE_PATH}")

print(f"Melhor acurácia de validação: {best:.2%}")