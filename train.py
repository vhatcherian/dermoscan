"""
Fine-tuning do MobileNetV2 para benigno x melanoma.

Estrutura esperada das pastas (baixe imagens do ISIC / HAM10000 e separe
treino/validação POR PACIENTE ou por lesão, nunca aleatoriamente por imagem):

data/
  train/benign/*.jpg
  train/melanoma/*.jpg
  val/benign/*.jpg
  val/melanoma/*.jpg

Uso:  python train.py --data data --epochs 10
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

from model_utils import MODEL_PATH, build_model, get_image_transforms

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="data")
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch", type=int, default=32)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--target_sens", type=float, default=0.95)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    norm,
])

train_ds = datasets.ImageFolder(os.path.join(args.data, "train"), train_tf)
val_ds = datasets.ImageFolder(os.path.join(args.data, "val"), get_image_transforms())
assert train_ds.class_to_idx == {"benign": 0, "melanoma": 1}, train_ds.class_to_idx

# Balanceamento (melanoma costuma ser minoria)
counts = np.bincount(train_ds.targets)
sample_w = [1.0 / counts[t] for t in train_ds.targets]
sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)

train_dl = DataLoader(train_ds, batch_size=args.batch, sampler=sampler, num_workers=0)
val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

model = build_model(pretrained=True).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
best_auc = 0.0

for epoch in range(1, args.epochs + 1):
    model.train()
    total = 0.0
    for x, y in train_dl:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y)

    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for x, y in val_dl:
            p = torch.softmax(model(x.to(device)), dim=1)[:, 1].cpu().numpy()
            probs.extend(p)
            labels.extend(y.numpy())
    probs, labels = np.array(probs), np.array(labels)

    auc = roc_auc_score(labels, probs)
    fpr, tpr, thr = roc_curve(labels, probs)
    idx = int(np.where(tpr >= args.target_sens)[0][0])  # maior limiar com sensibilidade alvo
    threshold = float(min(thr[idx], 1.0))
    sens, spec = float(tpr[idx]), float(1 - fpr[idx])

    print(f"Época {epoch}/{args.epochs} | loss {total/len(train_ds):.4f} | "
          f"AUC {auc:.3f} | limiar {threshold:.3f} -> sens {sens:.2%}, espec {spec:.2%}")

    if auc > best_auc:
        best_auc = auc
        torch.save({"state_dict": model.state_dict(), "threshold": threshold,
                    "val_auc": auc, "val_sens": sens, "val_spec": spec}, MODEL_PATH)
        print(f"  ✔ modelo salvo em {MODEL_PATH}")

print(f"Melhor AUC de validação: {best_auc:.3f}")
