import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

MODEL_PATH = os.path.join("models", "mobilenet_skin.pt")

# Mesma ordem do ImageFolder (alfabética): benign=0, melanoma=1
CLASSES = {0: "lesão benigna", 1: "melanoma (suspeito)"}


def build_model(pretrained: bool = False) -> nn.Module:
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    model.classifier[1] = nn.Linear(model.last_channel, len(CLASSES))
    return model


def load_classification_model(path: str = MODEL_PATH):
    """Carrega o modelo TREINADO. Retorna None se o arquivo de pesos não existir."""
    if not os.path.exists(path):
        return None
    ckpt = torch.load(path, map_location="cpu")
    model = build_model(pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    model.threshold = float(ckpt.get("threshold", 0.5))  # limiar de alta sensibilidade
    return model


def get_image_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def check_image_quality(image: Image.Image) -> list:
    """Checagem simples (heurística) de brilho e nitidez. Ajuste os limiares aos seus dados."""
    gray = np.asarray(image.convert("L").resize((224, 224)), dtype=np.float32)
    brightness = float(gray.mean())
    lap = (gray[1:-1, 1:-1] * 4 - gray[:-2, 1:-1] - gray[2:, 1:-1]
           - gray[1:-1, :-2] - gray[1:-1, 2:])
    sharpness = float(lap.var())

    issues = []
    if brightness < 50:
        issues.append("imagem muito escura")
    if brightness > 220:
        issues.append("imagem muito clara / estourada")
    if sharpness < 30:
        issues.append("imagem possivelmente desfocada")
    return issues


def classify_skin_lesion(image: Image.Image, model: nn.Module) -> dict:
    tensor = get_image_transforms()(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1)[0]

    p_mel = probs[1].item()
    thr = getattr(model, "threshold", 0.5)

    if p_mel >= thr:
        level = "suspeito"
    elif p_mel >= thr / 2:
        level = "indeterminado"
    else:
        level = "baixo"

    return {
        "level": level,
        "prob_melanoma": p_mel * 100,
        "prob_benign": probs[0].item() * 100,
        "threshold": thr * 100,
    }