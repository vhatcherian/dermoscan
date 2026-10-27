"""
Porteiro de imagem: decide se a foto é PELE, NÃO-PELE ou TATUAGEM antes do classificador de melanoma.

Dois modos:
  1) Com modelo treinado (models/skin_gate.pt, gerado por train_gate.py): distingue
     pele / não-pele / tatuagem.
  2) Sem modelo (fallback): heurística de cor (YCbCr) que só separa pele de não-pele.
     Ela NÃO detecta tatuagem, pois tinta escura se parece com uma pinta.
"""
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models

from model_utils import get_image_transforms

GATE_PATH = os.path.join("models", "skin_gate.pt")
MIN_SKIN_RATIO = 0.20   # fração mínima de pixels com cor de pele na região central
TATTOO_THRESHOLD = 0.50

MESSAGES = {
    "not_skin": ("Não parece ser pele",
                 "A imagem não mostra pele de forma clara. Aproxime a câmera da lesão, com boa luz, e tente de novo."),
    "tattoo": ("Parece ser uma tatuagem",
               "O modelo identificou tinta de tatuagem, não uma lesão de pele. "
               "A análise de melanoma não se aplica a tatuagens."),
}


def build_gate_model(n_classes: int = 3) -> nn.Module:
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, n_classes)
    return model


def load_gate_model(path: str = GATE_PATH):
    """Retorna o modelo do porteiro, ou None se ainda não foi treinado."""
    if not os.path.exists(path):
        return None
    ckpt = torch.load(path, map_location="cpu")
    names = [n for n, _ in sorted(ckpt["class_to_idx"].items(), key=lambda kv: kv[1])]
    model = build_gate_model(len(names))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    model.class_names = names
    return model


def skin_ratio(image: Image.Image) -> float:
    """Fração de pixels com cor de pele na região central (75%) da imagem."""
    ycc = np.asarray(image.convert("YCbCr").resize((224, 224)), dtype=np.uint8)
    y, cb, cr = ycc[..., 0], ycc[..., 1], ycc[..., 2]
    mask = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127) & (y >= 40)
    return float(mask[28:196, 28:196].mean())


def check_skin(image: Image.Image, gate_model=None) -> dict:
    """Retorna {'label': 'skin'|'not_skin'|'tattoo', 'confidence': %, 'method': str, 'skin_ratio': float}."""
    ratio = skin_ratio(image)

    if gate_model is None:
        label = "skin" if ratio >= MIN_SKIN_RATIO else "not_skin"
        return {"label": label, "confidence": None, "method": "heuristic", "skin_ratio": ratio}

    tensor = get_image_transforms()(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = F.softmax(gate_model(tensor), dim=1)[0]
    p = {n: probs[i].item() for i, n in enumerate(gate_model.class_names)}

    if p.get("tattoo", 0.0) >= TATTOO_THRESHOLD:
        label = "tattoo"
    elif p.get("not_skin", 0.0) > p.get("skin", 0.0):
        label = "not_skin"
    else:
        label = "skin"
    return {"label": label, "confidence": p[label] * 100, "method": "model", "skin_ratio": ratio}