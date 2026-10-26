"""
Verificação de conteúdo: a imagem mostra uma lesão de pele analisável?

Usa regras simples (sem treino) para barrar: ambientes sem pele, pele sem lesão,
enquadramento ruim e marcações/desenhos na pele. Os limiares ficam em CFG e
podem ser calibrados com o test_gate.py.
"""
import numpy as np
from PIL import Image, ImageFilter

CFG = {
    "size": 96,            # imagem é reduzida para size x size antes da análise
    "black_floor": 28,     # pixels quase pretos ligados às bordas = moldura do dermatoscópio
    "min_area": 0.015,     # área mínima da lesão (fração da imagem)
    "max_area": 0.70,      # acima disso, a lesão está perto demais
    "min_contrast": 20.0,  # diferença de luminância lesão x pele ao redor (0-255)
    "min_skin": 0.40,      # fração do entorno da lesão com tom de pele (foto clínica)
    "min_texture": 6.0,    # textura interna mínima; desenho tende a ser chapado
    "flat_texture": 3.5,   # abaixo disso, preenchimento uniforme => artificial
    "max_sharp": 0.30,     # borda abrupta demais (junto com pouca textura) => artificial
    "dark_lum": 55.0,      # região muito escura (brilho médio) e chapada (textura baixa) => tela/tinta, não lesão
    "dark_texture": 8.0,
    "ink_blue": 8.0,       # lesão azul/verde (tinta de caneta) => artificial
    "min_chroma": 3.5,     # a lesão precisa diferir do entorno em COR, não só em brilho (sombra não conta)
}

MESSAGES = {
    "sem_lesao": "Não identificamos uma lesão de pele nesta imagem. Aproxime a câmera e centralize a lesão, com a pele ao redor visível, boa luz e foco.",
    "sem_pele": "A imagem não parece mostrar pele. Aponte a câmera para a região da lesão.",
    "muito_proxima": "A lesão ocupa quase toda a imagem. Afaste um pouco a câmera para mostrar a pele ao redor.",
    "artificial": "Encontramos uma região com aparência artificial, compatível com marcação, desenho ou objeto (cor de tinta, preenchimento uniforme ou borda muito nítida). "
                  "A análise não foi feita, porque o modelo foi treinado apenas com lesões reais.",
}


def _otsu(vals: np.ndarray) -> int:
    hist, _ = np.histogram(vals, bins=256, range=(0, 256))
    total, sum_all = hist.sum(), float(np.dot(np.arange(256), hist))
    wb = sb = 0.0
    best, thr = -1.0, 128
    for t in range(256):
        wb += hist[t]
        if wb == 0:
            continue
        wf = total - wb
        if wf == 0:
            break
        sb += t * hist[t]
        var = wb * wf * (sb / wb - (sum_all - sb) / wf) ** 2
        if var > best:
            best, thr = var, t
    return thr


def _dilate(m: np.ndarray, k: int = 1) -> np.ndarray:
    for _ in range(k):
        p = np.pad(m, 1)
        m = p[1:-1, 1:-1] | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    return m


def _erode(m: np.ndarray, k: int = 1) -> np.ndarray:
    return ~_dilate(~m, k)


def _components(mask: np.ndarray) -> list:
    h, w = mask.shape
    m = mask.tolist()
    seen = [[False] * w for _ in range(h)]
    comps = []
    for y in range(h):
        for x in range(w):
            if m[y][x] and not seen[y][x]:
                seen[y][x] = True
                stack, pts = [(y, x)], []
                while stack:
                    cy, cx = stack.pop()
                    pts.append((cy, cx))
                    for ny, nx in ((cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)):
                        if 0 <= ny < h and 0 <= nx < w and m[ny][nx] and not seen[ny][nx]:
                            seen[ny][nx] = True
                            stack.append((ny, nx))
                comps.append(pts)
    return sorted(comps, key=len, reverse=True)


def _pick(dark: np.ndarray, S: int, c: dict):
    """Maior região escura que pode ser uma lesão (ignora cantos escuros que tocam 2 bordas)."""
    for pts in _components(dark)[:6]:
        area = len(pts) / (S * S)
        if area < c["min_area"]:
            break
        ys, xs = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
        sides = int(ys.min() == 0) + int(ys.max() == S - 1) + int(xs.min() == 0) + int(xs.max() == S - 1)
        if sides >= 2 and area < c["max_area"]:
            continue
        return ys, xs, area
    return None


def _result(status: str, metrics: dict) -> dict:
    return {"ok": status == "ok", "status": status, "message": MESSAGES.get(status, ""), "metrics": metrics}


def check_lesion(image: Image.Image) -> dict:
    c, S = CFG, CFG["size"]
    im = image.convert("RGB").resize((S, S), Image.BILINEAR).filter(ImageFilter.GaussianBlur(1))
    a = np.asarray(im, dtype=np.float32)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    L = 0.299 * R + 0.587 * G + 0.114 * B
    Cr, Cb = (R - L) * 0.713 + 128, (B - L) * 0.564 + 128
    skin_mask = (Cr >= 133) & (Cr <= 173) & (Cb >= 77) & (Cb <= 127) & (L > 60)
    metrics = {"skin": float(skin_mask.mean())}

    # moldura preta do dermatoscópio: pixels quase pretos que tocam 2 ou mais bordas
    valid = np.ones((S, S), dtype=bool)
    for pts in _components(L < c["black_floor"]):
        ys0, xs0 = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
        if int(ys0.min() == 0) + int(ys0.max() == S - 1) + int(xs0.min() == 0) + int(xs0.max() == S - 1) >= 2:
            valid[ys0, xs0] = False
    if valid.mean() < 0.2:
        return _result("sem_lesao", metrics)

    thr0 = _otsu(L[valid])
    cand = None
    for off in (0, 15, 30):  # lesões claras, com pelos ou fundo irregular, aparecem em limiares mais baixos
        cand = _pick(valid & (L <= thr0 - off), S, c)
        if cand is not None:
            break
    if cand is None:
        return _result("sem_lesao", metrics)

    ys, xs, area = cand
    mask = np.zeros((S, S), dtype=bool)
    mask[ys, xs] = True
    ring = _dilate(mask, 5) & ~_dilate(mask, 1) & valid
    contrast = float(L[ring].mean() - L[mask].mean()) if ring.any() else 0.0
    inner = _erode(mask, 2)
    if inner.sum() < 10:
        inner = mask
    texture = float(L[inner].std())
    lum = float(L[mask].mean())
    r, g, b = a[mask].mean(axis=0)
    ink = float(max(b - r, g - r))
    edge = mask & ~_erode(mask, 1)
    gy, gx = np.gradient(L)
    sharp = float(np.hypot(gx, gy)[edge].mean() / max(contrast, 1.0))
    ring_skin = float(skin_mask[ring].mean()) if ring.any() else 0.0
    # diferença de cor (cromaticidade) entre lesão e entorno: sombra muda o brilho, mas quase não muda a cor
    tot = a.sum(axis=2) + 1e-6
    rg = np.stack([R / tot, G / tot])
    chroma = float(np.linalg.norm(rg[:, mask].mean(axis=1) - rg[:, ring].mean(axis=1)) * 100) if ring.any() else 0.0
    ring_skin = float(skin_mask[ring].mean()) if ring.any() else 0.0
    metrics.update(area=area, contrast=contrast, texture=texture, ink=ink, sharp=sharp, skin=ring_skin, chroma=chroma, lum=lum)

    if area > c["max_area"]:
        return _result("muito_proxima", metrics)
    if contrast < c["min_contrast"] or chroma < c["min_chroma"]:
        return _result("sem_lesao", metrics)
    dermo_like = texture >= 4.5 and contrast >= 30 and chroma >= 5.0 and (r - b) >= 8  # dermatoscopia: lesão marrom/preta, fundo sem tom de pele
    if ring_skin < c["min_skin"] and not dermo_like:
        return _result("sem_pele", metrics)
    dark_flat = lum < c["dark_lum"] and texture < c["dark_texture"]
    if dark_flat or ink > c["ink_blue"] or texture < c["flat_texture"] or (texture < c["min_texture"] and sharp > c["max_sharp"]):
        return _result("artificial" if ring_skin >= c["min_skin"] else "sem_pele", metrics)
    return _result("ok", metrics)