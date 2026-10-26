import math

import torch

import streamlit as st
from PIL import Image, ImageOps

from image_gate import check_lesion
from isic_service import search_isic_gallery
from model_utils import MODEL_PATH, check_image_quality, classify_skin_lesion, load_classification_model

st.set_page_config(page_title="Dermoscan", page_icon="🩺", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;600&family=Manrope:wght@400;500;600&display=swap');
:root { --ink:#111C2D; --muted:#5A6778; --line:#D9D5CD; --teal:#1E3A5F; --blush:#EBD7D0; }
html, body, .stApp, [class*="st-"] { font-family:'Manrope', system-ui, sans-serif; }
.stApp { background:radial-gradient(circle at 92% -4%, #EBD7D0 0, rgba(235,215,208,0) 36%), #ECEAE5; color:var(--ink); }
#MainMenu, footer, header[data-testid="stHeader"] { visibility:hidden; height:0; }
.block-container { max-width:1080px; padding-top:2.6rem; padding-bottom:4rem; }

.hero { padding:0 0 2rem 0; }
.hero h1 { font-family:'Cormorant Garamond', Georgia, serif; font-weight:600; font-size:4rem;
           letter-spacing:-0.01em; line-height:1; margin:0 0 .6rem 0; color:var(--teal); }
.hero p { font-size:1.12rem; color:var(--muted); margin:0; max-width:34rem; line-height:1.55; }
.hero .note { margin-top:1.1rem; display:inline-block; font-size:.88rem; color:var(--ink);
              background:var(--blush); padding:.4rem .9rem; border-radius:999px; }

button[data-baseweb="tab"] { font-weight:500; font-size:1.02rem; color:var(--muted); padding-left:0; padding-right:1.6rem; }
button[data-baseweb="tab"][aria-selected="true"] { color:var(--ink); }
div[data-baseweb="tab-highlight"] { background:var(--teal); height:2px; }
div[data-baseweb="tab-border"] { background:var(--line); }

.stButton > button { background:var(--teal); color:#fff; border:none; border-radius:999px;
                     padding:.6rem 1.5rem; font-weight:500; }
.stButton > button:hover { background:#142B48; color:#fff; }
[data-testid="stImage"] img { border-radius:18px; }

.result { border-left:3px solid var(--c); padding:4px 0 4px 22px; margin:0 0 2rem 0; }
.result h2 { font-family:'Cormorant Garamond', Georgia, serif; font-weight:600; font-size:2.5rem;
             line-height:1.05; color:var(--c); margin:0 0 .5rem 0; }
.result p { margin:0; color:var(--ink); line-height:1.6; max-width:30rem; }

.scale { position:relative; padding-top:38px; margin:0 14px; }
.scale-bar { height:10px; border-radius:999px; }
.scale-marker { position:absolute; top:0; height:48px; width:0; border-left:2px solid var(--ink); }
.scale-marker span { position:absolute; top:-2px; left:0; transform:translateX(-50%); background:var(--ink);
                     color:#fff; font-size:.82rem; font-weight:600; padding:.15rem .6rem; border-radius:999px; white-space:nowrap; }
.scale-ticks { position:relative; height:22px; margin-top:8px; font-size:.82rem; color:var(--muted); }
.scale-ticks span { position:absolute; white-space:nowrap; }
.scale-ticks .mid { transform:translateX(-50%); }
.detail { margin:1.2rem 0 0 0; font-size:.9rem; color:var(--muted); line-height:1.55; }
.notice { margin-top:1rem; padding:.8rem 1rem; border-radius:12px; background:#F6ECDD; color:#6B4A17; font-size:.92rem; }
.meta { font-size:.88rem; color:var(--muted); line-height:1.5; margin:.3rem 0 0 0; }
.meta b { color:var(--ink); font-weight:600; }
.fine { margin-top:3rem; padding-top:1.2rem; border-top:1px solid var(--line); font-size:.85rem; color:var(--muted); line-height:1.6; }
h3 { font-family:'Cormorant Garamond', Georgia, serif; font-weight:600; font-size:1.8rem; color:var(--teal); }

.hero-art svg { width:100%; max-width:380px; display:block; margin:0 auto; }
.stats { display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin:.4rem 0 0 0; }
.stat { background:#FBFAF7; border:1px solid var(--line); border-radius:20px; padding:1.2rem 1.2rem 1rem 1.2rem; }
.stat .num { font-family:'Manrope', system-ui, sans-serif; font-weight:600; font-size:2.3rem; letter-spacing:-0.02em; font-variant-numeric:lining-nums tabular-nums; line-height:1.05; color:var(--teal); }
.stat .lab { font-size:.88rem; color:var(--muted); margin-top:.45rem; line-height:1.35; }
.stat .bar { height:5px; border-radius:999px; background:#E2DED6; margin-top:.9rem; overflow:hidden; }
.stat .bar i { display:block; height:100%; background:var(--teal); border-radius:999px; }
.grid3 { display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin-top:1.2rem; }
.grid5 { display:grid; grid-template-columns:repeat(5,1fr); gap:14px; margin-top:1rem; }
.card { background:#FBFAF7; border:1px solid var(--line); border-radius:20px; padding:1.2rem; }
.card svg { width:56px; height:56px; display:block; margin-bottom:.7rem; }
.card h4 { font-family:'Cormorant Garamond', Georgia, serif; font-weight:600; font-size:1.4rem; margin:0 0 .3rem 0; color:var(--ink); }
.card p { margin:0; font-size:.9rem; color:var(--muted); line-height:1.5; }
.badge { width:34px; height:34px; border-radius:50%; background:var(--teal); color:#fff; display:flex;
         align-items:center; justify-content:center; font-weight:600; margin-bottom:.8rem; }
.grid2 { display:grid; grid-template-columns:repeat(2,1fr); gap:14px; margin-top:.4rem; }
.chart svg { width:100%; height:auto; display:block; margin-top:.6rem; }
.chart svg text { font-family:'Manrope', system-ui, sans-serif; }
@media (max-width:820px) { .stats { grid-template-columns:repeat(2,1fr); } .grid3, .grid2 { grid-template-columns:1fr; }
                           .grid5 { grid-template-columns:repeat(2,1fr); } }
</style>
""", unsafe_allow_html=True)

LEVELS = {
    "suspeito": ("Padrão suspeito",
                 "O modelo encontrou características associadas a melanoma. Procure um dermatologista para avaliação presencial.",
                 "#8E3B46"),
    "indeterminado": ("Resultado indeterminado",
                      "O modelo não consegue separar o caso com segurança. A avaliação presencial é recomendada.",
                      "#A8742A"),
    "baixo": ("Baixo risco segundo o modelo",
              "Isso não exclui câncer de pele. Se a lesão mudar de cor, forma ou tamanho, sangrar ou coçar, procure um médico.",
              "#2F6B5A"),
}
SEX = {"female": "feminino", "male": "masculino"}


def pct(x: float) -> str:
    return f"{x:.1f}".replace(".", ",") + "%"


def pos(v: float) -> float:
    """Escala em raiz quadrada: dá mais espaço às probabilidades baixas, onde fica o limiar."""
    return math.sqrt(max(0.0, min(v, 100.0)) / 100) * 100


def risk_scale(p: float, thr: float) -> str:
    a, b = pos(thr / 2), pos(thr)
    m = min(max(pos(p), 1.5), 98.5)
    grad = (f"linear-gradient(90deg,#2F6B5A 0%,#2F6B5A {a:.1f}%,#A8742A {a:.1f}%,"
            f"#A8742A {b:.1f}%,#8E3B46 {b:.1f}%,#8E3B46 100%)")
    return (f'<div class="scale"><div class="scale-bar" style="background:{grad}"></div>'
            f'<div class="scale-marker" style="left:{m:.1f}%"><span>{pct(p)}</span></div>'
            f'<div class="scale-ticks"><span style="left:0">0%</span>'
            f'<span class="mid" style="left:{b:.1f}%">limiar {pct(thr)}</span>'
            f'<span style="right:0">100%</span></div></div>')


def _hero_art() -> str:
    rings = "".join(
        f'<ellipse cx="180" cy="150" rx="{118 * (1 - i * 0.13):.1f}" ry="{96 * (1 - i * 0.13):.1f}" '
        f'transform="rotate({-14 + i * 5} 180 150)" fill="none" stroke="#1E3A5F" '
        f'stroke-opacity="{0.16 + i * 0.08:.2f}" stroke-width="{1.2 + i * 0.15:.2f}"/>'
        for i in range(7))
    return ('<div class="hero-art"><svg viewBox="0 0 360 300" xmlns="http://www.w3.org/2000/svg" role="img" '
            'aria-label="Ilustração abstrata de uma lesão de pele em curvas de nível">'
            '<defs><radialGradient id="bl" cx="50%" cy="45%" r="62%"><stop offset="0" stop-color="#F8E9E1"/>'
            '<stop offset="1" stop-color="#E4C0B3"/></radialGradient></defs>'
            '<circle cx="180" cy="150" r="146" fill="none" stroke="#1E3A5F" stroke-opacity=".18" stroke-dasharray="2 9"/>'
            '<circle cx="180" cy="150" r="134" fill="url(#bl)"/>'
            '<circle cx="288" cy="62" r="30" fill="#1E3A5F" fill-opacity=".09"/>'
            '<circle cx="66" cy="238" r="20" fill="#EBD7D0"/>'
            f'{rings}'
            '<ellipse cx="180" cy="150" rx="19" ry="15" transform="rotate(16 180 150)" fill="#1E3A5F"/>'
            '<circle cx="186" cy="146" r="4.5" fill="#EBD7D0" fill-opacity=".85"/>'
            '</svg></div>')


HERO_ART = _hero_art()

STEPS_HTML = (
    '<div class="grid3">'
    '<div class="card"><div class="badge">1</div><h4>Envie a imagem</h4><p>Use a webcam ou uma foto da lesão, com boa luz e foco.</p></div>'
    '<div class="card"><div class="badge">2</div><h4>O modelo analisa</h4><p>Uma rede neural compara a lesão com milhares de casos já conhecidos.</p></div>'
    '<div class="card"><div class="badge">3</div><h4>Veja o resultado</h4><p>Você recebe uma probabilidade e uma orientação, não um diagnóstico.</p></div>'
    '</div>')

_S = 'viewBox="0 0 64 64" fill="none" stroke="#1E3A5F" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"'
ABCDE_HTML = (
    '<div class="grid5">'
    f'<div class="card"><svg {_S}><path d="M32 8a24 24 0 0 0 0 48z" fill="#7A4A3E" stroke="none"/><path d="M32 8a24 24 0 0 1 0 48z"/><path d="M32 8v48" stroke-dasharray="3 4"/></svg><h4>Assimetria</h4><p>Uma metade da pinta é diferente da outra.</p></div>'
    f'<div class="card"><svg {_S}><path d="M32 9c7 0 9 6 15 8s9 7 8 14 3 9-2 15-8 6-14 9-9 6-16 4-10-7-14-12-3-9 0-15 3-9 8-13 9-9 15-10z" fill="#EBD7D0"/></svg><h4>Bordas</h4><p>Contorno irregular, recortado ou mal definido.</p></div>'
    f'<div class="card"><svg {_S}><circle cx="32" cy="32" r="24" fill="#D9A58F"/><circle cx="36" cy="30" r="15" fill="#8A5A4A" stroke="none"/><circle cx="38" cy="29" r="7" fill="#3E2420" stroke="none"/></svg><h4>Cor</h4><p>Mais de um tom: marrom, preto, vermelho, azul ou branco.</p></div>'
    f'<div class="card"><svg {_S}><circle cx="32" cy="28" r="18" fill="#EBD7D0"/><path d="M10 54h44M10 50v8M54 50v8"/></svg><h4>Diâmetro</h4><p>Em geral maior que 6 mm, o tamanho de uma borracha de lápis.</p></div>'
    f'<div class="card"><svg {_S}><circle cx="15" cy="40" r="7" fill="#7A4A3E"/><path d="M26 40h12m-5-5 5 5-5 5"/><circle cx="48" cy="32" r="12" fill="#7A4A3E"/></svg><h4>Evolução</h4><p>Mudança de tamanho, forma, cor ou sintomas ao longo do tempo.</p></div>'
    '</div>')


# Histórico do treino de 5 épocas no Colab. Se treinar de novo, atualize estes valores.
HISTORY = {"loss": [0.4661, 0.3743, 0.3289, 0.2751, 0.2461],
           "auc": [0.899, 0.898, 0.896, 0.895, 0.902]}


def line_chart(values, lo, hi, color, fmt) -> str:
    W, H, L, R, T, B = 320, 170, 46, 306, 20, 136
    n = len(values)
    xs = [L + i * (R - L) / (n - 1) for i in range(n)]
    ys = [B - (v - lo) / (hi - lo) * (B - T) for v in values]
    grid = ""
    for k in range(3):
        v = lo + (hi - lo) * k / 2
        y = B - (v - lo) / (hi - lo) * (B - T)
        grid += (f'<line x1="{L}" x2="{R}" y1="{y:.1f}" y2="{y:.1f}" stroke="#D9D5CD"/>'
                 f'<text x="{L - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="11" fill="#5A6778">{fmt(v)}</text>')
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    area = f"M{xs[0]:.1f},{B} " + " ".join(f"L{x:.1f},{y:.1f}" for x, y in zip(xs, ys)) + f" L{xs[-1]:.1f},{B} Z"
    dots = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#FBFAF7" stroke="{color}" stroke-width="2.5"/>'
                   for x, y in zip(xs, ys))
    ticks = "".join(f'<text x="{x:.1f}" y="{B + 20}" text-anchor="middle" font-size="11" fill="#5A6778">{i + 1}</text>'
                    for i, x in enumerate(xs))
    last = (f'<text x="{xs[-1]:.1f}" y="{ys[-1] - 11:.1f}" text-anchor="end" font-size="12" font-weight="600" '
            f'fill="{color}">{fmt(values[-1])}</text>')
    return (f'<svg viewBox="0 0 {W} {H}" role="img">{grid}<path d="{area}" fill="{color}" fill-opacity=".09"/>'
            f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.5" stroke-linejoin="round"/>'
            f'{dots}{ticks}{last}'
            f'<text x="{(L + R) / 2:.0f}" y="{H - 2}" text-anchor="middle" font-size="11" fill="#5A6778">época</text></svg>')


def training_html() -> str:
    comma = lambda v: f"{v:.2f}".replace(".", ",")
    c1 = line_chart(HISTORY["loss"], 0.20, 0.50, "#8E3B46", comma)
    c2 = line_chart(HISTORY["auc"], 0.88, 0.92, "#1E3A5F", comma)
    return ('<div class="grid2">'
            f'<div class="card chart"><h4>Perda no treino</h4><p>Cai a cada época: o modelo está aprendendo com os exemplos.</p>{c1}</div>'
            f'<div class="card chart"><h4>AUC na validação</h4><p>Fica perto de 0,90: mais épocas trouxeram pouco ganho.</p>{c2}</div>'
            '</div><p class="detail">Os eixos verticais não começam em zero, para destacar a variação.</p>')


def stat(label: str, value: str, bar=None) -> str:
    b = f'<div class="bar"><i style="width:{bar:.0f}%"></i></div>' if bar is not None else ""
    return f'<div class="stat"><div class="num">{value}</div><div class="lab">{label}</div>{b}</div>'


def metrics_html(m) -> str:
    auc, sens, spec = m
    cards = (stat("AUC na validação", f"{auc:.2f}".replace(".", ","), auc * 100)
             + stat("Sensibilidade", pct(sens * 100), sens * 100)
             + stat("Especificidade", pct(spec * 100), spec * 100)
             + stat("Imagens dermatoscópicas do HAM10000", "10.015"))
    return ('<div class="stats">' + cards + '</div>'
            '<p class="detail">Medido na validação, com imagens dermatoscópicas. '
            'Em fotos de webcam, o desempenho tende a ser menor.</p>')


@st.cache_data(show_spinner=False)
def model_metrics():
    try:
        c = torch.load(MODEL_PATH, map_location="cpu")
        return c["val_auc"], c["val_sens"], c["val_spec"]
    except Exception:
        return None


@st.cache_resource
def get_model():
    return load_classification_model()


@st.cache_data(show_spinner=False, ttl=3600)
def cached_search(diagnosis: str):
    samples, error = search_isic_gallery(diagnosis=diagnosis, limit=4)
    if error:
        raise RuntimeError(error)  # exceções não são guardadas no cache
    return samples


model = get_model()

col_h1, col_h2 = st.columns([1.15, 1], vertical_alignment="center", gap="large")
with col_h1:
    st.markdown(
        '<div class="hero"><h1>Dermoscan</h1>'
        '<p>Triagem de lesões de pele com apoio de inteligência artificial, para orientar quando procurar um dermatologista.</p>'
        '<span class="note">Protótipo acadêmico, sem registro na ANVISA</span></div>',
        unsafe_allow_html=True)
with col_h2:
    st.markdown(HERO_ART, unsafe_allow_html=True)

metrics = model_metrics() if model is not None else None
if metrics:
    st.markdown("### Desempenho do modelo")
    st.markdown(metrics_html(metrics), unsafe_allow_html=True)
    st.markdown("### Como o modelo aprendeu")
    st.markdown(training_html(), unsafe_allow_html=True)
    st.write("")

tab1, tab2 = st.tabs(["Analisar lesão", "Galeria ISIC"])

with tab1:
    if model is None:
        st.warning("Modelo treinado não encontrado em `models/mobilenet_skin.pt`. "
                   "Rode `python train.py` para gerá-lo. Sem ele, o app não exibe classificação.")
    else:
        method = st.radio("Como enviar a imagem", ["Webcam", "Enviar foto"], horizontal=True)
        if method == "Webcam":
            source = st.camera_input("Centralize a lesão, com boa luz e foco")
        else:
            source = st.file_uploader("Escolha uma foto da lesão", type=["jpg", "jpeg", "png"])

        if source is not None:
            image = ImageOps.exif_transpose(Image.open(source)).convert("RGB")
            gate = check_lesion(image)
            force = False
            if not gate["ok"]:
                st.write("")
                col_i, col_m = st.columns([1, 1.15], gap="large")
                with col_i:
                    st.image(image, width="stretch")
                with col_m:
                    st.markdown('<div class="result" style="--c:#A8742A"><h2>Não foi possível analisar esta imagem</h2>'
                                f'<p>{gate["message"]}</p></div>', unsafe_allow_html=True)
                    with st.expander("Detalhes técnicos da verificação"):
                        m = gate["metrics"]
                        st.caption(f'Motivo: {gate["status"]} · pele ao redor: {m.get("skin", 0):.2f} · '
                                   f'contraste: {m.get("contrast", 0):.0f} · textura: {m.get("texture", 0):.1f} · '
                                   f'nitidez da borda: {m.get("sharp", 0):.2f}')
                        force = st.checkbox("Analisar mesmo assim (resultado pode não ser confiável)")
            if gate["ok"] or force:
                issues = check_image_quality(image)
                with st.spinner("Analisando a imagem..."):
                    r = classify_skin_lesion(image, model)

                st.write("")
                col_img, col_res = st.columns([1, 1.15], gap="large")
                with col_img:
                    st.image(image, width="stretch")
                with col_res:
                    if not gate["ok"]:
                        st.markdown('<div class="notice">Análise forçada: a verificação da imagem falhou, então o resultado não é confiável.</div>',
                                    unsafe_allow_html=True)
                    title, text, color = LEVELS[r["level"]]
                    st.markdown(f'<div class="result" style="--c:{color}"><h2>{title}</h2><p>{text}</p></div>',
                                unsafe_allow_html=True)
                    st.markdown(risk_scale(r["prob_melanoma"], r["threshold"]), unsafe_allow_html=True)
                    st.markdown(
                        f'<p class="detail">Probabilidade estimada de melanoma: <b>{pct(r["prob_melanoma"])}</b>. '
                        f'O alerta dispara a partir de {pct(r["threshold"])}, um limiar baixo de propósito para '
                        f'não deixar passar casos, o que gera alarmes falsos. A escala amplia a faixa das probabilidades baixas.</p>',
                        unsafe_allow_html=True)
                    if issues:
                        st.markdown(f'<div class="notice">Qualidade da imagem: {", ".join(issues)}. '
                                    'O resultado pode ser pouco confiável. Tente de novo com mais luz e foco.</div>',
                                    unsafe_allow_html=True)
        else:
            st.markdown(STEPS_HTML, unsafe_allow_html=True)

    st.write("")
    st.markdown("### Os sinais ABCDE do melanoma")
    st.write("Uma regra usada por dermatologistas para reconhecer pintas que merecem avaliação.")
    st.markdown(ABCDE_HTML, unsafe_allow_html=True)

with tab2:
    st.markdown("### Casos de referência")
    st.write("Lesões documentadas no ISIC Archive, um banco público de imagens dermatológicas.")
    col_f1, col_f2 = st.columns([2, 1], vertical_alignment="bottom")
    with col_f1:
        diag = st.selectbox("Condição", ["melanoma", "nevus"])
    with col_f2:
        go = st.button("Buscar exemplos", width="stretch")

    if go:
        samples, error = [], None
        with st.spinner("Consultando o ISIC Archive..."):
            try:
                samples = cached_search(diag)
            except Exception as e:
                error = str(e)
        if error:
            st.error(error)
        else:
            st.write("")
            cols = st.columns(len(samples), gap="medium")
            for col, s in zip(cols, samples):
                with col:
                    st.image(s["image"], width="stretch")
                    sex = SEX.get(str(s["sex"]).lower(), "sexo não informado")
                    age_txt = f'{s["age"]} anos' if s["age"] != "N/A" else "idade não informada"
                    site_html = f'{s["anatom_site"]}<br>' if s["anatom_site"] not in ("N/A", None, "") else ""
                    st.markdown(
                        f'<p class="meta"><b>{s["id"]}</b><br>{age_txt}, {sex}<br>{site_html}'
                        f'Licença {s["license"]}<br><a href="{s["gallery_url"]}" target="_blank">Ver no ISIC Gallery</a></p>',
                        unsafe_allow_html=True)

st.markdown(
    '<div class="fine">Ferramenta de apoio educacional e de pesquisa, alinhada à ODS 3 (Saúde e Bem-Estar). '
    'Não substitui consulta médica nem biópsia, e não é um dispositivo médico regularizado. '
    'A imagem é processada em memória e não é armazenada. Fotos de webcam variam em foco e iluminação, '
    'o que pode reduzir a acurácia.</div>',
    unsafe_allow_html=True)