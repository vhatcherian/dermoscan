from io import BytesIO

import requests
from PIL import Image

ISIC_SEARCH_URL = "https://api.isic-archive.com/api/v2/images/search/"

# Sintaxe atual do ISIC (campos diagnosis_1, diagnosis_2, diagnosis_3...).
# O campo antigo "diagnosis:melanoma" causava erro 400.
# Cada condição tem uma lista de consultas: a primeira que retornar imagens é usada.
QUERIES = {
    "melanoma": [
        "diagnosis_3:Melanoma*",
        'diagnosis_3:"Melanoma Invasive"',
        "diagnosis_1:Malignant AND melanocytic:true",
    ],
    "nevus": [
        "diagnosis_3:Nevus*",
        "diagnosis_1:Benign AND melanocytic:true",
        "diagnosis_1:Benign",
    ],
}


def _load_image(file_info: dict):
    url = file_info.get("thumbnail_256", {}).get("url") or file_info.get("full", {}).get("url")
    if not url:
        return None
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return Image.open(BytesIO(r.content)).convert("RGB")


def _parse_items(items: list) -> list:
    results = []
    for item in items:
        try:
            img = _load_image(item.get("files", {}))
            if img is None:
                continue
            isic_id = item["isic_id"]
            clinical = item.get("metadata", {}).get("clinical", {})
            results.append({
                "id": isic_id,
                "image": img,
                "age": clinical.get("age_approx", "N/A"),
                "sex": clinical.get("sex", "N/A"),
                "anatom_site": clinical.get("anatom_site_general", "N/A"),
                "license": item.get("copyright_license", "N/A"),
                "attribution": item.get("attribution", "N/A"),
                "gallery_url": f"https://gallery.isic-archive.com/#!/topWithHeader/onlyHeaderTop/gallery?filter=isic_id%3A{isic_id}",
            })
        except Exception as e:  # uma imagem com problema não derruba as outras
            print(f"Imagem ignorada: {e}")
    return results


def search_isic_gallery(diagnosis: str = "melanoma", limit: int = 4):
    """Retorna (lista_de_casos, mensagem_de_erro_ou_None)."""
    queries = QUERIES.get(diagnosis, [f'diagnosis_3:"{diagnosis}"'])
    last_error = "Nenhum resultado encontrado no ISIC para essa condição."

    for q in queries:
        try:
            resp = requests.get(ISIC_SEARCH_URL, params={"query": q, "limit": limit}, timeout=15)
            resp.raise_for_status()
            items = resp.json().get("results", [])
        except requests.HTTPError as e:
            detail = e.response.text[:200] if e.response is not None else ""
            last_error = f"Erro na consulta '{q}': {e} {detail}"
            continue  # tenta a próxima consulta
        except Exception as e:
            return [], f"Falha ao consultar a API do ISIC: {e}"

        results = _parse_items(items)
        if results:
            return results, None

    return [], last_error