import io
import os
import base64
import requests
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from PIL import Image

# Lazy-loaded globals
_txt_model = None
_clip_model = None
_preprocess = None
_device = "cuda" if os.environ.get("FORCE_CUDA","0")=="1" else ("cuda" if __import__("torch").cuda.is_available() else "cpu")

app = FastAPI(title="Granola Multimodal Embeddings API", version="0.1.0")

class TextsIn(BaseModel):
    texts: List[str]

class EmbedOut(BaseModel):
    vectors: List[List[float]]

def _load_text_model():
    global _txt_model
    if _txt_model is None:
        from sentence_transformers import SentenceTransformer
        # multilingual & small -> good CPU perf
        _txt_model = SentenceTransformer("intfloat/multilingual-e5-small")
    return _txt_model

def _load_clip():
    global _clip_model, _preprocess
    if _clip_model is None or _preprocess is None:
        import open_clip
        import torch
        _clip_model, _, _preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        _clip_model = _clip_model.to(_device)
        _clip_model.eval()
    return _clip_model, _preprocess

def _pil_from_any(image_file: Optional[UploadFile], image_b64: Optional[str], image_url: Optional[str]):
    from PIL import Image
    if image_file is not None:
        return Image.open(image_file.file).convert("RGB")
    if image_b64:
        raw = base64.b64decode(image_b64)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    if image_url:
        resp = requests.get(image_url, timeout=20)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    raise HTTPException(status_code=400, detail="Provide one of: file, image_b64, image_url.")

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/embed_text", response_model=EmbedOut)
def embed_text(payload: TextsIn):
    model = _load_text_model()
    vectors = model.encode(payload.texts, normalize_embeddings=True).tolist()
    return {"vectors": vectors}

@app.post("/embed_image", response_model=EmbedOut)
def embed_image(
    image_file: UploadFile | None = File(default=None),
    image_b64: str | None = Form(default=None),
    image_url: str | None = Form(default=None),
):
    img = _pil_from_any(image_file, image_b64, image_url)
    clip_model, preprocess = _load_clip()
    import torch
    t = preprocess(img).unsqueeze(0).to(_device)
    with torch.no_grad():
        feats = clip_model.encode_image(t)
        feats /= feats.norm(dim=-1, keepdim=True)
    return {"vectors": feats.cpu().numpy().tolist()}

# --- Optional helper for n8n ingest/debug ---
class UpsertProduct(BaseModel):
    article: str
    title: Optional[str] = None
    description: Optional[str] = None
    image_url: Optional[str] = None
    nutrition: Optional[str] = None

@app.post("/debug/summarize_product")
def summarize_product(p: UpsertProduct):
    """Purely for testing end-to-end without a DB; echoes normalized fields."""
    tags = []
    text = (p.description or "") + " " + (p.nutrition or "")
    low = text.lower()
    if "без глютену" in low or "gluten free" in low:
        tags.append("gluten_free")
    if "веган" in low or "vegan" in low:
        tags.append("vegan")
    return {"article": p.article, "title": p.title, "image_url": p.image_url, "tags": tags}
