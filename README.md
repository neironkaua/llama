# Granola Multimodal Embeddings API (Render)

A tiny FastAPI service that exposes **/embed_text** and **/embed_image** for your granola shop RAG pipeline.

## Endpoints

- `GET /healthz` → `{ ok: true }`
- `POST /embed_text` (JSON)
```
{ "texts": ["Гранола без глютену 300г", "Склад: вівсяні пластівці, арахіс..."] }
```
Response:
```
{ "vectors": [[...],[...]] }
```

- `POST /embed_image` (multipart/form-data OR form)
Send exactly one of:
  - `image_file: <file>`
  - `image_b64: <base64 string>`
  - `image_url: https://...`

Response:
```
{ "vectors": [[...]] }
```

## Deploy on Render

1. Create a **new Web Service** → **"Build & Deploy from a Git repo"** or **"Blueprint (render.yaml)"**.
2. Point to this repo and Render will build the Docker image.
3. Once live, test:
```
curl -X POST "$RENDER_URL/embed_text" -H "Content-Type: application/json"   -d '{"texts":["Яка гранола підійде для дитини без горіхів?"]}'
```

## Notes
- Models: `intfloat/multilingual-e5-small` for text, `open_clip ViT-B-32` for images.
- Works on CPU (free plan) though first call will be slower (weights download).
- You can add Supabase upsert/search later; this starter is focused on **embeddings API**.
