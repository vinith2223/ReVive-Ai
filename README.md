# ReVive AI

Detection pipeline changed from BLIP+Groq captioning to **YOLOv8 object
detection + a Cloudflare Workers AI vision LLM**, matching your latest
notebook. Everything downstream (classification, receiver matching,
logistics, sustainability, report) is unchanged in behavior.

## What changed from the previous version

- `agents.py` — rewritten around `yolo_detection()` (ultralytics YOLOv8n)
  and `vision_llm()` (Cloudflare's `llama-3.2-11b-vision-instruct`), with
  the same `WEIGHT_RULES` / `CATEGORY_MAP` / `NORMALIZE_MAP` lookup tables
  from your notebook. Multiple images/video frames are supported by running
  detection on each one and merging counts by normalized waste type before
  computing weight — your notebook only handled a single `image_path`.
- `graph_build.py` — node ids renamed (`classify`, `recommend`,
  `generate_report`) so they don't collide with the state field names
  `classification` / `report` your new `EcoShaadiState` uses. LangGraph
  rejects a node whose name matches a state key; this bit us with the
  previous version's state names too.
- `main.py` — reads `result["classification"]` / `result["report"]`
  (matching your new state) instead of the old `classification_result`
  / `report_text` field names.
- `requirements.txt` — dropped `transformers`, `torch` (pinned),
  `langchain`, `langchain-groq` (none of these are used by the new
  pipeline); added `ultralytics` (pulls in a compatible `torch` itself)
  and `requests`.
- `.env.example` — now asks for `CLOUDFLARE_ACCOUNT_ID` /
  `CLOUDFLARE_AUTH_TOKEN` instead of `GROQ_API_KEY`, since the new
  pipeline doesn't call Groq at all.

## ⚠️ Rotate your credentials

Two secrets were hardcoded in plain text in the notebook you shared and are
now exposed to me:
- `GROQ_API_KEY` (cell 2) — unused by the new pipeline, but still rotate it
  at https://console.groq.com/keys since it was exposed.
- `CLOUDFLARE_AUTH_TOKEN` and `CLOUDFLARE_ACCOUNT_ID` (cell 4) — rotate the
  token at https://dash.cloudflare.com/ → your account → API Tokens.

## Setup

```bash
cd backend
python3 -m venv venv
source venv/bin/activate         # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env, fill in CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_AUTH_TOKEN

uvicorn app.main:app --reload --port 8000
```

Open `http://127.0.0.1:8000` — the backend also serves the frontend, no
separate server needed.

The first request downloads the small YOLOv8n weights file (~6MB) from
Ultralytics automatically; this needs outbound internet access once.

## Notes

- `receivers_data.py` is still demo data (placeholder phone numbers) — swap
  in a real partner directory before going live.
- YOLO's built-in classes only map "bottle" → Plastic Bottles today; the
  vision LLM call is what catches flowers, banana leaves, decorations, etc.
  If Cloudflare ever errors or is unreachable, `vision_agent` degrades
  gracefully (logs and continues with whatever YOLO found) rather than
  crashing the whole request.
