ReVive AI
ReVive AI is a smart waste classification and sustainability analysis pipeline. It utilizes YOLOv8 object detection coupled with a Cloudflare Workers AI vision LLM (llama-3.2-11b-vision-instruct) to identify event waste, map items to localized waste categories, and dynamically route them to appropriate processing streams via LangGraph.
Architecture & Features
Dual-Stage Vision Pipeline: Combines deterministic object tracking via Ultralytics YOLOv8 with contextual analysis via Cloudflare’s Llama 3.2 Vision LLM.
Multi-Frame Processing: Seamlessly handles multiple sequential images or video frames by aggregating detections and normalizing waste types before downstream evaluation.
LangGraph Orchestration: Utilizes state machine nodes (classify, recommend, generate_report) passing an isolated EcoShaadiState tracking dictionary.
Graceful Degradation: The pipeline automatically defaults to YOLO-only extractions if the Cloudflare API endpoint experiences network latency or drops.
Project Structure
Plaintext
my-langgraph-app/
├── .gitignore
├── backend/
│   ├── app/
│   │   ├── agents.py           # YOLO + Cloudflare Vision logic
│   │   ├── graph_build.py      # LangGraph state configuration
│   │   ├── main.py             # FastAPI entrypoint & CORS middleware
│   │   ├── receivers_data.py   # Registry database for local logistics
│   │   └── video_utils.py      # Video framing and image conversion tools
│   ├── .env.example            # Deployment environment template
│   └── requirements.txt        # Production dependencies (Ultralytics, FastAPI, etc.)
└── frontend/
    ├── index.html              # Core client interface
    ├── css/                    # Component stylesheets
    └── js/                     # API client handlers
Local Setup & Development
1. Backend Configuration
Navigate to the backend directory, initialize your isolated environment, and install dependencies:
Bash
cd backend
python3 -m venv venv
source venv/bin/activate         # Windows: venv\Scripts\activate
python3 -m pip install -r requirements.txt
2. Environment Variables
Create your active environment configuration file from the template:
Bash
cp .env.example .envHere is a clean, professional, and well-structured markdown README file tailored for your production deployment.

---

# ReVive AI

ReVive AI is a smart waste classification and sustainability analysis pipeline. It utilizes **YOLOv8 object detection** coupled with a **Cloudflare Workers AI vision LLM** (`llama-3.2-11b-vision-instruct`) to identify event waste, map items to localized waste categories, and dynamically route them to appropriate processing streams via LangGraph.

---

## Architecture & Features

* **Dual-Stage Vision Pipeline:** Combines deterministic object tracking via Ultralytics YOLOv8 with contextual analysis via Cloudflare’s Llama 3.2 Vision LLM.
* **Multi-Frame Processing:** Seamlessly handles multiple sequential images or video frames by aggregating detections and normalizing waste types before downstream evaluation.
* **LangGraph Orchestration:** Utilizes state machine nodes (`classify`, `recommend`, `generate_report`) passing an isolated `EcoShaadiState` tracking dictionary.
* **Graceful Degradation:** The pipeline automatically defaults to YOLO-only extractions if the Cloudflare API endpoint experiences network latency or drops.

---

## Project Structure

```text
my-langgraph-app/
├── .gitignore
├── backend/
│   ├── app/
│   │   ├── agents.py           # YOLO + Cloudflare Vision logic
│   │   ├── graph_build.py      # LangGraph state configuration
│   │   ├── main.py             # FastAPI entrypoint & CORS middleware
│   │   ├── receivers_data.py   # Registry database for local logistics
│   │   └── video_utils.py      # Video framing and image conversion tools
│   ├── .env.example            # Deployment environment template
│   └── requirements.txt        # Production dependencies (Ultralytics, FastAPI, etc.)
└── frontend/
    ├── index.html              # Core client interface
    ├── css/                    # Component stylesheets
    └── js/                     # API client handlers

```

---

## Local Setup & Development

### 1. Backend Configuration

Navigate to the backend directory, initialize your isolated environment, and install dependencies:

```bash
cd backend
python3 -m venv venv
source venv/bin/activate         # Windows: venv\Scripts\activate
python3 -m pip install -r requirements.txt

```

### 2. Environment Variables

Create your active environment configuration file from the template:

```bash
cp .env.example .env

```

Open `.env` and populate it with your active production variables:

```env
CLOUDFLARE_ACCOUNT_ID=your_account_id_here
CLOUDFLARE_AUTH_TOKEN=your_auth_token_here

```

> ⚠️ **Security Warning:** Never check your `.env` file into GitHub. Ensure your root `.gitignore` contains a specific rule for `.env`.

### 3. Launching the Local Server

Boot up the FastAPI gateway with live reload enabled:

```bash
uvicorn app.main:app --reload --port 8000

```

*Note: On its initial execution, the application will automatically pull down the 6MB `yolov8n.pt` weights directly into your backend directory.*

---

## Production Deployment Checklist

### Frontend (GitHub Pages)

1. Push the root project up to GitHub.
2. Under repository **Settings** -> **Pages**, configure deployment to track the `main` branch.
3. Update the folder source path dropdown from `/ (root)` to `/frontend`.
4. Update `frontend/js/` network configurations to look for your live backend URL rather than `localhost`.

### Backend (Render)

1. Provision a new **Web Service** on Render mapped to your GitHub repository.
2. Apply the following key parameters in the control board:
* **Root Directory:** `backend`
* **Build Command:** `pip install -r requirements.txt`
* **Start Command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`


3. Navigate to **Environment Variables** and insert your production keys (`CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_AUTH_TOKEN`). Ensure they match the keys set up inside FastAPI's CORS configuration origins block.
Open .env and populate it with your active production variables:
Code snippet
CLOUDFLARE_ACCOUNT_ID=your_account_id_here
CLOUDFLARE_AUTH_TOKEN=your_auth_token_here
⚠️ Security Warning: Never check your .env file into GitHub. Ensure your root .gitignore contains a specific rule for .env.
3. Launching the Local Server
Boot up the FastAPI gateway with live reload enabled:
Bash
uvicorn app.main:app --reload --port 8000
Note: On its initial execution, the application will automatically pull down the 6MB yolov8n.pt weights directly into your backend directory.
Production Deployment Checklist
Frontend (GitHub Pages)
Push the root project up to GitHub.
Under repository Settings -> Pages, configure deployment to track the main branch.
Update the folder source path dropdown from / (root) to /frontend.
Update frontend/js/ network configurations to look for your live backend URL rather than localhost.
Backend (Render)
Provision a new Web Service on Render mapped to your GitHub repository.
Apply the following key parameters in the control board:
Root Directory: backend
Build Command: pip install -r requirements.txt
Start Command: uvicorn app.main:app --host 0.0.0.0 --port $PORT
Navigate to Environment Variables and insert your production keys (CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_AUTH_TOKEN). Ensure they match the keys set up inside FastAPI's CORS configuration origins block.