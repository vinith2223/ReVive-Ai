"""
EcoShaadi AI agents — v2 pipeline.

Detection now combines:
  - YOLOv8 (ultralytics) object detection, for a fast/free first pass
  - A vision LLM (Meta Llama 3.2 11B Vision, via Cloudflare Workers AI) for
    everything YOLO's generic COCO classes can't recognize (flowers, banana
    leaves, decorations, etc.)

Results from both are normalized against a fixed vocabulary, converted to
estimated kg using fixed per-item weight rules, then classified and matched
to receivers exactly as before.

Adapted from the original notebook to:
  - accept multiple images (and video frames) instead of a single Colab upload,
    merging counts for the same waste type across all of them
  - run inside a FastAPI backend instead of a notebook
  - read credentials from the environment / .env instead of being hardcoded
"""

import base64
import json
import os
import re
import gc
import torch
from ultralytics import YOLO
from functools import lru_cache
from typing import TypedDict

import requests

from .receivers_data import receivers_df

VALID_CATEGORIES = ["Reuse", "Recycle", "Compost", "Donation"]

WEIGHT_RULES = {
    "Plastic Bottles": 0.025,
    "Flowers": 0.20,
    "Leftover Food": 0.25,
    "Paper Plates": 0.01,
    "Banana Leaves": 0.05,
    "Decorations": 0.20,
    "Fabric": 0.50,
    "Wood": 1.00,
}

CATEGORY_MAP = {
    "Plastic Bottles": "Recycle",
    "Flowers": "Compost",
    "Leftover Food": "Donation",
    "Paper Plates": "Compost",
    "Banana Leaves": "Compost",
    "Decorations": "Reuse",
    "Fabric": "Reuse",
    "Wood": "Reuse",
}

NORMALIZE_MAP = {
    "plastic water bottles": "Plastic Bottles",
    "plastic bottles": "Plastic Bottles",
    "water bottles": "Plastic Bottles",
    "plastic bottle": "Plastic Bottles",

    "flowers": "Flowers",
    "flower garlands": "Flowers",
    "flower petals": "Flowers",

    "leftover food": "Leftover Food",
    "food": "Leftover Food",
    "rice": "Leftover Food",

    "banana leaf": "Banana Leaves",
    "banana leaves": "Banana Leaves",

    "paper plate": "Paper Plates",
    "paper plates": "Paper Plates",

    "cloth": "Fabric",
    "fabrics": "Fabric",

    "wood pieces": "Wood",
    "wooden planks": "Wood",

    "decorative items": "Decorations",
    "decoration": "Decorations",
    "decorations": "Decorations",
}


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class EcoShaadiState(TypedDict, total=False):
    image_paths: list
    detected_items: list
    vision_warnings: list
    classification: dict
    recommendations: dict
    pickup_plan: dict
    sustainability_metrics: dict
    report: str


# ---------------------------------------------------------------------------
# Model / credential loading (lazy + cached)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_yolo_model():
    from ultralytics import YOLO
    return YOLO("yolov8n.pt")


def _cloudflare_credentials():
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    auth_token = os.environ.get("CLOUDFLARE_AUTH_TOKEN")
    if not account_id or not auth_token:
        raise RuntimeError(
            "CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_AUTH_TOKEN are not set. "
            "Copy backend/.env.example to backend/.env and fill them in."
        )
    return account_id, auth_token


def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def yolo_detection(image_path: str) -> dict:
    # Force PyTorch to use a single thread to minimize RAM footprint on Render Free Tier
    torch.set_num_threads(1)
    
    # Load the model directly inside the function so it can be completely cleared after use
    model = YOLO("yolov8n.pt")
    
    # Run the prediction using CPU and force float16 half-precision to cut memory usage in half
    results = model.predict(source=image_path, device="cpu", half=True, verbose=False)
    
    counts = {}
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            name = model.names[cls]
            counts[name] = counts.get(name, 0) + 1
            
    # CRITICAL: Manually delete the massive model instances and force clear system RAM
    del model
    del results
    gc.collect()
    
    return counts


VISION_PROMPT = """
You are an expert wedding waste management AI.

Analyze the image carefully.

Only return waste items that are ACTUALLY visible in the image.

Do NOT include categories with zero count.

If no wedding waste is visible, return:

{
  "items": []
}

Allowed categories:

- Flowers
- Leftover Food
- Plastic Bottles
- Decorations
- Fabric
- Wood
- Paper Plates
- Banana Leaves

Return ONLY JSON.

Example 1:
{
  "items":[
    {
      "name":"Flowers",
      "count_estimate":20
    }
  ]
}

Example 2:
{
  "items":[]
}
"""


def vision_llm(image_path: str) -> str:
    account_id, auth_token = _cloudflare_credentials()
    image_base64 = encode_image(image_path)

    response = requests.post(
        f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/@cf/meta/llama-3.2-11b-vision-instruct",
        headers={"Authorization": f"Bearer {auth_token}"},
        json={
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                        },
                        {"type": "text", "text": VISION_PROMPT},
                    ],
                }
            ]
        },
        timeout=60,
    )
    response.raise_for_status()
    result = response.json()
    return result["result"]["response"]


# ---------------------------------------------------------------------------
# Agent 1: Vision Agent
# ---------------------------------------------------------------------------

def _detect_for_image(image_path: str) -> tuple:
    """Run YOLO + vision LLM on a single image, return (raw items, warning|None)."""
    detected_items = []
    warning = None

    # 1. YOLO detection (currently only "bottle" is mapped to a waste type)
    try:
        yolo_counts = yolo_detection(image_path)
    except Exception as exc:  # noqa: BLE001
        print(f"[vision_agent] YOLO failed on {image_path}: {exc}")
        yolo_counts = {}
        warning = f"YOLO detection failed: {exc}"

    if "bottle" in yolo_counts:
        detected_items.append({
            "name": "Plastic Bottles",
            "count_estimate": yolo_counts["bottle"],
        })

    # 2. Vision LLM (catches everything YOLO's generic classes can't)
    try:
        vision_output = vision_llm(image_path)
    except Exception as exc:  # noqa: BLE001
        print(f"[vision_agent] vision_llm failed on {image_path}: {exc}")
        vision_output = None
        warning = f"Vision LLM (Cloudflare) call failed: {exc}"

    data = None
    if isinstance(vision_output, dict):
        data = vision_output
    elif isinstance(vision_output, str):
        match = re.search(r"\{.*\}", vision_output, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError as exc:
                print(f"[vision_agent] JSON parsing failed: {exc}")
                warning = f"Vision LLM returned unparseable output: {exc}"

    if data and "items" in data:
        for item in data["items"]:
            if item.get("count_estimate", 0) > 0:
                detected_items.append(item)

    return detected_items, warning


def vision_agent(state: EcoShaadiState) -> dict:
    image_paths = state.get("image_paths") or []
    if not image_paths and state.get("image_path"):
        image_paths = [state["image_path"]]

    all_raw_items = []
    warnings = []
    for path in image_paths:
        items, warning = _detect_for_image(path)
        all_raw_items.extend(items)
        if warning:
            warnings.append(warning)

    # Merge duplicates (same normalized waste type) across every image/frame
    merged = {}
    for item in all_raw_items:
        raw_name = item["name"].lower().strip()
        name = NORMALIZE_MAP.get(raw_name, item["name"])
        count = item.get("count_estimate", 1)

        merged[name] = merged.get(name, 0) + count

    final_items = []
    for name, count in merged.items():
        if count <= 0:
            continue
        weight = count * WEIGHT_RULES.get(name, 0.1)
        final_items.append({
            "name": name,
            "estimated_quantity_kg": round(weight, 2),
        })

    return {"detected_items": final_items, "vision_warnings": warnings}


# ---------------------------------------------------------------------------
# Agent 2: Classification Agent
# ---------------------------------------------------------------------------

def classification_agent(state: EcoShaadiState) -> dict:
    if not state.get("detected_items"):
        return {"classification": {}}

    classifications = {}
    for item in state["detected_items"]:
        name = item["name"]
        classifications[name] = {
            "quantity_kg": item["estimated_quantity_kg"],
            "category": CATEGORY_MAP.get(name, "Recycle"),
        }

    return {"classification": classifications}


# ---------------------------------------------------------------------------
# Agent 3: Recommendation Agent
# ---------------------------------------------------------------------------

def recommendation_agent(state: EcoShaadiState) -> dict:
    if not state.get("classification"):
        return {"recommendations": {}}

    recommendations = {}
    for waste_name, details in state["classification"].items():
        category = details["category"]

        matches = receivers_df[
            (receivers_df["waste_type"] == waste_name)
            & (receivers_df["category"] == category)
        ]

        if len(matches) == 0:
            continue

        matches = matches.sort_values(by="distance_km")
        best = matches.iloc[0]

        recommendations[waste_name] = {
            "receiver": best["receiver"],
            "distance_km": int(best["distance_km"]),
            "contact": best["contact"],
            "category": best["category"],
        }

    return {"recommendations": recommendations}


# ---------------------------------------------------------------------------
# Agent 4: Logistics Agent
# ---------------------------------------------------------------------------

def logistics_agent(state: EcoShaadiState) -> dict:
    if not state.get("recommendations"):
        return {"pickup_plan": {}}

    total_distance = 0
    stops = []

    for waste, receiver in state["recommendations"].items():
        total_distance += receiver["distance_km"]
        stops.append({
            "waste": waste,
            "receiver": receiver["receiver"],
            "distance_km": receiver["distance_km"],
        })

    return {
        "pickup_plan": {
            "vehicle": "Mini Truck",
            "estimated_distance": total_distance,
            "stops": stops,
        }
    }


# ---------------------------------------------------------------------------
# Agent 5: Sustainability Agent
# ---------------------------------------------------------------------------

def sustainability_agent(state: EcoShaadiState) -> dict:
    if not state.get("classification"):
        return {
            "sustainability_metrics": {
                "waste_diverted_kg": 0,
                "co2_saved_kg": 0,
            }
        }

    total = sum(item["quantity_kg"] for item in state["classification"].values())

    return {
        "sustainability_metrics": {
            "waste_diverted_kg": round(total, 2),
            "co2_saved_kg": round(total * 1.8, 2),
        }
    }


# ---------------------------------------------------------------------------
# Agent 6: Report Agent
# ---------------------------------------------------------------------------

def report_agent(state: EcoShaadiState) -> dict:
    if not state.get("classification"):
        return {
            "report": """
========== EcoShaadi Sustainability Report =========

No wedding waste was detected in the uploaded photos/video.

Please upload images containing:
- Flowers
- Leftover Food
- Plastic Bottles
- Decorations
- Fabric
- Wood
- Paper Plates
- Banana Leaves
"""
        }

    report = "\n========== EcoShaadi Sustainability Report =========\n\n"
    report += "Detected Waste\n---------------------------------\n"

    for waste, data in state["classification"].items():
        report += f"- {waste}: {data['quantity_kg']} kg ({data['category']})\n"

    if state.get("recommendations"):
        report += "\nRecommended Receivers\n---------------------------------\n"
        for waste, rec in state["recommendations"].items():
            report += (
                f"\n{waste}\n"
                f"   Receiver : {rec['receiver']}\n"
                f"   Distance : {rec['distance_km']} km\n"
                f"   Contact  : {rec['contact']}\n"
            )

    metrics = state.get("sustainability_metrics", {"waste_diverted_kg": 0, "co2_saved_kg": 0})
    report += f"\n\nWaste Diverted : {metrics['waste_diverted_kg']} kg\n"
    report += f"CO2 Saved      : {metrics['co2_saved_kg']} kg\n"

    return {"report": report}