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
import re
from typing import TypedDict

import requests

import onnxruntime as ort
import cv2
import numpy as np

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


import os

_current_dir = os.path.dirname(os.path.abspath(__file__))
_model_path = os.path.join(_current_dir, "yolov8n.onnx")

_session = ort.InferenceSession(_model_path, providers=["CPUExecutionProvider"])
_input_name = _session.get_inputs()[0].name
_IMG_SIZE = 320

_CLASS_NAMES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
    "toothbrush"
]

_CONF_THRESHOLD = 0.4
_IOU_THRESHOLD = 0.45


def _preprocess(image_path, img_size=_IMG_SIZE):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    img_resized = cv2.resize(img, (img_size, img_size))
    img_rgb = img_resized[:, :, ::-1]
    img_chw = img_rgb.transpose(2, 0, 1)
    img_norm = np.ascontiguousarray(img_chw, dtype=np.float32) / 255.0
    return np.expand_dims(img_norm, axis=0)


def _nms(boxes, scores, iou_threshold=_IOU_THRESHOLD):
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep

def yolo_detection(image_path: str) -> dict:
    input_tensor = _preprocess(image_path)
    outputs = _session.run(None, {_input_name: input_tensor})

    predictions = outputs[0]
    if predictions.shape[1] == 84:
        predictions = predictions[0].transpose(1, 0)
    else:
        predictions = predictions[0]

    boxes_xywh = predictions[:, :4]
    class_scores = predictions[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    confidences = np.max(class_scores, axis=1)

    mask = confidences > _CONF_THRESHOLD
    boxes_xywh = boxes_xywh[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    counts = {}
    if len(boxes_xywh) == 0:
        return counts

    x_center, y_center, w, h = (
        boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
    )
    x1 = x_center - w / 2
    y1 = y_center - h / 2
    x2 = x_center + w / 2
    y2 = y_center + h / 2
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    final_class_ids = []
    for cls_id in np.unique(class_ids):
        cls_mask = class_ids == cls_id
        keep = _nms(boxes_xyxy[cls_mask], confidences[cls_mask])
        final_class_ids.extend([cls_id] * len(keep))

    for cls_id in final_class_ids:
        name = _CLASS_NAMES[int(cls_id)]
        counts[name] = counts.get(name, 0) + 1

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