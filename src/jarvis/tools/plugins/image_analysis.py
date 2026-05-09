"""Plugin: analyze_image — analyze image files with YOLO detection or Ollama vision.

Modes:
  detect   → YOLOv8 object detection (returns objects + confidence scores)
  describe → Ollama vision model scene description
  ocr      → Ollama vision model text extraction (OCR)
"""
from __future__ import annotations

import base64
from pathlib import Path


def handle(tool_input: dict) -> str:
    try:
        image_path = str(tool_input.get("image_path", "")).strip()
        mode = str(tool_input.get("mode", "describe")).lower()
        prompt_override = str(tool_input.get("prompt", "")).strip()

        if not image_path:
            return "ERROR: 'image_path' is required"

        path = Path(image_path)
        if not path.exists():
            return f"ERROR: file not found — {image_path}"
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            return f"ERROR: unsupported format '{path.suffix}'. Use JPG, PNG, BMP, or WEBP."

        if mode == "detect":
            return _yolo_detect(path)
        elif mode == "describe":
            prompt = prompt_override or "Describe this image in detail. What do you see?"
            return _ollama_vision(path, prompt)
        elif mode == "ocr":
            prompt = prompt_override or "Extract all visible text from this image exactly as it appears. Return only the text, preserving layout."
            return _ollama_vision(path, prompt)
        else:
            return f"ERROR: unknown mode '{mode}'. Use: detect, describe, ocr"

    except Exception as e:
        return f"ERROR: analyze_image failed — {e}"


def _yolo_detect(path: Path) -> str:
    import cv2
    from ultralytics import YOLO

    frame = cv2.imread(str(path))
    if frame is None:
        return f"ERROR: could not load image — {path}"

    model = YOLO("yolov8n.pt")
    results = model(frame, verbose=False)

    detections = []
    for result in results:
        for box in result.boxes:
            label = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            detections.append(f"{label} ({conf:.0%})")

    if not detections:
        return f"Image: {path.name}\nNo objects detected."
    return f"Image: {path.name}\nDetected objects: {', '.join(detections)}"


def _ollama_vision(path: Path, prompt: str) -> str:
    import ollama
    from jarvis.config import get_settings

    vision_model = get_settings().vision_model
    img_bytes = path.read_bytes()
    img_b64 = base64.standard_b64encode(img_bytes).decode()

    response = ollama.chat(
        model=vision_model,
        messages=[{
            "role": "user",
            "content": prompt,
            "images": [img_b64],
        }],
    )
    return f"Image: {path.name}\n{response.message.content}"


SCHEMA: dict = {
    "name": "analyze_image",
    "description": (
        "Analyze an image file from disk using local computer vision models. "
        "Modes: 'detect' (YOLO object detection — lists all detected objects with confidence), "
        "'describe' (Ollama vision — natural language scene description), "
        "'ocr' (Ollama vision — extract all text from the image). "
        "Supports JPG, PNG, BMP, WEBP."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "image_path": {
                "type": "string",
                "description": "Absolute or relative path to the image file.",
            },
            "mode": {
                "type": "string",
                "enum": ["detect", "describe", "ocr"],
                "description": "Analysis mode. Default: 'describe'.",
            },
            "prompt": {
                "type": "string",
                "description": "Optional custom prompt for 'describe' or 'ocr' modes.",
            },
        },
        "required": ["image_path"],
    },
}
