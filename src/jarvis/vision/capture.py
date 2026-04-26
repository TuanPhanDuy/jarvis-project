"""Tools: capture_camera (YOLO detection) and describe_scene (Claude vision captioning)."""
from __future__ import annotations

import base64
import time
from pathlib import Path


def handle_capture_camera(tool_input: dict, screenshots_dir: Path) -> str:
    try:
        import cv2
        from ultralytics import YOLO

        camera_index = int(tool_input.get("camera_index", 0))
        save_image = bool(tool_input.get("save_image", True))

        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            return "ERROR: Could not open camera."
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            return "ERROR: Failed to capture frame from camera."

        model = YOLO("yolov8n.pt")  # nano model — auto-downloads on first use
        results = model(frame, verbose=False)

        detections = []
        for result in results:
            for box in result.boxes:
                label = model.names[int(box.cls[0])]
                conf = float(box.conf[0])
                detections.append(f"{label} ({conf:.0%})")

        saved_msg = ""
        if save_image:
            screenshots_dir.mkdir(parents=True, exist_ok=True)
            out_path = screenshots_dir / f"capture_{int(time.time())}.jpg"
            cv2.imwrite(str(out_path), results[0].plot())
            saved_msg = f" Annotated image saved to: {out_path}"

        if not detections:
            return f"Snapshot taken. No objects detected.{saved_msg}"
        return f"Snapshot taken. Detected: {', '.join(detections)}.{saved_msg}"

    except Exception as e:
        return f"ERROR: capture_camera failed — {e}"


def handle_describe_scene(tool_input: dict, screenshots_dir: Path, api_key: str) -> str:
    """Capture a webcam frame and return a natural-language scene description via Claude vision."""
    try:
        import cv2
        import anthropic

        camera_index = int(tool_input.get("camera_index", 0))
        prompt = str(tool_input.get("prompt", "Describe what you see in detail."))

        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            return "ERROR: Could not open camera."
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            return "ERROR: Failed to capture frame from camera."

        # Encode frame as JPEG bytes → base64
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return "ERROR: Failed to encode frame as JPEG."
        img_b64 = base64.standard_b64encode(buf.tobytes()).decode()

        # Save raw snapshot for audit
        screenshots_dir.mkdir(parents=True, exist_ok=True)
        out_path = screenshots_dir / f"scene_{int(time.time())}.jpg"
        cv2.imwrite(str(out_path), frame)

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            system=[{
                "type": "text",
                "text": "You are JARVIS's vision system. Describe what you see concisely and factually.",
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        description = response.content[0].text
        return f"Scene description: {description}\n(Snapshot saved: {out_path})"

    except Exception as e:
        return f"ERROR: describe_scene failed — {e}"


DESCRIBE_SCHEMA: dict = {
    "name": "describe_scene",
    "description": (
        "Take a webcam snapshot and return a natural-language description of the scene "
        "using Claude's vision capability. More detailed than capture_camera's object labels."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "camera_index": {
                "type": "integer",
                "description": "Camera device index (default 0 = primary webcam).",
                "default": 0,
            },
            "prompt": {
                "type": "string",
                "description": "What to focus on (default: 'Describe what you see in detail.').",
                "default": "Describe what you see in detail.",
            },
        },
        "required": [],
    },
}

SCHEMA: dict = {
    "name": "capture_camera",
    "description": (
        "Take a webcam snapshot and run YOLOv8 object detection. "
        "Returns detected objects with confidence scores."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "camera_index": {
                "type": "integer",
                "description": "Camera device index (default 0 = primary webcam).",
                "default": 0,
            },
            "save_image": {
                "type": "boolean",
                "description": "Save the annotated image to disk (default true).",
                "default": True,
            },
        },
        "required": [],
    },
}
