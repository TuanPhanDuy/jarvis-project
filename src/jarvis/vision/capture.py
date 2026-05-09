"""Tools: capture_camera (YOLO detection) and describe_scene (Ollama vision)."""
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

        model = YOLO("yolov8n.pt")
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


def handle_describe_scene(tool_input: dict, screenshots_dir: Path, vision_model: str) -> str:
    """Describe a scene from a webcam frame or an image file via Ollama vision."""
    try:
        import ollama

        prompt = str(tool_input.get("prompt", "Describe what you see in detail."))
        image_path = str(tool_input.get("image_path", "")).strip()

        if image_path:
            # Analyze an existing image file
            from pathlib import Path as _Path
            path = _Path(image_path)
            if not path.exists():
                return f"ERROR: file not found — {image_path}"
            img_b64 = base64.standard_b64encode(path.read_bytes()).decode()
            source_label = f"(Image: {path.name})"
        else:
            # Capture from webcam
            import cv2
            camera_index = int(tool_input.get("camera_index", 0))
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                return "ERROR: Could not open camera."
            ret, frame = cap.read()
            cap.release()
            if not ret or frame is None:
                return "ERROR: Failed to capture frame from camera."
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                return "ERROR: Failed to encode frame as JPEG."
            img_b64 = base64.standard_b64encode(buf.tobytes()).decode()
            screenshots_dir.mkdir(parents=True, exist_ok=True)
            out_path = screenshots_dir / f"scene_{int(time.time())}.jpg"
            cv2.imwrite(str(out_path), frame)
            source_label = f"(Snapshot saved: {out_path})"

        response = ollama.chat(
            model=vision_model,
            messages=[{"role": "user", "content": prompt, "images": [img_b64]}],
        )
        return f"Scene description: {response.message.content}\n{source_label}"

    except Exception as e:
        return f"ERROR: describe_scene failed — {e}"


DESCRIBE_SCHEMA: dict = {
    "name": "describe_scene",
    "description": (
        "Describe a scene using the local Ollama vision model. "
        "Either captures a live webcam frame (default) or analyzes an image file when 'image_path' is given. "
        "Returns a natural-language description. More detailed than capture_camera's object labels."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "image_path": {
                "type": "string",
                "description": "Path to an image file to analyze (JPG/PNG). If omitted, captures from webcam.",
            },
            "camera_index": {
                "type": "integer",
                "description": "Webcam device index (default 0). Ignored when image_path is set.",
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
