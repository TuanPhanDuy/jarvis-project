"""Tool: recognize_face — detect faces in a webcam snapshot via OpenCV Haar cascades."""
from __future__ import annotations

import time
from pathlib import Path


def handle_recognize_face(tool_input: dict, screenshots_dir: Path) -> str:
    try:
        import cv2

        camera_index = int(tool_input.get("camera_index", 0))
        save_image = bool(tool_input.get("save_image", True))

        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            return "ERROR: Could not open camera."
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            return "ERROR: Failed to capture frame from camera."

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        count = len(faces)

        saved_msg = ""
        if save_image and count > 0:
            screenshots_dir.mkdir(parents=True, exist_ok=True)
            out_path = screenshots_dir / f"faces_{int(time.time())}.jpg"
            annotated = frame.copy()
            for (x, y, w, h) in faces:
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imwrite(str(out_path), annotated)
            saved_msg = f" Annotated image saved to: {out_path}"

        if count == 0:
            return "No faces detected in webcam snapshot."
        return f"Detected {count} face(s) in webcam snapshot.{saved_msg}"

    except Exception as e:
        return f"ERROR: recognize_face failed — {e}"


SCHEMA: dict = {
    "name": "recognize_face",
    "description": (
        "Take a webcam snapshot and detect human faces using OpenCV Haar cascades. "
        "Returns the number of faces detected and saves an annotated image."
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
