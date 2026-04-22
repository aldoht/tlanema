import asyncio
import json
import logging
import os

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from realtimePredictions.main import (
    FONT_SIZE,
    FONT_THICKNESS,
    HANDEDNESS_TEXT_COLOR,
    detect_img,
    draw_landmarks_on_image,
    map_prediction_to_label,
    model_dict,
    process_hand_image,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LSM Hand Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_IDX = int(os.getenv("MODEL_IDX", 0))
model_path, greyscale_value, channel_size = model_dict[MODEL_IDX]

if not os.path.exists(model_path):
    raise RuntimeError(
        f"Model not found at {model_path}. Set MODEL_IDX env var correctly."
    )

logger.info("Loading model from %s ...", model_path)
loaded_cnn: tf.keras.Model = tf.keras.models.load_model(model_path, compile=True)
logger.info("Model loaded.")


class ConnectionManager:
    def __init__(self) -> None:
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.active.append(ws)
        logger.info("Client connected  | total=%d", len(self.active))

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self.active:
            self.active.remove(ws)
        logger.info("Client disconnected | total=%d", len(self.active))


manager = ConnectionManager()


def _process_frame(raw: bytes) -> tuple[bytes, str]:
    """
    Full pipeline for one frame:
      decode → detect → annotate landmarks → predict sign → encode

    Returns:
        (annotated_jpeg_bytes, prediction_label_or_empty_string)
    """

    arr = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode frame bytes.")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    original_detection, image, _ = detect_img(mp_image)

    if original_detection == -1:
        cv2.putText(
            bgr,
            "No hand detected",
            (10, 30),
            cv2.FONT_HERSHEY_DUPLEX,
            FONT_SIZE,
            (0, 0, 200),
            FONT_THICKNESS,
            cv2.LINE_AA,
        )
        _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buf.tobytes(), ""

    annotated_rgb, text_x, text_y = draw_landmarks_on_image(
        mp_image.numpy_view(), original_detection
    )

    prediction_text = ""
    _, hand_img = process_hand_image(
        image, greyscale=greyscale_value, height=224, width=224
    )
    if hand_img is not None:
        hand_reshaped = hand_img.reshape(1, 224, 224, channel_size)
        prediction = loaded_cnn.predict(hand_reshaped, verbose=0)
        label = map_prediction_to_label(prediction)
        confidence = float(np.max(prediction)) * 100
        prediction_text = f"{label}: {confidence:.1f}%"

        cv2.putText(
            annotated_rgb,
            prediction_text,
            (text_x, text_y + FONT_SIZE * 30),
            cv2.FONT_HERSHEY_DUPLEX,
            FONT_SIZE,
            HANDEDNESS_TEXT_COLOR,
            FONT_THICKNESS,
            cv2.LINE_AA,
        )

    annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".jpg", annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes(), prediction_text


@app.websocket("/ws/annotate")
async def annotate_stream(websocket: WebSocket):
    await manager.connect(websocket)
    loop = asyncio.get_event_loop()

    try:
        while True:
            raw: bytes = await websocket.receive_bytes()

            annotated_bytes, prediction = await loop.run_in_executor(
                None, _process_frame, raw
            )

            # Send annotated frame
            await websocket.send_bytes(annotated_bytes)
            # Send prediction as JSON text
            await websocket.send_text(json.dumps({"prediction": prediction}))

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as exc:
        logger.exception("Error processing frame: %s", exc)
        try:
            await websocket.send_text(json.dumps({"error": str(exc)}))
        except Exception:
            pass
        manager.disconnect(websocket)


@app.get("/health")
async def health():
    return {"status": "ok", "model": model_path}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server.server:app", host="0.0.0.0", port=8000, reload=False)
