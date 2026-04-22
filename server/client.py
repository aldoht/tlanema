import argparse
import asyncio
import json
import sys

import cv2
import numpy as np
import websockets

SERVER_URI = "ws://localhost:8000/ws/annotate"


async def stream(source: int | str) -> None:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        sys.exit(1)

    print(f"[INFO] Connecting to {SERVER_URI} ...")
    async with websockets.connect(SERVER_URI, max_size=10 * 1024 * 1024) as ws:
        print("[INFO] Connected. Press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] Stream ended.")
                break

            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            await ws.send(buf.tobytes())

            frame_data = await ws.recv()
            if not isinstance(frame_data, bytes):
                continue  # or raise, or log
            arr = np.frombuffer(frame_data, dtype=np.uint8)
            annotated = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            pred_data = await ws.recv()
            try:
                pred = json.loads(pred_data)
                if pred.get("error"):
                    print(f"[SERVER ERROR] {pred['error']}")
                elif pred.get("prediction"):
                    print(f"[PREDICTION] {pred['prediction']}")
            except json.JSONDecodeError:
                pass

            if annotated is not None:
                cv2.imshow("Annotated", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[INFO] User quit.")
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", default=0, help="Webcam index (default 0) or path to a video file"
    )
    args = parser.parse_args()

    source = args.source
    try:
        source = int(source)
    except ValueError:
        pass

    asyncio.run(stream(source))
