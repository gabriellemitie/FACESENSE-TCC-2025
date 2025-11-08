"""Remote inference server for Facesense PoC.

Endpoints:
- POST /infer_frame -> accepts image file (multipart/form-data, field name 'image')
  returns JSON with posture_class, posture_confidence, posture_label, face_detected

Run with:
    source venv/bin/activate
    uvicorn facesense_posture.remote_server:app --host 0.0.0.0 --port 8000

Note: the server reuses the project's MicrogestureMonitor to load models and MediaPipe pipelines.
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import time
import threading
from pathlib import Path

app = FastAPI(title="Facesense Remote Inference")

# Lazy-loaded monitor and lock
_monitor = None
_monitor_lock = threading.Lock()

from modules.microgesture_model import MicrogestureMonitor


def get_monitor():
    global _monitor
    if _monitor is None:
        # instantiate without starting camera and without drawing on screen
        _monitor = MicrogestureMonitor(
            model_path="models/model_face_temporal.pkl",
            features_json_path="models/features.json",
            posture_model_path="models/best_model.keras",
            beep_callback=None,
            update_callback=None
        )
        _monitor.show_on_screen = False
        # Do not call iniciar() (which opens a camera). We only need the face_mesh/pose and models.
    return _monitor


@app.get("/health")
def health():
    try:
        mon = get_monitor()
        status = {
            "posture_model_loaded": mon.posture_model is not None,
            "microgesture_model_loaded": mon.microgesture_model is not None,
            "feature_count": len(mon.feature_names),
        }
        return JSONResponse(status_code=200, content={"status": "ok", "detail": status})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/infer_frame")
async def infer_frame(image: UploadFile = File(...)):
    """Recebe um JPEG/PNG enviado pelo app e retorna predições.
    Espera um campo multipart 'image'. Retorna JSON com postura e detecção de face.
    """
    contents = await image.read()
    arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Não foi possível decodificar a imagem")

    mon = get_monitor()

    # run MediaPipe face/pose and classification under a lock to avoid race conditions
    with _monitor_lock:
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_results = mon.face_mesh.process(rgb)
            pose_results = mon.pose.process(rgb)

            face_detected = bool(face_results and getattr(face_results, 'multi_face_landmarks', None))

            posture_class = None
            posture_confidence = None
            posture_label = None
            posture_prediction = None

            if mon.posture_model is not None and pose_results and getattr(pose_results, 'pose_landmarks', None):
                cls, conf = mon._classificar_postura(frame, pose_results)
                posture_class = int(cls) if isinstance(cls, (int, np.integer)) else cls
                posture_confidence = float(conf) if conf is not None else None
                posture_label = getattr(mon, 'last_posture_label', None)
                posture_prediction = getattr(mon, 'last_posture_prediction', None)

            # Optionally return facial features (small dict) to allow client-side logic
            facial_features = None
            if face_detected:
                try:
                    landmarks = face_results.multi_face_landmarks[0].landmark
                    vals, ypr = mon._extract_facial_features(landmarks, frame.shape)
                    facial_features = {k: float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else None for k, v in vals.items()}
                except Exception:
                    facial_features = None

            res = {
                "face_detected": bool(face_detected),
                "posture_class": posture_class,
                "posture_confidence": posture_confidence,
                "posture_label": posture_label,
                "posture_prediction": posture_prediction,
                "facial_features": facial_features,
                "timestamp": time.time()
            }
            return JSONResponse(status_code=200, content=res)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
