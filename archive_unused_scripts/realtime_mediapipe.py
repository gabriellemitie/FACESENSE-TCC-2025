# app_realtime_mediapipe_minute.py — Webcam + MediaPipe + modelo
# - 1 classificação por minuto (últimos 60s)
# - Landmarks desenhados + eixos de cabeça
# - Threshold com "shift" POSITIVO (mais difícil dar estresse)
# - Atalhos: [ / ] ajustam threshold_shift; r recalibra baseline; i alterna debug

import time
import json
from pathlib import Path
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib

# ===== CONFIG =====
MODEL_PATH = Path("facesense_posture/models/model_face_temporal.pkl")
FEATS_JSON = Path("facesense_posture/models/features.json")

WINDOW_SEC = 600            # buffer para baseline (10 min)
BUCKET_SEC = 60             # janela de classificação = 60 s (1/min)
BASELINE_WARMUP_SEC = 120   # 2 min para travar baseline
CAM_INDEX = 0

# Tornar mais difícil classificar como estresse:
#   threshold_ajustado = threshold_base + THRESHOLD_SHIFT  (SHIFT > 0 => fica mais difícil dar "1")
THRESHOLD_SHIFT = 0.08      # ajuste fino ao seu gosto (0.05 ~ 0.12)

# Visual
DRAW_DEBUG = True           # pode alternar com tecla 'i'
CIRCLE_R = 2

ALIASES = [
    "SleftEyeClosed", "SrightEyeClosed", "SAu43_EyesClosed",
    "SmouthOpen", "SAu25_LipsPart", "SAu26_JawDrop", "SAu27_MouthStretch",
    "HeadYaw", "HeadPitch", "HeadRoll"
]

# ===== util geom =====
def _dist(a, b): return float(np.linalg.norm(np.array(a) - np.array(b)))
def _ear(eye_top, eye_bottom, eye_left, eye_right):
    open_v = _dist(eye_top, eye_bottom); width = _dist(eye_left, eye_right) + 1e-9; return open_v/width
def _mar(mouth_top, mouth_bottom, mouth_left, mouth_right):
    open_v = _dist(mouth_top, mouth_bottom); width = _dist(mouth_left, mouth_right) + 1e-9; return open_v/width
def _jaw_drop(chin, upper_lip, ref_width): return _dist(chin, upper_lip)/(ref_width+1e-9)
def _mouth_stretch(mouth_left, mouth_right, ref_width): return _dist(mouth_left, mouth_right)/(ref_width+1e-9)

# ===== head pose (yaw/pitch/roll) via solvePnP =====
MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0), (0.0, -63.6, -12.5),
    (-43.3, 32.7, -26.0), (43.3, 32.7, -26.0),
    (-28.9, -28.9, -24.1), (28.9, -28.9, -24.1)
], dtype=np.float32)

IDX = {
    "nose_tip": 1, "chin": 152,
    "left_eye_outer": 33, "right_eye_outer": 263,
    "mouth_left": 61, "mouth_right": 291,
    "mouth_top": 13, "mouth_bottom": 14,
    "eye_left_top": 159, "eye_left_bottom": 145, "eye_left_inner": 133, "eye_left_outer": 33,
    "eye_right_top": 386, "eye_right_bottom": 374, "eye_right_inner": 362, "eye_right_outer": 263
}

def head_pose_ypr(image_shape, lm):
    h, w = image_shape[:2]
    pts2d = np.array([
        [lm[IDX["nose_tip"]][0]*w, lm[IDX["nose_tip"]][1]*h],
        [lm[IDX["chin"]][0]*w, lm[IDX["chin"]][1]*h],
        [lm[IDX["left_eye_outer"]][0]*w, lm[IDX["left_eye_outer"]][1]*h],
        [lm[IDX["right_eye_outer"]][0]*w, lm[IDX["right_eye_outer"]][1]*h],
        [lm[IDX["mouth_left"]][0]*w, lm[IDX["mouth_left"]][1]*h],
        [lm[IDX["mouth_right"]][0]*w, lm[IDX["mouth_right"]][1]*h],
    ], dtype=np.float32)
    f = w
    K = np.array([[f,0,w/2],[0,f,h/2],[0,0,1]], dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(MODEL_POINTS_3D, pts2d, K, np.zeros((4,1)), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok: return (np.nan, np.nan, np.nan), None, None
    R, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2); singular = sy < 1e-6
    if not singular:
        pitch = np.arctan2(-R[2,0], sy); yaw = np.arctan2(R[1,0], R[0,0]); roll = np.arctan2(R[2,1], R[2,2])
    else:
        pitch = np.arctan2(-R[2,0], sy); yaw = np.arctan2(-R[0,1], R[1,1]); roll = 0
    ypr = (np.degrees(yaw), np.degrees(pitch), np.degrees(roll))
    return ypr, rvec, tvec

def draw_head_axes(frame, K, rvec, tvec, length=80):
    if rvec is None or tvec is None: return
    # eixos 3D
    axes_3d = np.float32([[length,0,0],[0,length,0],[0,0,length],[0,0,0]])
    dist = np.zeros((4,1))
    pts2d, _ = cv2.projectPoints(axes_3d, rvec, tvec, K, dist)
    pt_o = tuple(pts2d[3].ravel().astype(int))
    pt_x = tuple(pts2d[0].ravel().astype(int))
    pt_y = tuple(pts2d[1].ravel().astype(int))
    pt_z = tuple(pts2d[2].ravel().astype(int))
    cv2.line(frame, pt_o, pt_x, (0,0,255), 2)   # X (vermelho)
    cv2.line(frame, pt_o, pt_y, (0,255,0), 2)   # Y (verde)
    cv2.line(frame, pt_o, pt_z, (255,0,0), 2)   # Z (azul)

# ===== buffer/feature builder =====
class MinuteAggregator:
    def __init__(self, window_sec=WINDOW_SEC, bucket_sec=BUCKET_SEC, warmup_sec=BASELINE_WARMUP_SEC):
        self.window_sec = window_sec
        self.bucket_sec = bucket_sec
        self.warmup_sec = warmup_sec
        self.ts = deque()
        self.data = {a: deque() for a in ALIASES}
        self.mu = {a: None for a in ALIASES}
        self.sd = {a: None for a in ALIASES}
        self.locked = False
        self.last_bucket_idx = None

    def append(self, t, vals):
        self.ts.append(t)
        for a in ALIASES:
            v = vals.get(a, None)
            if v is not None and np.isfinite(v):
                self.data[a].append((t, float(v)))
        self._trim(t)
        self._update_baseline()

    def _trim(self, now):
        cutoff = now - self.window_sec
        while self.ts and self.ts[0] < cutoff:
            self.ts.popleft()
        for a in ALIASES:
            dq = self.data[a]
            while dq and dq[0][0] < cutoff:
                dq.popleft()

    def _update_baseline(self):
        if self.locked or not self.ts: return
        span = self.ts[-1] - self.ts[0]
        for a in ALIASES:
            arr = np.array([v for _, v in self.data[a]], float)
            if arr.size >= 5:
                mu = float(np.nanmean(arr))
                s = float(np.nanstd(arr, ddof=1))
                self.mu[a] = mu
                self.sd[a] = s if (np.isfinite(s) and s > 1e-8) else np.nan
        if span >= self.warmup_sec:
            self.locked = True

    def _window_df(self, t_end, span):
        frames = []
        for a in ALIASES:
            if self.data[a]:
                tt = [ti for ti,_ in self.data[a]]
                vv = [vi for _,vi in self.data[a]]
                frames.append(pd.DataFrame({"t":tt, a:vv}).set_index("t"))
        if not frames: return pd.DataFrame()
        df = pd.concat(frames, axis=1, join="outer").sort_index()
        t0 = t_end - span
        return df.loc[df.index.to_series().between(t0, t_end)]

    def _zscore(self, df):
        z = pd.DataFrame(index=df.index)
        for a in ALIASES:
            if a in df.columns:
                mu = self.mu.get(a, None); sd = self.sd.get(a, None)
                col = df[a].astype(float)
                if mu is not None and sd is not None and np.isfinite(sd) and sd>1e-8:
                    z[a+"_z"] = (col - mu)/sd
                else:
                    z[a+"_z"] = col
        return z

    @staticmethod
    def _p90(x):
        return float(np.nanpercentile(x, 90)) if (len(x)>0 and np.isfinite(x).any()) else np.nan
    @staticmethod
    def _jitter(x):
        if len(x)<=1: return np.nan
        return float(np.nansum(np.abs(np.diff(x))))

    def build_row_last_minute(self, feature_names, now_ts):
        df = self._window_df(now_ts, BUCKET_SEC)
        if df.empty: return None
        z = self._zscore(df)
        agg = {}
        for a in ALIASES:
            col = a+"_z"
            if col in z.columns:
                v = z[col].dropna().values.astype(float)
                agg[f"{col}_mean"] = float(np.nanmean(v)) if v.size else np.nan
                agg[f"{col}_p90"]  = self._p90(v)
                agg[f"{col}_jitter"] = self._jitter(v)
        row = {name: agg.get(name, np.nan) for name in feature_names}
        return row

    def is_new_minute(self, now_ts):
        idx = int(now_ts // self.bucket_sec)
        if self.last_bucket_idx is None:
            self.last_bucket_idx = idx
            return False
        if idx != self.last_bucket_idx:
            self.last_bucket_idx = idx
            return True
        return False

# ===== modelo =====
def load_model_and_meta(threshold_shift=THRESHOLD_SHIFT):
    if not MODEL_PATH.exists(): raise FileNotFoundError(f"Modelo não encontrado: {MODEL_PATH}")
    if not FEATS_JSON.exists(): raise FileNotFoundError(f"features.json não encontrado: {FEATS_JSON}")
    pipe = joblib.load(MODEL_PATH)
    meta = json.loads(FEATS_JSON.read_text(encoding="utf-8"))
    feats = meta["features"]
    thr_base = float(meta.get("threshold", 0.5))
    thr_adj = max(0.01, min(0.99, thr_base + float(threshold_shift)))
    return pipe, feats, thr_base, thr_adj

# ===== desenho de landmarks =====
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

def draw_landmarks(frame, face_landmarks, draw_all=True, color=(0,255,0)):
    """Desenha apenas pontos essenciais (não a malha inteira)."""
    if face_landmarks is None:
        return
    h, w = frame.shape[:2]
    important = [
        "nose_tip", "chin",
        "mouth_left", "mouth_right", "mouth_top", "mouth_bottom",
        "eye_left_outer", "eye_left_inner", "eye_left_top", "eye_left_bottom",
        "eye_right_outer", "eye_right_inner", "eye_right_top", "eye_right_bottom"
    ]
    for name in important:
        idx = IDX.get(name)
        if idx is None: 
            continue
        lm = face_landmarks.landmark[idx]
        x = int(lm.x * w)
        y = int(lm.y * h)
        cv2.circle(frame, (x, y), CIRCLE_R, color, -1)

def put_kp(frame, xy, color=(0,255,0), r=CIRCLE_R):
    h, w = frame.shape[:2]
    cv2.circle(frame, (int(xy[0]*w), int(xy[1]*h)), r, color, -1)

# Desativa desenho dos eixos (plano cartesiano vetorial)
def draw_head_axes(frame, K, rvec, tvec, length=80):
    return

# ===== main loop =====
def main():
    global DRAW_DEBUG
    pipe, feature_names, thr_base, thr_adj = load_model_and_meta()

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.6, min_tracking_confidence=0.6
    )

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("Não consegui abrir a câmera."); return

    # reduz resolução se quiser mais FPS (descomente)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    agg = MinuteAggregator(WINDOW_SEC, BUCKET_SEC, BASELINE_WARMUP_SEC)
    last_proba = None
    threshold_shift = THRESHOLD_SHIFT

    try:
        while True:
            ok, frame = cap.read()
            if not ok: break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)
            h, w = frame.shape[:2]

            vals = {a: np.nan for a in ALIASES}
            ypr, rvec, tvec = (np.nan, np.nan, np.nan), None, None

            if res.multi_face_landmarks:
                fl = res.multi_face_landmarks[0]
                lm = fl.landmark

                # desenha malha/contornos
                if DRAW_DEBUG:
                    draw_landmarks(frame, fl, draw_all=True)

                def P(i): li = lm[i]; return (li.x, li.y)

                # pontos chave e marcadores
                mouth_top, mouth_bottom = P(IDX["mouth_top"]), P(IDX["mouth_bottom"])
                mouth_left, mouth_right = P(IDX["mouth_left"]), P(IDX["mouth_right"])
                chin = P(IDX["chin"])
                le_top, le_bottom, le_outer, le_inner = P(IDX["eye_left_top"]), P(IDX["eye_left_bottom"]), P(IDX["eye_left_outer"]), P(IDX["eye_left_inner"])
                re_top, re_bottom, re_outer, re_inner = P(IDX["eye_right_top"]), P(IDX["eye_right_bottom"]), P(IDX["eye_right_outer"]), P(IDX["eye_right_inner"])

                if DRAW_DEBUG:
                    for pt in [mouth_top, mouth_bottom, mouth_left, mouth_right, chin,
                               le_top, le_bottom, le_outer, le_inner, re_top, re_bottom, re_outer, re_inner]:
                        put_kp(frame, pt, (0,255,0), CIRCLE_R)

                inter_pupil = _dist(le_inner, re_inner) + 1e-9
                mar_v = _mar(mouth_top, mouth_bottom, mouth_left, mouth_right)
                vals["SmouthOpen"] = mar_v
                vals["SAu25_LipsPart"] = mar_v
                vals["SAu26_JawDrop"] = _jaw_drop(chin, mouth_top, inter_pupil)
                vals["SAu27_MouthStretch"] = _mouth_stretch(mouth_left, mouth_right, inter_pupil)

                le_ear = _ear(le_top, le_bottom, le_outer, le_inner)
                re_ear = _ear(re_top, re_bottom, re_inner, re_outer)
                vals["SleftEyeClosed"]  = 1.0 - le_ear
                vals["SrightEyeClosed"] = 1.0 - re_ear
                vals["SAu43_EyesClosed"] = (vals["SleftEyeClosed"] + vals["SrightEyeClosed"]) / 2.0

                # head pose + eixos
                lm_xy = [(li.x, li.y) for li in lm]
                ypr, rvec, tvec = head_pose_ypr(frame.shape, lm_xy)
                yaw, pitch, roll = ypr
                vals["HeadYaw"] = yaw/45.0; vals["HeadPitch"] = pitch/45.0; vals["HeadRoll"] = roll/45.0

                if DRAW_DEBUG and rvec is not None:
                    # intrínseca para projetar eixos
                    f = w
                    K = np.array([[f,0,w/2],[0,f,h/2],[0,0,1]], dtype=np.float64)
                    draw_head_axes(frame, K, rvec, tvec, length=int(0.15*w))

            else:
                cv2.putText(frame, "Sem face", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80,80,255), 2)

            now = time.time()
            agg.append(now, vals)

            # fecha o minuto e classifica
            if agg.is_new_minute(now) and agg.locked:
                row = agg.build_row_last_minute(feature_names, now)
                if row is not None:
                    X = pd.DataFrame([row], columns=feature_names)
                    proba = float(pipe.predict_proba(X)[:,1][0])
                    thr_adj = max(0.01, min(0.99, thr_base + threshold_shift))
                    label = int(proba >= thr_adj)
                    last_proba = proba
                    nivel = "ALTO" if proba>=0.65 else ("MÉDIO" if proba>=0.45 else "BAIXO")
                    print(f"[MIN] proba={proba:.3f} thr_base={thr_base:.3f} thr_adj={thr_adj:.3f} (shift={threshold_shift:+.2f}) → label={label} ({nivel}) | baseline_locked={agg.locked}")

            # overlay principal
            if not agg.locked:
                cv2.putText(frame, "Calibrando baseline...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
                span = (agg.ts[-1]-agg.ts[0]) if agg.ts else 0
                cv2.putText(frame, f"{int(span)}/{BASELINE_WARMUP_SEC}s", (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
            else:
                secs_to_next = BUCKET_SEC - int(time.time()) % BUCKET_SEC
                if last_proba is not None:
                    nivel = "ALTO" if last_proba>=0.65 else ("MÉDIO" if last_proba>=0.45 else "BAIXO")
                    color = (0,200,0) if nivel=="BAIXO" else ((0,165,255) if nivel=="MÉDIO" else (0,0,255))
                    cv2.rectangle(frame, (10,10), (410,105), (20,20,20), -1)
                    cv2.putText(frame, f"Stress (min): {nivel}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    cv2.putText(frame, f"Proba: {last_proba:.3f} | thr_adj {max(0.01, min(0.99, thr_base + threshold_shift)):.3f}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                else:
                    cv2.putText(frame, "Aguardando 1º minuto fechar...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                cv2.putText(frame, f"T-{secs_to_next:02d}s", (330, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2)

            # mini HUD com YPR
            if np.isfinite(ypr[0]):
                cv2.putText(frame, f"Y:{ypr[0]:+.1f}  P:{ypr[1]:+.1f}  R:{ypr[2]:+.1f}", (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,180), 2)

            cv2.imshow("Stress (1-min) — Face+Head", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break
            elif key == ord('r'):
                # Recalibrar baseline em runtime
                agg.mu = {a: None for a in ALIASES}
                agg.sd = {a: None for a in ALIASES}
                agg.locked = False
                print("[info] baseline resetado; recalibrando por ~2 min.")
            elif key == ord('['):
                threshold_shift = max(-0.40, threshold_shift - 0.02)
                print(f"[info] threshold_shift = {threshold_shift:+.2f}")
            elif key == ord(']'):
                threshold_shift = min(+0.40, threshold_shift + 0.02)
                print(f"[info] threshold_shift = {threshold_shift:+.2f}")
            elif key == ord('i'):
                DRAW_DEBUG = not DRAW_DEBUG
                print(f"[info] DRAW_DEBUG = {DRAW_DEBUG}")

    finally:
        cap.release()
        face_mesh.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
