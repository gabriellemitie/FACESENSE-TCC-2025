import cv2
import mediapipe as mp
import numpy as np
import time
import csv
import os
from datetime import datetime
from collections import deque


class PostureMonitor:
    """
    Classe modular para monitoramento postural com MediaPipe.
    Adaptada para integração com PySide6 (interface gráfica).
    Foca exclusivamente em alinhamento de tronco e ombros.
    """

    def __init__(
        self,
        alerta_segundos=2,
        lembrete_alongar_min=45,
        historico_path="historico_postura.csv",
        beep_callback=None,
        update_callback=None,
        debounce_frames=3
    ):
        # ===== Configurações =====
        self.ALERTA_SEGUNDOS_POSTURA = alerta_segundos
        self.VOLUME = 0.8
        self.LEMBRETE_ALONGAR_MIN = lembrete_alongar_min
        self.CAMINHO_HISTORICO = historico_path
        self.beep_callback = beep_callback
        self.update_callback = update_callback

        # ===== MediaPipe =====
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)

        # ===== Estado =====
        self.cap = None
        self.flip_enabled = True
        self.fila_postura = deque(maxlen=45)
        self.shoulder_ref = 0.015
        self.angulo_ref = 3.0
        self.calibrado = False
        self.running = False
        self._last_debug_log = 0.0

        # ===== Debounce =====
        self.DEBOUNCE_FRAMES = int(debounce_frames) if debounce_frames and debounce_frames > 0 else 3
        self._status = "indefinido"
        self._debounce_counter = 0

        self._iniciar_csv()

    # ----------------------------------------------------------------
    def _play_tone(self, freq=1000, dur=300, volume=1.0):
        """Gera um beep simples usando simpleaudio, se disponível."""
        try:
            import simpleaudio as sa
            fs = 44100
            t = np.linspace(0, dur / 1000, int(fs * dur / 1000), False)
            tone = np.sin(freq * t * 2 * np.pi)
            audio = (tone * (32767 * volume)).astype(np.int16)
            play_obj = sa.play_buffer(audio, 1, 2, fs)
            play_obj.wait_done()
        except Exception:
            print(f"[PostureMonitor] Beep: freq={freq}, dur={dur}")

    def _beep_postura(self):
        """Som de alerta de má postura"""
        self._play_tone(1200, 300, self.VOLUME)

    def _beep_alongamento(self):
        """Som duplo para lembrete de alongamento"""
        self._play_tone(600, 250, self.VOLUME)
        time.sleep(0.15)
        self._play_tone(1000, 250, self.VOLUME)

    # ----------------------------------------------------------------
    def _iniciar_csv(self):
        """Cria cabeçalho do histórico se não existir"""
        if not os.path.exists(self.CAMINHO_HISTORICO):
            with open(self.CAMINHO_HISTORICO, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "timestamp", "status_postura", "angulo_coluna_deg",
                    "shoulder_diff", "duracao_postura_ruim_s", "min_desde_ultimo_alongar"
                ])

    def _salvar_historico(self, status, angulo_coluna, shoulder_diff, dur_ruim, min_desde_alongar):
        """Salva métricas da sessão"""
        with open(self.CAMINHO_HISTORICO, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                status, f"{angulo_coluna:.2f}",
                f"{shoulder_diff:.4f}", f"{dur_ruim:.1f}", f"{min_desde_alongar:.1f}"
            ])

    # ----------------------------------------------------------------
    def calibrar(self, duracao=5):
        """Calibra posição neutra dos ombros e coluna."""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Não foi possível acessar a câmera!")

        print("🧍 Mantenha boa postura por alguns segundos para calibrar...")
        shoulder_list, angulo_list = [], []
        t0 = time.time()

        while time.time() - t0 < duracao:
            ok, frame = self.cap.read()
            if not ok:
                continue

            if self.flip_enabled:
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.pose.process(rgb)
            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                l_sh = [lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                r_sh = [lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                        lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                l_hip = [lm[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
                         lm[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
                r_hip = [lm[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                         lm[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y]

                mid_sh = np.mean([l_sh, r_sh], axis=0)
                mid_hip = np.mean([l_hip, r_hip], axis=0)
                dx, dy = mid_hip[0] - mid_sh[0], mid_hip[1] - mid_sh[1]
                angulo = np.degrees(np.arctan2(abs(dx), abs(dy)))

                shoulder_list.append(abs(l_sh[1] - r_sh[1]))
                angulo_list.append(angulo)

        self.shoulder_ref = float(np.mean(shoulder_list)) if shoulder_list else 0.015
        self.angulo_ref = float(np.mean(angulo_list)) if angulo_list else 3.0
        self.calibrado = True
        print(f"✅ Calibrado | shoulder_ref={self.shoulder_ref:.3f}, ang_ref={self.angulo_ref:.2f}")

    # ----------------------------------------------------------------
    def iniciar(self):
        """Inicia o monitoramento postural."""
        if not self.calibrado:
            self.calibrar()

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Não foi possível acessar a câmera!")

        self.running = True
        self.inicio_ruim = None
        self.ultimo_alerta_postura = 0
        self.ultimo_alongamento = time.time()

    def parar(self):
        """Encerra o monitoramento."""
        self.running = False
        if self.cap:
            self.cap.release()

    # ----------------------------------------------------------------
    def processar_frame(self):
        """Processa um frame e retorna imagem, status e cor."""
        ret, frame = self.cap.read()
        if not ret:
            return None, "indefinido", "Sem imagem", (255, 255, 255)

        if self.flip_enabled:
            frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)
        image = frame.copy()

        status = "indefinido"
        dica = ""
        cor = (255, 255, 255)
        dur_ruim = 0.0
        angulo_coluna = 0.0
        shoulder_diff = 0.0

        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            l_sh = [lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                    lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            r_sh = [lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                    lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            l_hip = [lm[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
                     lm[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
            r_hip = [lm[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                     lm[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y]

            mid_sh = np.mean([l_sh, r_sh], axis=0)
            mid_hip = np.mean([l_hip, r_hip], axis=0)
            dx, dy = mid_hip[0] - mid_sh[0], mid_hip[1] - mid_sh[1]
            angulo_coluna = np.degrees(np.arctan2(abs(dx), abs(dy)))
            shoulder_diff = abs(l_sh[1] - r_sh[1])

            ruim_postura = (angulo_coluna > 6 or shoulder_diff > self.shoulder_ref + 0.02)
            self.fila_postura.append("ruim" if ruim_postura else "boa")

            proporcao_ruim = self.fila_postura.count("ruim") / len(self.fila_postura)
            postura_final = proporcao_ruim > 0.6

            if postura_final:
                if self.inicio_ruim is None:
                    self.inicio_ruim = time.time()
                dur_ruim = time.time() - self.inicio_ruim
                if dur_ruim >= self.ALERTA_SEGUNDOS_POSTURA and \
                        time.time() - self.ultimo_alerta_postura > self.ALERTA_SEGUNDOS_POSTURA:
                    self._beep_postura()
                    self.ultimo_alerta_postura = time.time()
                dica = "Ajuste sua postura."
                cor = (0, 0, 255)
                status = "ruim"
            else:
                self.inicio_ruim = None
                dica = "Postura boa!"
                cor = (0, 255, 0)
                status = "boa"

            self.mp_drawing.draw_landmarks(image, res.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        # Lembrete de alongamento
        min_desde_alongar = (time.time() - self.ultimo_alongamento) / 60.0
        if min_desde_alongar >= self.LEMBRETE_ALONGAR_MIN:
            self._beep_alongamento()
            self.ultimo_alongamento = time.time()

        if self.update_callback:
            self.update_callback(status, dica)

        self._salvar_historico(status, angulo_coluna, shoulder_diff, dur_ruim, min_desde_alongar)
        return image, status, dica, cor
