import cv2
import mediapipe as mp
import numpy as np
import time
import csv
import os
import platform
from datetime import datetime
from collections import deque


class PostureMonitor:
    """
    Classe modular para monitoramento postural com MediaPipe.
    Adaptada para integração com PySide6 (interface gráfica).
    """

    def __init__(
        self,
        alerta_segundos=5,
        lembrete_alongar_min=45,
        angulo_cabeca_limite=18,
        historico_path="historico_postura.csv",
        beep_callback=None,
        update_callback=None
    ):
        # ===== Configurações =====
        self.ALERTA_SEGUNDOS_POSTURA = alerta_segundos
        self.VOLUME = 0.8  # volume padrão (80%)
        self.LEMBRETE_ALONGAR_MIN = lembrete_alongar_min
        self.ANGULO_CABECA_LIMITE = angulo_cabeca_limite
        self.CAMINHO_HISTORICO = historico_path
        self.beep_callback = beep_callback
        self.update_callback = update_callback  # usado para atualizar a UI

        # ===== MediaPipe =====
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)

        # ===== Estado =====
        self.cap = None
        self.flip_enabled = True
        self.fila_postura = deque(maxlen=45)
        self.pitch_media = deque(maxlen=30)
        self.shoulder_ref = 0.015
        self.angulo_ref = 3.0
        self.calibrado = False
        self.running = False

        # ===== Som =====
    def _play_tone(self, freq=1000, dur=300, volume=1.0):
        fs = 44100  # taxa de amostragem
        t = np.linspace(0, dur / 1000, int(fs * dur / 1000), False)
        tone = np.sin(freq * t * 2 * np.pi)
        audio = (tone * (32767 * volume)).astype(np.int16)
        play_obj = sa.play_buffer(audio, 1, 2, fs)
        play_obj.wait_done()

    def _beep_postura(self):
        """Som para alerta de postura"""
        self._play_tone(1200, 300, self.VOLUME)

    def _beep_alongamento(self):
        """Som duplo para lembrete de alongamento"""
        self._play_tone(600, 250, self.VOLUME)
        time.sleep(0.15)
        self._play_tone(1000, 250, self.VOLUME)

        # ===== Histórico =====
        self._iniciar_csv()

    # ----------------------------------------------------------------
    def _iniciar_csv(self):
        """Cria cabeçalho do histórico se ainda não existir"""
        if not os.path.exists(self.CAMINHO_HISTORICO):
            with open(self.CAMINHO_HISTORICO, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "timestamp", "status_postura", "angulo_coluna_deg", "pitch_cabeca_deg",
                    "shoulder_diff", "duracao_postura_ruim_s", "min_desde_ultimo_alongar"
                ])

    def _salvar_historico(self, status, angulo_coluna, pitch_cabeca, shoulder_diff, dur_ruim, min_desde_alongar):
        with open(self.CAMINHO_HISTORICO, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"), status,
                f"{angulo_coluna:.2f}", f"{pitch_cabeca:.2f}",
                f"{shoulder_diff:.4f}", f"{dur_ruim:.1f}", f"{min_desde_alongar:.1f}"
            ])

    # ----------------------------------------------------------------
    def calibrar(self, duracao=5):
        """Calibra ombros e ângulo neutro de postura."""
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
                l_sh = [lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                r_sh = [lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                l_hip = [lm[self.mp_pose.PoseLandmark.LEFT_HIP.value].x, lm[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
                r_hip = [lm[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x, lm[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y]

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
        self.inicio_cabeca_caida = None
        self.ultimo_alerta_cabeca = 0

    def parar(self):
        """Encerra o monitoramento."""
        self.running = False
        if self.cap:
            self.cap.release()

    # ----------------------------------------------------------------
    def processar_frame(self):
        # ===== LEITURA DO FRAME =====
        ret, frame = self.cap.read()
        if not ret:
            return None, "indefinido", "Sem imagem", (255, 255, 255)

        # Espelhar imagem (se habilitado)
        if getattr(self, "flip_enabled", True):
            frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)
        image = frame.copy()

        # ===== INICIALIZAÇÃO PADRÃO =====
        status = "indefinido"
        dica = ""
        cor = (255, 255, 255)
        dur_ruim = 0.0
        angulo_coluna = 0.0
        shoulder_diff = 0.0
        pitch_cabeca = 0.0
        postura_final = False

        # ===== DETECÇÃO DE POSTURA =====
        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            l_sh = [lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            r_sh = [lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            l_hip = [lm[self.mp_pose.PoseLandmark.LEFT_HIP.value].x, lm[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
            r_hip = [lm[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x, lm[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            nose  = [lm[self.mp_pose.PoseLandmark.NOSE.value].x, lm[self.mp_pose.PoseLandmark.NOSE.value].y]
            l_ear = [lm[self.mp_pose.PoseLandmark.LEFT_EAR.value].x, lm[self.mp_pose.PoseLandmark.LEFT_EAR.value].y]
            r_ear = [lm[self.mp_pose.PoseLandmark.RIGHT_EAR.value].x, lm[self.mp_pose.PoseLandmark.RIGHT_EAR.value].y]

            mid_sh  = np.mean([l_sh, r_sh], axis=0)
            mid_hip = np.mean([l_hip, r_hip], axis=0)
            mid_ear = np.mean([l_ear, r_ear], axis=0)

            dx = mid_hip[0] - mid_sh[0]
            dy = mid_hip[1] - mid_sh[1]
            angulo_coluna = np.degrees(np.arctan2(abs(dx), abs(dy)))
            shoulder_diff = abs(l_sh[1] - r_sh[1])
            pitch_cabeca_instant = np.degrees(np.arctan2(mid_ear[1] - nose[1], abs(mid_ear[0] - nose[0])))

            self.pitch_media.append(pitch_cabeca_instant)
            pitch_cabeca = np.mean(self.pitch_media)

            ruim_postura = (angulo_coluna > 6 or shoulder_diff > self.shoulder_ref + 0.02)

            # --- Cabeça caída por tempo prolongado ---
            if pitch_cabeca > self.ANGULO_CABECA_LIMITE:
                if self.inicio_cabeca_caida is None:
                    self.inicio_cabeca_caida = time.time()
                dur_cabeca = time.time() - self.inicio_cabeca_caida
            else:
                dur_cabeca = 0
                self.inicio_cabeca_caida = None

            cabeca_caida_longa = dur_cabeca >= 12
            ruim_instant = ruim_postura or cabeca_caida_longa

            self.fila_postura.append("ruim" if ruim_instant else "boa")
            proporcao_ruim = self.fila_postura.count("ruim") / len(self.fila_postura) if self.fila_postura else 0
            postura_final = proporcao_ruim > 0.6

            # ===== LÓGICA DE ALERTAS =====
            if postura_final:
                if self.inicio_ruim is None:
                    self.inicio_ruim = time.time()
                dur_ruim = time.time() - self.inicio_ruim

                if dur_ruim >= self.ALERTA_SEGUNDOS_POSTURA and time.time() - self.ultimo_alerta_postura > self.ALERTA_SEGUNDOS_POSTURA:
                    self._beep_postura()
                    self.ultimo_alerta_postura = time.time()
            else:
                self.inicio_ruim = None

            # ===== DICAS E STATUS =====
            if postura_final:
                dica = "Levante o queixo!" if cabeca_caida_longa else "Ajuste sua postura."
                cor = (0, 0, 255)
                status = "ruim"
            else:
                dica = "Postura boa!"
                cor = (0, 255, 0)
                status = "boa"

            self.mp_drawing.draw_landmarks(image, res.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        else:
            # Se não detectou landmarks
            cv2.putText(image, "Nenhuma pessoa detectada", (40, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (200, 200, 200), 3)

        # ===== LEMBRETE DE ALONGAMENTO ====
        min_desde_alongar = (time.time() - self.ultimo_alongamento) / 60.0
        if min_desde_alongar >= self.LEMBRETE_ALONGAR_MIN:
            self._beep_alongamento()
            self.ultimo_alongamento = time.time()


        # ===== CALLBACK DE UI =====
        if self.update_callback:
            self.update_callback(status, dica)

        # ===== SALVA HISTÓRICO =====
        self._salvar_historico(status, angulo_coluna, pitch_cabeca, shoulder_diff, dur_ruim, min_desde_alongar)

        return image, status, dica, cor


    # ----------------------------------------------------------------
    def _emitir_alerta(self, cabeca_caida=False):
        """Emite alerta sonoro e visual"""
        if self.beep_callback:
            self.beep_callback()
        else:
            self._beep(1000, 600)
        if cabeca_caida:
            self._beep(800, 300)
