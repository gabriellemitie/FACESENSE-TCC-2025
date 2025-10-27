"""
Módulo integrado: Classificador de postura + Microgesture
Combina modelo Keras de postura com modelo de microgesture facial
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
import json
from pathlib import Path
from collections import deque
from tensorflow import keras
import joblib


class IntegratedClassifier:
    """
    Classificador integrado que combina:
    1. Modelo Keras para classificação de postura corporal (29 classes)
    2. Modelo de microgesture facial para detecção de estresse
    """

    def __init__(
        self,
        posture_model_path="facesense_posture/models/stress.keras",
        microgesture_model_path="facesense_posture/models/model_face_temporal.pkl",
        features_json_path="facesense_posture/models/features.json",
        beep_callback=None,
        update_callback=None,
        alerta_segundos=5,
        lembrete_alongar_min=45
    ):
        # Configurações gerais
        self.beep_callback = beep_callback
        self.update_callback = update_callback
        self.ALERTA_SEGUNDOS_POSTURA = alerta_segundos
        self.LEMBRETE_ALONGAR_MIN = lembrete_alongar_min
        self.VOLUME = 0.8
        
        # MediaPipe para postura e face
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        
        # Modelos
        self.posture_model = None
        self.microgesture_model = None
        self.feature_names = []
        
        self._carregar_modelos(posture_model_path, microgesture_model_path, features_json_path)
        
        # Buffer temporal para postura
        self.buffer_landmarks = []
        self.max_frames = 500
        self.min_frames = 50
        
        # Filtro de landmarks
        self.filtro_landmarks = LandmarkFilter(suavizacao=0.8)
        
        # Microgesture aggregator
        self.microgesture_agg = MicrogestureAggregator()
        
        # Estado do classificador
        self.cap = None
        self.running = False
        self.ultima_predicao_postura = None
        self.ultima_predicao_microgesture = None
        self.confianca_postura = 0.0
        self.classe_postura = -1
        self.nivel_estresse = "BAIXO"
        self.proba_estresse = 0.0
        
        # Controle de alertas
        self.ultimo_alerta = 0
        self.contador_postura_ruim = 0
        self.inicio_postura_ruim = None

    def _carregar_modelos(self, posture_path, microgesture_path, features_path):
        """Carrega ambos os modelos"""
        # Modelo de postura
        try:
            self.posture_model = keras.models.load_model(posture_path)
            print(f"✅ Modelo de postura carregado: {self.posture_model.output_shape[1]} classes")
        except Exception as e:
            print(f"❌ Erro ao carregar modelo de postura: {e}")
            self.posture_model = None
        
        # Modelo de microgesture
        try:
            self.microgesture_model = joblib.load(microgesture_path)
            
            # Carrega features
            if Path(features_path).exists():
                with open(features_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                    self.feature_names = meta["features"]
                    self.threshold_base = float(meta.get("threshold", 0.5))
            
            print(f"✅ Modelo de microgesture carregado com {len(self.feature_names)} features")
        except Exception as e:
            print(f"❌ Erro ao carregar modelo de microgesture: {e}")
            self.microgesture_model = None

    def iniciar(self):
        """Inicia a captura de vídeo"""
        if self.posture_model is None or self.microgesture_model is None:
            raise RuntimeError("Um ou ambos os modelos não foram carregados!")
            
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Não foi possível acessar a câmera!")
            
        self.running = True
        self.buffer_landmarks = []
        self.microgesture_agg.reset()
        print("🎯 Classificador integrado iniciado!")

    def parar(self):
        """Para a captura de vídeo"""
        self.running = False
        if self.cap:
            self.cap.release()
        print("🛑 Classificador integrado parado.")

    def processar_frame(self):
        """
        Processa um frame e retorna: (frame, status, dica, cor)
        Combina informações de postura e microgesture
        """
        if not self.running or not self.cap:
            return None, "parado", "Sistema parado", (128, 128, 128)

        ret, frame = self.cap.read()
        if not ret:
            return None, "erro", "Erro na câmera", (255, 0, 0)

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Processa pose e face simultaneamente
        pose_results = self.pose.process(rgb)
        face_results = self.face_mesh.process(rgb)
        
        status_postura = "sem_dados"
        status_microgesture = "sem_dados"
        
        # === PROCESSAMENTO DE POSTURA ===
        if pose_results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
            )
            
            # Extrai e filtra landmarks de postura
            landmarks_raw, qualidades = self._extrair_landmarks_postura(pose_results.pose_landmarks)
            landmarks_filtrados = self.filtro_landmarks.filtrar(landmarks_raw, qualidades)
            
            # Adiciona ao buffer
            self.buffer_landmarks.append(landmarks_filtrados)
            if len(self.buffer_landmarks) > self.max_frames:
                self.buffer_landmarks.pop(0)
            
            # Indicadores visuais das mãos
            self._desenhar_indicadores_maos(frame, pose_results.pose_landmarks, w, h)
            
            # Classificação de postura
            if len(self.buffer_landmarks) >= self.min_frames:
                status_postura = self._classificar_postura()
        
        # === PROCESSAMENTO DE MICROGESTURE ===
        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]
            
            # Desenha pontos faciais principais
            self._desenhar_pontos_faciais(frame, face_landmarks, w, h)
            
            # Extrai features faciais
            microgesture_features = self._extrair_features_faciais(face_landmarks, frame.shape)
            
            # Adiciona ao aggregator
            now = time.time()
            self.microgesture_agg.append(now, microgesture_features)
            
            # Classificação de microgesture (a cada minuto)
            if self.microgesture_agg.is_new_minute(now) and self.microgesture_agg.locked:
                status_microgesture = self._classificar_microgesture(now)
        
        # === COMBINA RESULTADOS ===
        status_combinado, dica_combinada, cor_combinada = self._combinar_resultados(
            status_postura, status_microgesture
        )
        
        # Desenha interface combinada
        self._desenhar_interface_combinada(frame, w, h)
        
        return frame, status_combinado, dica_combinada, cor_combinada

    def _extrair_landmarks_postura(self, pose_landmarks):
        """Extrai landmarks de postura com análise de qualidade"""
        landmarks_raw = []
        qualidades = []
        
        for i, lm in enumerate(pose_landmarks.landmark):
            landmarks_raw.extend([lm.x, lm.y, lm.z])
            visibility = getattr(lm, 'visibility', 1.0)
            
            if i in [15, 16, 17, 18, 19, 20, 21, 22]:  # mãos
                if visibility < 0.6 or lm.z > 0.8:
                    qualidades.extend([0.2, 0.2, 0.2])
                else:
                    qualidades.extend([visibility, visibility, visibility])
            else:
                qualidades.extend([visibility, visibility, visibility])
        
        return landmarks_raw, qualidades

    def _desenhar_indicadores_maos(self, frame, pose_landmarks, w, h):
        """Desenha indicadores visuais para landmarks das mãos"""
        landmarks_maos = [15, 16, 17, 18, 19, 20, 21, 22]
        
        for i in landmarks_maos:
            lm = pose_landmarks.landmark[i]
            x, y = int(lm.x * w), int(lm.y * h)
            visibility = getattr(lm, 'visibility', 1.0)
            
            if 0 <= x < w and 0 <= y < h:
                if visibility < 0.6 or lm.z > 0.8:
                    cv2.circle(frame, (x, y), 6, (0, 0, 255), 2)
                else:
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

    def _desenhar_pontos_faciais(self, frame, face_landmarks, w, h):
        """Desenha pontos faciais principais"""
        # Índices dos pontos importantes
        important_points = [1, 152, 33, 263, 61, 291, 13, 14, 159, 145, 386, 374]
        
        for idx in important_points:
            if idx < len(face_landmarks.landmark):
                lm = face_landmarks.landmark[idx]
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 2, (255, 255, 0), -1)

    def _extrair_features_faciais(self, face_landmarks, image_shape):
        """Extrai features para o modelo de microgesture"""
        # Esta é uma versão simplificada - você pode expandir com todas as features
        # do modelo original de microgesture
        features = {}
        
        # Pontos básicos para cálculo de features
        lm = face_landmarks.landmark
        
        # Exemplo de algumas features básicas
        try:
            # Mouth features
            mouth_top = lm[13]
            mouth_bottom = lm[14]
            mouth_left = lm[61]
            mouth_right = lm[291]
            
            mouth_height = abs(mouth_top.y - mouth_bottom.y)
            mouth_width = abs(mouth_left.x - mouth_right.x)
            
            features['SmouthOpen'] = mouth_height / (mouth_width + 1e-9)
            features['SAu25_LipsPart'] = mouth_height / (mouth_width + 1e-9)
            
            # Eye features (simplified)
            left_eye_top = lm[159]
            left_eye_bottom = lm[145]
            right_eye_top = lm[386]
            right_eye_bottom = lm[374]
            
            left_eye_height = abs(left_eye_top.y - left_eye_bottom.y)
            right_eye_height = abs(right_eye_top.y - right_eye_bottom.y)
            
            features['SleftEyeClosed'] = 1.0 - (left_eye_height * 10)  # normalizado
            features['SrightEyeClosed'] = 1.0 - (right_eye_height * 10)
            features['SAu43_EyesClosed'] = (features['SleftEyeClosed'] + features['SrightEyeClosed']) / 2.0
            
            # Head pose (simplified)
            nose = lm[1]
            chin = lm[152]
            
            features['HeadPitch'] = (nose.y - chin.y) * 2  # normalizado
            features['HeadYaw'] = (nose.x - 0.5) * 2
            features['HeadRoll'] = 0.0  # placeholder
            
        except Exception as e:
            print(f"Erro ao extrair features faciais: {e}")
            # Features padrão em caso de erro
            for alias in ['SleftEyeClosed', 'SrightEyeClosed', 'SAu43_EyesClosed',
                         'SmouthOpen', 'SAu25_LipsPart', 'SAu26_JawDrop', 'SAu27_MouthStretch',
                         'HeadYaw', 'HeadPitch', 'HeadRoll']:
                features[alias] = 0.0
        
        return features

    def _classificar_postura(self):
        """Executa classificação de postura"""
        try:
            sequence = np.zeros((200, 99))
            frames_recentes = self.buffer_landmarks[-200:] if len(self.buffer_landmarks) >= 200 else self.buffer_landmarks
            
            for i, frame_coords in enumerate(frames_recentes):
                if i < 200:
                    sequence[i] = frame_coords
            
            pred = self.posture_model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
            self.classe_postura = np.argmax(pred)
            self.confianca_postura = np.max(pred)
            self.ultima_predicao_postura = pred
            
            return f"postura_classe_{self.classe_postura}"
            
        except Exception as e:
            print(f"Erro na classificação de postura: {e}")
            return "erro_postura"

    def _classificar_microgesture(self, now_ts):
        """Executa classificação de microgesture"""
        try:
            row = self.microgesture_agg.build_row_last_minute(self.feature_names, now_ts)
            if row is not None:
                X = pd.DataFrame([row], columns=self.feature_names)
                proba = float(self.microgesture_model.predict_proba(X)[:,1][0])
                
                self.proba_estresse = proba
                
                if proba >= 0.65:
                    self.nivel_estresse = "ALTO"
                elif proba >= 0.45:
                    self.nivel_estresse = "MÉDIO"
                else:
                    self.nivel_estresse = "BAIXO"
                
                return f"estresse_{self.nivel_estresse.lower()}"
        except Exception as e:
            print(f"Erro na classificação de microgesture: {e}")
            return "erro_microgesture"

    def _combinar_resultados(self, status_postura, status_microgesture):
        """Combina resultados de postura e microgesture"""
        # Status combinado
        if "erro" in status_postura or "erro" in status_microgesture:
            return "erro", "Erro no sistema", (255, 0, 0)
        
        if "sem_dados" in status_postura and "sem_dados" in status_microgesture:
            return "sem_dados", "Aguardando detecção", (128, 128, 128)
        
        # Dica combinada
        dica_parts = []
        if self.ultima_predicao_postura is not None:
            dica_parts.append(f"Postura: Classe {self.classe_postura} ({self.confianca_postura:.2%})")
        
        if self.nivel_estresse != "BAIXO":
            dica_parts.append(f"Estresse: {self.nivel_estresse} ({self.proba_estresse:.2%})")
        
        dica = " | ".join(dica_parts) if dica_parts else "Monitorando..."
        
        # Cor baseada no pior resultado
        if self.nivel_estresse == "ALTO" or self.confianca_postura < 0.4:
            cor = (0, 0, 255)  # Vermelho
        elif self.nivel_estresse == "MÉDIO":
            cor = (0, 165, 255)  # Laranja
        else:
            cor = (0, 255, 0)  # Verde
        
        return "monitorando", dica, cor

    def _desenhar_interface_combinada(self, frame, w, h):
        """Desenha interface combinada com ambas as informações"""
        # Fundo para informações
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (500, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Título
        cv2.putText(frame, "MONITORAMENTO INTEGRADO", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Informações de postura
        if self.ultima_predicao_postura is not None:
            cv2.putText(frame, f"POSTURA: Classe {self.classe_postura}", (20, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, f"Confianca: {self.confianca_postura:.2%}", (20, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Informações de microgesture
        if self.microgesture_agg.locked:
            cor_stress = (0, 200, 0) if self.nivel_estresse == "BAIXO" else \
                        (0, 165, 255) if self.nivel_estresse == "MÉDIO" else (0, 0, 255)
            
            cv2.putText(frame, f"ESTRESSE: {self.nivel_estresse}", (20, 115),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor_stress, 2)
            cv2.putText(frame, f"Probabilidade: {self.proba_estresse:.2%}", (20, 135),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            cv2.putText(frame, "MICROGESTURE: Calibrando...", (20, 115),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Status do buffer de postura
        if len(self.buffer_landmarks) < self.min_frames:
            progress = len(self.buffer_landmarks) / self.min_frames
            cv2.putText(frame, f"Buffer postura: {len(self.buffer_landmarks)}/{self.min_frames}", (20, 165),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.rectangle(frame, (20, 175), (20 + int(200 * progress), 185), (0, 255, 0), -1)
        
        # Info técnica no canto
        cv2.putText(frame, f"Buffer: {len(self.buffer_landmarks)}/{self.max_frames}",
                   (w-180, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)


class LandmarkFilter:
    """Filtro temporal para suavizar landmarks"""
    
    def __init__(self, suavizacao=0.8):
        self.ultimo_valido = None
        self.alpha = suavizacao
        
    def filtrar(self, landmarks_atuais, qualidades):
        if self.ultimo_valido is None:
            self.ultimo_valido = landmarks_atuais.copy()
            return landmarks_atuais
            
        landmarks_filtrados = []
        for i, (atual, qualidade) in enumerate(zip(landmarks_atuais, qualidades)):
            if qualidade > 0.5:
                novo = self.alpha * atual + (1 - self.alpha) * self.ultimo_valido[i]
                landmarks_filtrados.append(novo)
            else:
                landmarks_filtrados.append(self.ultimo_valido[i] * 0.98)
        
        self.ultimo_valido = landmarks_filtrados.copy()
        return landmarks_filtrados


class MicrogestureAggregator:
    """Aggregator simplificado para microgesture baseado no código original"""
    
    def __init__(self, window_sec=600, bucket_sec=60, warmup_sec=120):
        self.window_sec = window_sec
        self.bucket_sec = bucket_sec
        self.warmup_sec = warmup_sec
        self.ts = deque()
        self.data = {}
        self.mu = {}
        self.sd = {}
        self.locked = False
        self.last_bucket_idx = None
        
        # Inicializa aliases
        self.aliases = [
            "SleftEyeClosed", "SrightEyeClosed", "SAu43_EyesClosed",
            "SmouthOpen", "SAu25_LipsPart", "SAu26_JawDrop", "SAu27_MouthStretch",
            "HeadYaw", "HeadPitch", "HeadRoll"
        ]
        
        for alias in self.aliases:
            self.data[alias] = deque()
            self.mu[alias] = None
            self.sd[alias] = None

    def reset(self):
        """Reseta o aggregator"""
        self.ts.clear()
        for alias in self.aliases:
            self.data[alias].clear()
            self.mu[alias] = None
            self.sd[alias] = None
        self.locked = False
        self.last_bucket_idx = None

    def append(self, t, vals):
        """Adiciona valores ao buffer"""
        self.ts.append(t)
        for alias in self.aliases:
            v = vals.get(alias, None)
            if v is not None and np.isfinite(v):
                self.data[alias].append((t, float(v)))
        self._trim(t)
        self._update_baseline()

    def _trim(self, now):
        """Remove dados antigos"""
        cutoff = now - self.window_sec
        while self.ts and self.ts[0] < cutoff:
            self.ts.popleft()
        for alias in self.aliases:
            dq = self.data[alias]
            while dq and dq[0][0] < cutoff:
                dq.popleft()

    def _update_baseline(self):
        """Atualiza baseline"""
        if self.locked or not self.ts:
            return
        
        span = self.ts[-1] - self.ts[0]
        for alias in self.aliases:
            arr = np.array([v for _, v in self.data[alias]], dtype=float)
            if arr.size >= 5:
                mu = float(np.nanmean(arr))
                s = float(np.nanstd(arr, ddof=1))
                self.mu[alias] = mu
                self.sd[alias] = s if (np.isfinite(s) and s > 1e-8) else np.nan
        
        if span >= self.warmup_sec:
            self.locked = True

    def is_new_minute(self, now_ts):
        """Verifica se é um novo minuto"""
        idx = int(now_ts // self.bucket_sec)
        if self.last_bucket_idx is None:
            self.last_bucket_idx = idx
            return False
        if idx != self.last_bucket_idx:
            self.last_bucket_idx = idx
            return True
        return False

    def build_row_last_minute(self, feature_names, now_ts):
        """Constrói row de features do último minuto"""
        # Versão simplificada - retorna features básicas
        # Em uma implementação completa, você processaria o último minuto de dados
        row = {}
        for name in feature_names:
            row[name] = np.random.random()  # Placeholder - implemente conforme necessário
        return row