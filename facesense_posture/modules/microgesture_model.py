"""
Módulo de monitoramento de microgesture facial
Baseado no realtime_mediapipe.py - detecta estresse através de microexpressões
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
import json
from pathlib import Path
from collections import deque
import joblib
import simpleaudio as sa


class MicrogestureMonitor:
    """
    Monitor integrado de microgesture facial + postura
    Combina análise de estresse facial com classificação de postura usando Keras
    """

    def __init__(
        self,
        model_path="models/model_face_temporal.pkl",
        features_json_path="models/features.json",
        posture_model_path="models/best_model.keras",
        beep_callback=None,
        update_callback=None,
        alerta_segundos=5,
        lembrete_alongar_min=45,
        # tempo de janela e agregação (compatível com realtime_mediapipe.py)
        window_sec=600,
        bucket_sec=60,
        warmup_sec=120,
        threshold_shift=0.08,
        aliases=None
    ):
        # Configurações
        self.beep_callback = beep_callback
        self.update_callback = update_callback
        self.ALERTA_SEGUNDOS_POSTURA = alerta_segundos
        self.LEMBRETE_ALONGAR_MIN = lembrete_alongar_min
        self.VOLUME = 0.8
        # Controla se a interface gráfica deve desenhar textos/overlays no frame
        # Defina como False para não mostrar resultados na imagem de vídeo
        self.show_on_screen = True
        
        # MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        
        # MediaPipe Pose para análise de postura
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Modelos e configurações
        self.microgesture_model = None
        self.posture_model = None
        self.feature_names = []
        self.threshold_base = 0.5
        self.threshold_shift = float(threshold_shift)  # Mais difícil classificar como estresse
        
        self._carregar_modelos(model_path, features_json_path, posture_model_path)

        # Aliases (possibilita sobrescrever via construtor)
        if aliases is not None and isinstance(aliases, (list, tuple)) and len(aliases) > 0:
            self.aliases = list(aliases)

        # Aggregator para features temporais (expõe parâmetros compatíveis com realtime_mediapipe)
        self.aggregator = MinuteAggregator(window_sec=window_sec, bucket_sec=bucket_sec, warmup_sec=warmup_sec)

        # Guarda os parâmetros para debug/inspeção
        self._cfg = dict(window_sec=window_sec, bucket_sec=bucket_sec, warmup_sec=warmup_sec, threshold_shift=self.threshold_shift)
        
        # Estado do monitor
        self.cap = None
        self.running = False
        self.last_proba = None
        self.nivel_estresse = "BAIXO"
        self.draw_debug = True
        
        # Controle de alertas
        self.ultimo_alerta = 0
        # Histórico de classes de postura com timestamps (para detectar repetições)
        self.posture_history = deque()
        # Histórico de eventos ALTO do detector facial (timestamps)
        self.micro_high_history = deque()
        # Indica se o último frame de postura estava ocluído
        self.last_posture_occluded = False
        # Última predição crua do modelo Keras (lista de floats) ou None
        self.last_posture_prediction = None
        # Cooldown mínimo entre alertas (segundos) para evitar spam
        self.alert_cooldown = 300  # 5 minutos por padrão
        self.last_alert_time = 0

        self._excluded_posture_classes = {1}
        
        # Aliases para features faciais (preserva override via construtor)
        if not hasattr(self, 'aliases') or not getattr(self, 'aliases'):
            self.aliases = [
                "SleftEyeClosed", "SrightEyeClosed", "SAu43_EyesClosed",
                "SmouthOpen", "SAu25_LipsPart", "SAu26_JawDrop", "SAu27_MouthStretch",
                "HeadYaw", "HeadPitch", "HeadRoll"
            ]
        
        # Índices dos landmarks importantes
        self.idx = {
            "nose_tip": 1, "chin": 152,
            "left_eye_outer": 33, "right_eye_outer": 263,
            "mouth_left": 61, "mouth_right": 291,
            "mouth_top": 13, "mouth_bottom": 14,
            "eye_left_top": 159, "eye_left_bottom": 145, 
            "eye_left_inner": 133, "eye_left_outer": 33,
            "eye_right_top": 386, "eye_right_bottom": 374, 
            "eye_right_inner": 362, "eye_right_outer": 263
        }

    def _carregar_modelos(self, model_path, features_path, posture_model_path):
        """Carrega os modelos de microgesture e postura"""
        try:
            # Tenta diferentes caminhos possíveis
            possible_paths = [
                model_path,  # caminho original passado
                Path(model_path),  # como Path object
            ]
            
            # Se não começar com '/', adiciona caminhos relativos possíveis
            if not model_path.startswith('/'):
                import os
                current_dir = Path.cwd()
                possible_paths.extend([
                    current_dir / model_path,
                    current_dir / "facesense_posture" / "models" / "model_face_temporal.pkl",
                    current_dir / ".." / "facesense_posture" / "models" / "model_face_temporal.pkl",
                    Path(__file__).parent.parent / "models" / "model_face_temporal.pkl"
                ])
            
            # Tenta carregar o modelo do primeiro caminho que existir
            model_loaded = False
            for path in possible_paths:
                if Path(path).exists():
                    self.microgesture_model = joblib.load(path)
                    print(f"✅ Modelo de microgesture carregado: {path}")
                    model_loaded = True
                    break
            
            if not model_loaded:
                raise FileNotFoundError(f"Modelo não encontrado em nenhum dos caminhos: {possible_paths}")
            
            # Carrega features e threshold - mesma lógica
            features_possible_paths = [
                features_path,
                Path(features_path),
            ]
            
            if not features_path.startswith('/'):
                current_dir = Path.cwd()
                features_possible_paths.extend([
                    current_dir / features_path,
                    current_dir / "facesense_posture" / "models" / "features.json",
                    current_dir / ".." / "facesense_posture" / "models" / "features.json",
                    Path(__file__).parent.parent / "models" / "features.json"
                ])
            
            features_loaded = False
            for path in features_possible_paths:
                if Path(path).exists():
                    with open(path, 'r', encoding='utf-8') as f:
                        meta = json.load(f)
                        self.feature_names = meta["features"]
                        self.threshold_base = float(meta.get("threshold", 0.5))
                    
                    print(f"✅ Features carregadas: {len(self.feature_names)} features")
                    print(f"✅ Threshold base: {self.threshold_base}")
                    features_loaded = True
                    break
            
            if not features_loaded:
                print(f"⚠️ Arquivo de features não encontrado: {features_possible_paths}")
            
            # Carrega modelo de postura Keras
            self._carregar_modelo_postura(posture_model_path)
                
        except Exception as e:
            print(f"❌ Erro ao carregar modelos: {e}")
            self.microgesture_model = None
            self.posture_model = None
    
    def _carregar_modelo_postura(self, posture_model_path):
        """Carrega o modelo Keras para classificação de postura"""
        try:
            # Tenta diferentes caminhos para o modelo de postura
            posture_paths = [
                posture_model_path,
                Path(posture_model_path),
            ]
            
            if not posture_model_path.startswith('/'):
                current_dir = Path.cwd()
                posture_paths.extend([
                    current_dir / posture_model_path,
                    current_dir / "facesense_posture" / "models" / "best_model.keras",
                    current_dir / ".." / "facesense_posture" / "models" / "best_model.keras",
                    Path(__file__).parent.parent / "models" / "best_model.keras"
                ])
            
            # Carregamento com TensorFlow/Keras
            import tensorflow as tf
            from tensorflow import keras
            
            posture_loaded = False
            for path in posture_paths:
                if Path(path).exists():
                    self.posture_model = keras.models.load_model(path)
                    print(f"✅ Modelo de postura Keras carregado: {path}")
                    print(f"📊 Input shape: {self.posture_model.input_shape}")
                    print(f"📊 Output shape: {self.posture_model.output_shape}")
                    posture_loaded = True
                    break
            
            if not posture_loaded:
                print(f"⚠️ Modelo de postura não encontrado: {posture_paths}")
                self.posture_model = None
                
        except Exception as e:
            print(f"❌ Erro ao carregar modelo de postura: {e}")
            self.posture_model = None

    def _play_tone(self, freq=1000, dur=300, volume=1.0):
        """Reproduz tom de alerta"""
        try:
            fs = 44100  # taxa de amostragem
            t = np.linspace(0, dur / 1000, int(fs * dur / 1000), False)
            tone = np.sin(freq * t * 2 * np.pi)
            audio = (tone * (32767 * volume)).astype(np.int16)
            play_obj = sa.play_buffer(audio, 1, 2, fs)
            play_obj.wait_done()
        except Exception as e:
            print(f"Erro ao reproduzir som: {e}")

    def iniciar(self):
        """Inicia o monitoramento integrado"""
        if self.microgesture_model is None:
            raise RuntimeError("Modelo de microgesture não foi carregado!")
        if self.posture_model is None:
            print("⚠️ Modelo de postura não disponível - usando apenas microgesture")
            
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Não foi possível acessar a câmera!")
            
        self.running = True
        # reseta aggregator e imprime parâmetros úteis para debug
        try:
            self.aggregator.reset()
            print(f"🧠 Monitor integrado (microgesture + postura) iniciado! config={self._cfg}")
        except Exception:
            pass

    def parar(self):
        """Para o monitoramento"""
        self.running = False
        if self.cap:
            self.cap.release()
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
        print("🛑 Monitor de microgesture parado.")

    def processar_frame(self):
        """
        Processa um frame e retorna: (frame, status, dica, cor)
        Compatível com a interface do app.py original
        """
        if not self.running or not self.cap:
            return None, "parado", "Sistema parado", (128, 128, 128)

        ret, frame = self.cap.read()
        if not ret:
            return None, "erro", "Erro na câmera", (255, 0, 0)

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Processa face e pose
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # FaceMesh pode por vezes lançar erros internos (ex: '_graph is None in SolutionBase')
        # em alguns ambientes. Tratamos e tentamos reinicializar o FaceMesh automaticamente.
        try:
            face_results = self.face_mesh.process(rgb)
        except Exception as e:
            msg = str(e)
            print(f"Erro ao processar FaceMesh: {msg}")
            if "_graph is None" in msg or "SolutionBase" in msg:
                try:
                    print("Tentando reiniciar FaceMesh...")
                    try:
                        self.face_mesh.close()
                    except Exception:
                        pass
                    self.face_mesh = self.mp_face_mesh.FaceMesh(
                        static_image_mode=False,
                        max_num_faces=1,
                        refine_landmarks=True,
                        min_detection_confidence=0.6,
                        min_tracking_confidence=0.6
                    )
                    face_results = self.face_mesh.process(rgb)
                    print("FaceMesh reiniciado com sucesso.")
                except Exception as e2:
                    print(f"Falha ao reiniciar FaceMesh: {e2}")
                    face_results = None
            else:
                face_results = None

        try:
            pose_results = self.pose.process(rgb)
        except Exception as e:
            print(f"Erro ao processar Pose: {e}")
            pose_results = None
        
        # Extrai features faciais
        vals = {a: np.nan for a in self.aliases}
        ypr = (np.nan, np.nan, np.nan)
        posture_status = "indefinido"
        posture_confidence = 0.5
        
        # Análise facial
        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]
            lm = face_landmarks.landmark
            
            # Desenha landmarks principais se debug ativo
            if self.draw_debug:
                self._draw_landmarks(frame, face_landmarks, w, h)
            
            # Extrai features
            vals, ypr = self._extract_facial_features(lm, frame.shape)
            
            # Adiciona ao aggregator
            now = time.time()
            self.aggregator.append(now, vals)
            
            # Classificação a cada minuto
            if self.aggregator.is_new_minute(now) and self.aggregator.locked:
                self._classificar_estresse(now)
        
        # Análise de postura
        if pose_results.pose_landmarks and self.posture_model is not None:
            posture_status, posture_confidence = self._classificar_postura(frame, pose_results)
            
            # Desenha pose se debug ativo
            if self.draw_debug:
                self._draw_pose(frame, pose_results)

            # Registro histórico de classes de postura e lógica de alertas
            # Janela: últimos 5 minutos. Contagem de mesma classe > 3 -> alerta
            try:
                now_ts = time.time()

                # Se posture_status for inteiro, registra no histórico
                if isinstance(posture_status, int):
                    self.posture_history.append((now_ts, int(posture_status)))

                # Mantém apenas os últimos 5 minutos
                cutoff = now_ts - (5 * 60)
                while self.posture_history and self.posture_history[0][0] < cutoff:
                    self.posture_history.popleft()

                # Conta ocorrências por classe dentro da janela
                counts = {}
                for _, c in self.posture_history:
                    counts[c] = counts.get(c, 0) + 1

                    # DEBUG: mostra estado atual do histórico de postura para diagnóstico
                    try:
                        recent = list(self.posture_history)[-12:]
                        print(f"[MICRO DEBUG] posture_class={posture_status} conf={posture_conf:.3f} recent_history={recent} counts={counts} last_proba={self.last_proba}")
                    except Exception:
                        pass

                # Conjunto de classes a ignorar para alertas (especificadas)
                ignore_classes = {15, 16, 22, 23}
                # também inclui qualquer classe previamente excluída
                try:
                    ignore_classes = ignore_classes.union(set(self._excluded_posture_classes))
                except Exception:
                    pass

                # Determina nível facial atual usando last_proba
                level = None
                if self.last_proba is not None and np.isfinite(self.last_proba):
                    if self.last_proba >= 0.65:
                        level = 'ALTO'
                    elif self.last_proba >= 0.45:
                        level = 'MÉDIO'
                    else:
                        level = 'BAIXO'

                # Para cada classe com contagem alta, decide ação
                for cls, cnt in counts.items():
                    if cls in ignore_classes:
                        continue
                    if cnt > 3 and level is not None:
                        now = now_ts
                        if now - self.last_alert_time <= self.alert_cooldown:
                            # ainda em cooldown
                            break

                        # Nível MÉDIO -> popup discreto no canto
                        if level == 'MÉDIO' and self.update_callback:
                            msg = (
                                f"Detector facial: nível MÉDIO.\n"
                                f"Classe {cls} repetida {cnt}x nos últimos 5 min.\n"
                                "Sugestão: faça uma pausa curta e relaxe."
                            )
                            try:
                                self.update_callback('alert_corner', msg)
                            except Exception:
                                pass
                            self.last_alert_time = now
                            break

                        # Nível ALTO -> popup central + som
                        if level == 'ALTO' and self.update_callback:
                            msg = (
                                f"ALERTA — Sinais de estresse detectados (nível ALTO).\n"
                                f"Classe {cls} repetida {cnt}x nos últimos 5 min.\n"
                                "Recomendação: pare por 2–5 minutos e relaxe."
                            )
                            try:
                                self.update_callback('alert_center', msg)
                            except Exception:
                                pass
                            # reproduz som de alerta se disponível
                            if self.beep_callback:
                                try:
                                    self.beep_callback()
                                except Exception:
                                    pass
                            self.last_alert_time = now
                            break
            except Exception as e:
                print(f"Erro ao processar histórico/alertas de postura: {e}")
        
        # Desenha interface (somente se permitido)
        if getattr(self, 'show_on_screen', True):
            try:
                self._draw_interface(frame, w, h, ypr)
            except Exception:
                pass
            try:
                self._draw_interface_integrada(frame, w, h, ypr, posture_status, posture_confidence)
            except Exception:
                pass
        
        # Determina status geral integrado
        status, dica, cor = self._get_status_integrado(posture_status, posture_confidence)
        
        return frame, status, dica, cor

    def _extract_facial_features(self, landmarks, image_shape):
        """Extrai features faciais do MediaPipe"""
        vals = {}
        
        def P(i): 
            li = landmarks[i]
            return (li.x, li.y)
        
        try:
            # Pontos principais
            mouth_top = P(self.idx["mouth_top"])
            mouth_bottom = P(self.idx["mouth_bottom"])
            mouth_left = P(self.idx["mouth_left"])
            mouth_right = P(self.idx["mouth_right"])
            chin = P(self.idx["chin"])
            
            le_top = P(self.idx["eye_left_top"])
            le_bottom = P(self.idx["eye_left_bottom"])
            le_outer = P(self.idx["eye_left_outer"])
            le_inner = P(self.idx["eye_left_inner"])
            
            re_top = P(self.idx["eye_right_top"])
            re_bottom = P(self.idx["eye_right_bottom"])
            re_outer = P(self.idx["eye_right_outer"])
            re_inner = P(self.idx["eye_right_inner"])
            
            # Cálculos de features
            inter_pupil = self._dist(le_inner, re_inner) + 1e-9
            
            # Mouth features
            mar_v = self._mar(mouth_top, mouth_bottom, mouth_left, mouth_right)
            vals["SmouthOpen"] = mar_v
            vals["SAu25_LipsPart"] = mar_v
            vals["SAu26_JawDrop"] = self._jaw_drop(chin, mouth_top, inter_pupil)
            vals["SAu27_MouthStretch"] = self._mouth_stretch(mouth_left, mouth_right, inter_pupil)
            
            # Eye features
            le_ear = self._ear(le_top, le_bottom, le_outer, le_inner)
            re_ear = self._ear(re_top, re_bottom, re_inner, re_outer)
            vals["SleftEyeClosed"] = 1.0 - le_ear
            vals["SrightEyeClosed"] = 1.0 - re_ear
            vals["SAu43_EyesClosed"] = (vals["SleftEyeClosed"] + vals["SrightEyeClosed"]) / 2.0
            
            # Head pose (simplificado)
            lm_xy = [(li.x, li.y) for li in landmarks]
            ypr = self._head_pose_simple(lm_xy)
            yaw, pitch, roll = ypr
            vals["HeadYaw"] = yaw / 45.0
            vals["HeadPitch"] = pitch / 45.0
            vals["HeadRoll"] = roll / 45.0
            
        except Exception as e:
            print(f"Erro ao extrair features: {e}")
            # Features padrão em caso de erro
            for alias in self.aliases:
                vals[alias] = 0.0
            ypr = (0.0, 0.0, 0.0)
        
        return vals, ypr

    def _classificar_estresse(self, now_ts):
        """Retorna predição pura do modelo de estresse"""
        try:
            # Debug: mostra se aggregator está bloqueado e quantas feature_names temos
            try:
                print(f"[MICRO DEBUG] aggregator.locked={self.aggregator.locked} feature_names={len(self.feature_names)}")
            except Exception:
                pass

            row = self.aggregator.build_row_last_minute(self.feature_names, now_ts)
            if row is None:
                print("[MICRO DEBUG] build_row_last_minute returned None (não há dados suficientes ou janela vazia)")
            else:
                try:
                    X = pd.DataFrame([row], columns=self.feature_names)
                    print(f"[MICRO DEBUG] Built DataFrame X shape={X.shape}")
                except Exception as e:
                    print(f"[MICRO DEBUG] Erro ao construir DataFrame X: {e}")
                    X = None

                if X is not None:
                    try:
                        # Alguns modelos podem não implementar predict_proba; tenta com predict se necessário
                        if hasattr(self.microgesture_model, 'predict_proba'):
                            probs = self.microgesture_model.predict_proba(X)
                            proba = float(probs[:, 1][0])
                        else:
                            preds = self.microgesture_model.predict(X)
                            # assume-se saída contínua entre 0 e 1
                            proba = float(preds[0])

                        # Armazena valor puro sem classificação
                        self.last_proba = proba
                        self.nivel_estresse = "RAW"
                        print(f"[MICROGESTURE] Valor puro={proba:.6f}")

                        # Callback para UI com valor puro
                        if self.update_callback:
                            status = "raw_microgesture"
                            dica = f"Microgesture raw: {proba:.6f}"
                            try:
                                self.update_callback(status, dica)
                            except Exception:
                                pass

                        # Registra eventos ALTO para lógica de oclusão (>= 0.65)
                        try:
                            if np.isfinite(proba) and proba >= 0.65:
                                now = time.time()
                                self.micro_high_history.append(now)
                                cutoff = now - (5 * 60)
                                while self.micro_high_history and self.micro_high_history[0] < cutoff:
                                    self.micro_high_history.popleft()
                        except Exception:
                            pass
                    except Exception as e:
                        print(f"[MICRO DEBUG] Erro ao chamar o modelo microgesture.predict_proba/predict: {e}")
                    
        except Exception as e:
            print(f"Erro na predição: {e}")

    def force_classify_short_window(self, seconds=10):
        """Força uma classificação usando uma janela curta (útil para testes quando o warmup é longo).
        Retorna a probabilidade bruta ou None se não houver dados suficientes.
        """
        try:
            now = time.time()
            df = self.aggregator._window_df(now, min(seconds, self.aggregator.bucket_sec))
            if df.empty:
                print(f"[MICRO DEBUG] force_classify_short_window: janela de {seconds}s vazia")
                return None

            z = self.aggregator._zscore(df)
            agg = {}
            for alias in self.aggregator.aliases:
                col = alias + "_z"
                if col in z.columns:
                    v = z[col].dropna().values.astype(float)
                    agg[f"{col}_mean"] = float(np.nanmean(v)) if v.size else np.nan
                    agg[f"{col}_p90"] = self.aggregator._p90(v)
                    agg[f"{col}_jitter"] = self.aggregator._jitter(v)

            # monta row alinhada com feature_names
            row = {name: agg.get(name, np.nan) for name in self.feature_names}
            X = pd.DataFrame([row], columns=self.feature_names)

            # chama o modelo (igual à lógica principal)
            if hasattr(self.microgesture_model, 'predict_proba'):
                probs = self.microgesture_model.predict_proba(X)
                proba = float(probs[:, 1][0])
            else:
                preds = self.microgesture_model.predict(X)
                proba = float(preds[0])

            self.last_proba = proba
            print(f"[MICRO DEBUG] force_classify_short_window produced proba={proba:.6f}")
            return proba
        except Exception as e:
            print(f"[MICRO DEBUG] force_classify_short_window error: {e}")
            return None

    def _draw_landmarks(self, frame, face_landmarks, w, h):
        """Desenha landmarks faciais principais"""
        important_points = [
            "nose_tip", "chin", "mouth_left", "mouth_right", 
            "mouth_top", "mouth_bottom", "eye_left_outer", 
            "eye_left_inner", "eye_left_top", "eye_left_bottom",
            "eye_right_outer", "eye_right_inner", "eye_right_top", "eye_right_bottom"
        ]
        
        for name in important_points:
            idx = self.idx.get(name)
            if idx is not None and idx < len(face_landmarks.landmark):
                lm = face_landmarks.landmark[idx]
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    def _draw_interface(self, frame, w, h, ypr):
        """Desenha interface do sistema"""
        # Status do aggregator
        if not self.aggregator.locked:
            span = (self.aggregator.ts[-1] - self.aggregator.ts[0]) if self.aggregator.ts else 0
            cv2.putText(frame, "Calibrando baseline...", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(frame, f"{int(span)}/{self.aggregator.warmup_sec}s", (20, 68), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            # Countdown para próxima classificação
            secs_to_next = 60 - int(time.time()) % 60
            
            if self.last_proba is not None:
                # Status com cor baseada no nível
                color = (0, 200, 0) if self.nivel_estresse == "BAIXO" else \
                       (0, 165, 255) if self.nivel_estresse == "MÉDIO" else (0, 0, 255)
                
                # Fundo para informações
                cv2.rectangle(frame, (10, 10), (450, 120), (20, 20, 20), -1)
                
                cv2.putText(frame, f"Estresse: {self.nivel_estresse}", (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
                thr_adj = max(0.01, min(0.99, self.threshold_base + self.threshold_shift))
                cv2.putText(frame, f"Prob: {self.last_proba:.3f} | Thr: {thr_adj:.3f}", 
                           (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
            else:
                cv2.putText(frame, "Aguardando 1º minuto...", (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            cv2.putText(frame, f"T-{secs_to_next:02d}s", (350, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        
        # Mini HUD com head pose
        if np.isfinite(ypr[0]):
            cv2.putText(frame, f"Y:{ypr[0]:+.1f} P:{ypr[1]:+.1f} R:{ypr[2]:+.1f}", 
                       (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)

    def _get_status(self):
        """Retorna status atual para o app principal"""
        if not self.aggregator.locked:
            return "calibrando", "Calibrando baseline facial...", (255, 255, 0)
        
        if self.last_proba is None:
            return "aguardando", "Aguardando primeira análise...", (255, 255, 255)
        
        if self.nivel_estresse == "ALTO":
            return "estresse_alto", f"Estresse elevado detectado ({self.last_proba:.1%})", (0, 0, 255)
        elif self.nivel_estresse == "MÉDIO":
            return "estresse_medio", f"Estresse moderado ({self.last_proba:.1%})", (0, 165, 255)
        else:
            return "estresse_baixo", f"Nível de estresse normal ({self.last_proba:.1%})", (0, 255, 0)

    # === FUNÇÕES AUXILIARES GEOMÉTRICAS ===
    
    def _dist(self, a, b):
        return float(np.linalg.norm(np.array(a) - np.array(b)))
    
    def _ear(self, eye_top, eye_bottom, eye_left, eye_right):
        open_v = self._dist(eye_top, eye_bottom)
        width = self._dist(eye_left, eye_right) + 1e-9
        return open_v / width
    
    def _mar(self, mouth_top, mouth_bottom, mouth_left, mouth_right):
        open_v = self._dist(mouth_top, mouth_bottom)
        width = self._dist(mouth_left, mouth_right) + 1e-9
        return open_v / width
    
    def _jaw_drop(self, chin, upper_lip, ref_width):
        return self._dist(chin, upper_lip) / (ref_width + 1e-9)
    
    def _mouth_stretch(self, mouth_left, mouth_right, ref_width):
        return self._dist(mouth_left, mouth_right) / (ref_width + 1e-9)
    
    def _head_pose_simple(self, landmarks_xy):
        """Estimativa simples de head pose"""
        try:
            nose = landmarks_xy[1]
            chin = landmarks_xy[152]
            left_eye = landmarks_xy[33]
            right_eye = landmarks_xy[263]
            
            # Pitch aproximado
            pitch = (nose[1] - chin[1]) * 90  # normalizado
            
            # Yaw aproximado
            eye_center_x = (left_eye[0] + right_eye[0]) / 2
            yaw = (nose[0] - eye_center_x) * 90
            
            # Roll aproximado (baseado na inclinação dos olhos)
            eye_diff_y = left_eye[1] - right_eye[1]
            roll = eye_diff_y * 90
            
            return (yaw, pitch, roll)
            
        except:
            return (0.0, 0.0, 0.0)


class MinuteAggregator:
    """Aggregator para análise temporal de features faciais"""
    
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
        
        # Aliases para features
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
        """Adiciona valores temporais"""
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
        """Atualiza baseline estatístico"""
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
        """Verifica se chegou um novo minuto"""
        idx = int(now_ts // self.bucket_sec)
        if self.last_bucket_idx is None:
            self.last_bucket_idx = idx
            return False
        if idx != self.last_bucket_idx:
            self.last_bucket_idx = idx
            return True
        return False

    def build_row_last_minute(self, feature_names, now_ts):
        """Constrói features do último minuto para o modelo"""
        df = self._window_df(now_ts, self.bucket_sec)
        if df.empty:
            return None
            
        z = self._zscore(df)
        agg = {}
        
        for alias in self.aliases:
            col = alias + "_z"
            if col in z.columns:
                v = z[col].dropna().values.astype(float)
                agg[f"{col}_mean"] = float(np.nanmean(v)) if v.size else np.nan
                agg[f"{col}_p90"] = self._p90(v)
                agg[f"{col}_jitter"] = self._jitter(v)
        
        row = {name: agg.get(name, np.nan) for name in feature_names}
        return row

    def _window_df(self, t_end, span):
        """Cria dataframe da janela temporal"""
        frames = []
        for alias in self.aliases:
            if self.data[alias]:
                tt = [ti for ti, _ in self.data[alias]]
                vv = [vi for _, vi in self.data[alias]]
                frames.append(pd.DataFrame({"t": tt, alias: vv}).set_index("t"))
        
        if not frames:
            return pd.DataFrame()
            
        df = pd.concat(frames, axis=1, join="outer").sort_index()
        t0 = t_end - span
        return df.loc[df.index.to_series().between(t0, t_end)]

    def _zscore(self, df):
        """Calcula z-scores baseado no baseline"""
        z = pd.DataFrame(index=df.index)
        for alias in self.aliases:
            if alias in df.columns:
                mu = self.mu.get(alias, None)
                sd = self.sd.get(alias, None)
                col = df[alias].astype(float)
                if mu is not None and sd is not None and np.isfinite(sd) and sd > 1e-8:
                    z[alias + "_z"] = (col - mu) / sd
                else:
                    z[alias + "_z"] = col
        return z

    @staticmethod
    def _p90(x):
        return float(np.nanpercentile(x, 90)) if (len(x) > 0 and np.isfinite(x).any()) else np.nan

    @staticmethod
    def _jitter(x):
        if len(x) <= 1:
            return np.nan
        return float(np.nansum(np.abs(np.diff(x))))
    
    # === MÉTODOS PARA ANÁLISE DE POSTURA ===
    
    def _classificar_postura(self, frame, pose_results):
        """Retorna predição pura do modelo Keras"""
        try:
            # Extrai features do frame para o modelo Keras
            features, _ = self._extrair_features_postura(frame, pose_results)

            # Faz predição (vetor bruto)
            prediction = self.posture_model.predict(features, verbose=0)[0]

            # Interpreta saída: index da classe e confiança máxima
            if getattr(prediction, 'size', None) == 1:
                class_idx = 0
                confidence = float(prediction[0])
            else:
                class_idx = int(np.argmax(prediction))
                confidence = float(np.max(prediction))

            # Armazena resultados brutos para relatórios/UI
            self.last_posture_class = class_idx
            self.last_posture_confidence = confidence
            try:
                self.last_posture_prediction = (prediction.tolist() if hasattr(prediction, 'tolist') else list(prediction))
            except Exception:
                self.last_posture_prediction = None
            self.last_posture_occluded = False

            # Tenta carregar nomes legíveis se existir arquivo; caso contrário mantém class_{i}
            try:
                num_classes = int(getattr(prediction, 'size', len(prediction)))
            except Exception:
                num_classes = None

            try:
                if num_classes == 3:
                    label_map = {0: 'sinais_estresse', 1: 'nao_estresse', 2: 'postural'}
                    self.last_posture_label = label_map.get(int(class_idx), f'class_{class_idx}')
                else:
                    labels_path = Path(__file__).parent.parent / "models" / "best_model_labels.json"
                    if labels_path.exists():
                        with open(labels_path, 'r', encoding='utf-8') as lf:
                            labels = json.load(lf)
                        if isinstance(labels, (list, tuple)) and num_classes is not None and len(labels) >= num_classes:
                            self.last_posture_label = labels[int(class_idx)]
                        else:
                            self.last_posture_label = f'class_{class_idx}'
                    else:
                        self.last_posture_label = f'class_{class_idx}'
            except Exception:
                self.last_posture_label = (f'class_{class_idx}' if class_idx is not None else None)

            # Retorna a classe prevista e a confiança bruta (mantém compatibilidade)
            return class_idx, confidence
            
        except Exception as e:
            print(f"Erro na predição de postura: {e}")
            return "erro", 0.0
    
    def _extrair_features_postura(self, frame, pose_results):
        """Extrai features do frame para o modelo Keras de postura"""
        # Extrai imagem em escala de cinza e redimensiona para o input do modelo
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_resized = cv2.resize(gray, (99, 200))

        # Normaliza e prepara batch
        frame_normalized = frame_resized.astype(np.float32) / 255.0
        frame_batch = np.expand_dims(frame_normalized, axis=0)

        # Não fazemos checagem de oclusão — retornamos sempre um batch válido
        return frame_batch, False
    
    def _draw_pose(self, frame, pose_results):
        """Desenha landmarks de pose"""
        if pose_results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, 
                pose_results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(0, 255, 0), thickness=2, circle_radius=2
                ),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(0, 255, 255), thickness=2
                )
            )
    
    def _draw_interface_integrada(self, frame, w, h, ypr, posture_status, posture_confidence):
        """Desenha interface com valores puros dos dois modelos"""
        # Fundo para informações
        cv2.rectangle(frame, (10, 10), (600, 170), (20, 20, 20), -1)
        
        # Título
        cv2.putText(frame, "VALORES PUROS DOS MODELOS", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Status do aggregator (microgesture)
        if not self.aggregator.locked:
            span = (self.aggregator.ts[-1] - self.aggregator.ts[0]) if self.aggregator.ts else 0
            cv2.putText(frame, "Calibrando baseline facial...", (20, 65), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"{int(span)}/{self.aggregator.warmup_sec}s", (20, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        else:
            # Valor puro do modelo de microgesture
            if self.last_proba is not None:
                cv2.putText(frame, f"MICROGESTURE: {self.last_proba:.6f}", (20, 65), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                cv2.putText(frame, "MICROGESTURE: Aguardando...", (20, 65), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Valor puro do modelo de postura (classe prevista + confiança) ou indicação de oclusão
        if getattr(self, 'last_posture_occluded', False):
            cv2.putText(frame, "POSTURA: OCLUSÃO BRAÇOS/MÃOS", (20, 95),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        elif self.posture_model is not None and isinstance(posture_status, int):
            cv2.putText(frame, f"POSTURA: class={posture_status} conf={posture_confidence:.6f}", (20, 95),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
        else:
            cv2.putText(frame, "POSTURA: Não disponível", (20, 95),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
        
        # Informações adicionais
        cv2.putText(frame, "Valores entre 0.0 e 1.0 (sem interpretação)", (20, 125), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Countdown próxima classificação
        if self.aggregator.locked:
            secs_to_next = 60 - int(time.time()) % 60
            cv2.putText(frame, f"Próxima análise: {secs_to_next:02d}s", (20, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        # Head pose
        if np.isfinite(ypr[0]):
            cv2.putText(frame, f"Y:{ypr[0]:+.1f} P:{ypr[1]:+.1f} R:{ypr[2]:+.1f}", 
                       (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 2)
    
    def _get_status_integrado(self, posture_status, posture_confidence):
        """Retorna valores puros dos dois modelos sem interpretação"""
        # Estado de calibração
        if not self.aggregator.locked:
            return "calibrando", "Calibrando baseline facial...", (255, 255, 0)
        
        if self.last_proba is None:
            return "aguardando", "Aguardando primeira análise...", (255, 255, 255)
        
        # Valores puros sem interpretação
        micro_raw = self.last_proba
        postura_raw = posture_confidence if isinstance(posture_status, int) else 0.0
        
        # Formatação dos valores puros
        valores_puros = f"MICRO: {micro_raw:.6f} | POSTURA: {postura_raw:.6f}"
        
        return "valores_puros", valores_puros, (0, 255, 255)


# --- Correção de vinculação de métodos ---
# Alguns métodos de postura foram acidentalmente definidos dentro da classe
# `MinuteAggregator` durante edições. Aqui vinculamos as funções como métodos
# da classe `MicrogestureMonitor` para garantir que instâncias da classe
# tenham acesso a eles (bound methods).
try:
    # Os métodos foram definidos como parte da classe MinuteAggregator durante
    # uma edição; aqui apontamos os métodos corretos para a classe
    MicrogestureMonitor._classificar_postura = MinuteAggregator._classificar_postura
    MicrogestureMonitor._extrair_features_postura = MinuteAggregator._extrair_features_postura
    MicrogestureMonitor._draw_pose = MinuteAggregator._draw_pose
    MicrogestureMonitor._draw_interface_integrada = MinuteAggregator._draw_interface_integrada
    MicrogestureMonitor._get_status_integrado = MinuteAggregator._get_status_integrado
except Exception:
    # Em cenários estranhos (por exemplo import parcialmente correto), ignore
    pass


