"""
Módulo de classificação de postura usando modelo Keras
Integrado com PySide6 e substituindo o sistema MediaPipe
"""

import cv2
import mediapipe as mp
import numpy as np
import time
from tensorflow import keras
from collections import deque


class KerasPostureClassifier:
    """
    Classificador de postura usando modelo Keras treinado
    Compatível com a interface PySide6 do app.py
    """

    def __init__(
        self,
        model_path="models/stress.keras",
        beep_callback=None,
        update_callback=None,
        alerta_segundos=5,
        lembrete_alongar_min=45
    ):
        # Configurações
        self.beep_callback = beep_callback
        self.update_callback = update_callback
        self.ALERTA_SEGUNDOS_POSTURA = alerta_segundos
        self.LEMBRETE_ALONGAR_MIN = lembrete_alongar_min
        self.VOLUME = 0.8
        
        # MediaPipe para extração de landmarks
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Modelo Keras
        self.modelo = None
        self._carregar_modelo(model_path)
        
        # Buffer temporal para o modelo
        self.buffer_landmarks = []
        self.max_frames = 500
        self.min_frames = 50
        
        # Estado do classificador
        self.cap = None
        self.running = False
        self.ultima_predicao = None
        self.confianca_atual = 0.0
        self.classe_atual = -1
        
        # Controle de alertas
        self.ultimo_alerta = 0
        self.contador_postura_ruim = 0
        self.inicio_postura_ruim = None
        
        # Filtro temporal para landmarks
        self.filtro_landmarks = LandmarkFilter(suavizacao=0.8)

    def _carregar_modelo(self, model_path):
        """Carrega o modelo Keras"""
        try:
            self.modelo = keras.models.load_model(model_path)
            print(f"✅ Modelo Keras carregado: {self.modelo.input_shape} -> {self.modelo.output_shape}")
        except Exception as e:
            print(f"❌ Erro ao carregar modelo: {e}")
            self.modelo = None

    def iniciar(self):
        """Inicia a captura de vídeo"""
        if self.modelo is None:
            raise RuntimeError("Modelo não foi carregado!")
            
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Não foi possível acessar a câmera!")
            
        self.running = True
        self.buffer_landmarks = []
        self.contador_postura_ruim = 0
        print("🎯 Classificador Keras iniciado!")

    def parar(self):
        """Para a captura de vídeo"""
        self.running = False
        if self.cap:
            self.cap.release()
        print("🛑 Classificador parado.")

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

        # Flip horizontal para melhor experiência
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Processa com MediaPipe para extrair landmarks
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        
        if results.pose_landmarks:
            # Desenha skeleton
            self.mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
            )
            
            # Extrai e filtra landmarks
            landmarks_raw, qualidades = self._extrair_landmarks(results.pose_landmarks)
            landmarks_filtrados = self.filtro_landmarks.filtrar(landmarks_raw, qualidades)
            
            # Adiciona ao buffer
            self.buffer_landmarks.append(landmarks_filtrados)
            if len(self.buffer_landmarks) > self.max_frames:
                self.buffer_landmarks.pop(0)
            
            # Indicadores visuais das mãos (para debug de oclusão)
            self._desenhar_indicadores_maos(frame, results.pose_landmarks, w, h)
            
            # Classificação se tiver dados suficientes
            if len(self.buffer_landmarks) >= self.min_frames:
                status, dica, cor = self._classificar_postura()
                
                # Informações na tela
                self._desenhar_info_tela(frame, w, h)
                
                return frame, status, dica, cor
            else:
                # Ainda coletando dados
                progresso = len(self.buffer_landmarks) / self.min_frames
                status = "coletando"
                dica = f"Coletando dados... {len(self.buffer_landmarks)}/{self.min_frames}"
                cor = (255, 255, 0)
                
                # Barra de progresso
                bar_w = 300
                cv2.rectangle(frame, (20, h-60), (20 + int(bar_w * progresso), h-40), (0, 255, 0), -1)
                cv2.rectangle(frame, (20, h-60), (20 + bar_w, h-40), (255, 255, 255), 2)
                cv2.putText(frame, dica, (20, h-70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                return frame, status, dica, cor
        else:
            # Nenhuma pessoa detectada
            cv2.putText(frame, "NENHUMA PESSOA DETECTADA", (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            return frame, "sem_pessoa", "Posicione-se na frente da câmera", (255, 0, 0)

    def _extrair_landmarks(self, pose_landmarks):
        """Extrai landmarks com análise de qualidade"""
        landmarks_raw = []
        qualidades = []
        
        for i, lm in enumerate(pose_landmarks.landmark):
            landmarks_raw.extend([lm.x, lm.y, lm.z])
            
            # Avalia qualidade
            visibility = getattr(lm, 'visibility', 1.0)
            
            # Landmarks das mãos são mais sensíveis à oclusão
            if i in [15, 16, 17, 18, 19, 20, 21, 22]:  # mãos e pulsos
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
                    # Landmark problemático
                    cv2.circle(frame, (x, y), 8, (0, 0, 255), 2)
                else:
                    # Landmark bom
                    cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

    def _classificar_postura(self):
        """Executa a classificação usando o modelo Keras"""
        # Prepara sequência temporal (últimos 200 frames)
        sequence = np.zeros((200, 99))
        frames_recentes = self.buffer_landmarks[-200:] if len(self.buffer_landmarks) >= 200 else self.buffer_landmarks
        
        for i, frame_coords in enumerate(frames_recentes):
            if i < 200:
                sequence[i] = frame_coords
        
        # Predição
        try:
            pred = self.modelo.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
            self.classe_atual = np.argmax(pred)
            self.confianca_atual = np.max(pred)
            self.ultima_predicao = pred
            
            # Interpreta resultado (você pode ajustar essa lógica baseado no seu modelo)
            return self._interpretar_classificacao(self.classe_atual, self.confianca_atual)
            
        except Exception as e:
            print(f"Erro na predição: {e}")
            return "erro", "Erro na classificação", (255, 0, 0)

    def _interpretar_classificacao(self, classe, confianca):
        """
        Mostra apenas o valor puro da predição do modelo sem interpretação
        """
        # Sem interpretação - apenas mostra os valores brutos do modelo
        status = f"classe_{classe}"
        dica = f"Classe {classe} com {confianca:.2%} de confiança"
        
        # Cor baseada apenas na confiança, não no tipo de postura
        if confianca >= 0.7:
            cor = (0, 255, 0)  # Verde para alta confiança
        elif confianca >= 0.4:
            cor = (0, 255, 255)  # Amarelo para média confiança
        else:
            cor = (0, 0, 255)  # Vermelho para baixa confiança
        
        # Atualiza callback se disponível
        if self.update_callback:
            self.update_callback(status, dica)
            
        return status, dica, cor

    def _controlar_alertas_postura_ruim(self):
        """Alertas desabilitados - apenas mostramos valores puros"""
        pass

    def _resetar_contador_postura_ruim(self):
        """Contador desabilitado - apenas mostramos valores puros"""
        pass

    def _desenhar_info_tela(self, frame, w, h):
        """Desenha valores puros da classificação na tela"""
        if self.ultima_predicao is None:
            return
            
        # Fundo expandido para mais informações
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (450, 300), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Título
        cv2.putText(frame, "VALORES PUROS DO MODELO", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Classe com maior probabilidade
        cv2.putText(frame, f"CLASSE MAXIMA: {self.classe_atual}", (20, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # Probabilidade com 4 casas decimais
        cv2.putText(frame, f"PROB: {self.confianca_atual:.4f} ({self.confianca_atual:.2%})", 
                   (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Top 8 classes com valores detalhados
        cv2.putText(frame, "TOP 8 CLASSES:", (20, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        top_indices = np.argsort(self.ultima_predicao)[-8:][::-1]
        for i, idx in enumerate(top_indices):
            y_pos = 140 + i * 16
            prob = self.ultima_predicao[idx]
            
            # Cor baseada na probabilidade
            if prob > 0.1:
                cor = (0, 255, 0)
            elif prob > 0.05:
                cor = (0, 255, 255)
            else:
                cor = (200, 200, 200)
                
            cv2.putText(frame, f"{i+1}. Classe {idx:2d}: {prob:.4f} ({prob:.1%})",
                       (25, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, cor, 1)
        
        # Estatísticas da distribuição no canto direito
        cv2.putText(frame, f"Buffer: {len(self.buffer_landmarks)}/{self.max_frames}",
                   (w-180, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)
        cv2.putText(frame, f"Soma: {np.sum(self.ultima_predicao):.4f}",
                   (w-180, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(frame, f"Media: {np.mean(self.ultima_predicao):.4f}",
                   (w-180, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(frame, f"Desvio: {np.std(self.ultima_predicao):.4f}",
                   (w-180, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)


class LandmarkFilter:
    """Filtro temporal para suavizar landmarks com oclusão"""
    
    def __init__(self, suavizacao=0.7):
        self.ultimo_valido = None
        self.alpha = suavizacao
        
    def filtrar(self, landmarks_atuais, qualidades):
        """Aplica filtro temporal nos landmarks"""
        if self.ultimo_valido is None:
            self.ultimo_valido = landmarks_atuais.copy()
            return landmarks_atuais
            
        landmarks_filtrados = []
        for i, (atual, qualidade) in enumerate(zip(landmarks_atuais, qualidades)):
            if qualidade > 0.5:  # Landmark confiável
                # Suavização temporal
                novo = self.alpha * atual + (1 - self.alpha) * self.ultimo_valido[i]
                landmarks_filtrados.append(novo)
            else:  # Landmark problemático - usa valor anterior
                landmarks_filtrados.append(self.ultimo_valido[i] * 0.98)  # Leve decay
        
        self.ultimo_valido = landmarks_filtrados.copy()
        return landmarks_filtrados