#!/usr/bin/env python3
"""
Script simples para capturar landmarks com MediaPipe e classificar postura com modelo Keras
"""

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os


class PosturaClassifier:
    def __init__(self, modelo_path="facesense_posture/models/stress.keras"):
        # MediaPipe setup
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Carrega modelo Keras
        self.modelo = None
        self.carregar_modelo(modelo_path)
        
        # Buffer para sequência temporal
        self.sequence_buffer = []
        self.sequence_length = 200  # Baseado no input shape do modelo
        
        # Camera
        self.cap = cv2.VideoCapture(0)
        
    def carregar_modelo(self, modelo_path):
        """Carrega o modelo Keras"""
        try:
            if os.path.exists(modelo_path):
                self.modelo = keras.models.load_model(modelo_path)
                print(f"✅ Modelo carregado: {modelo_path}")
                print(f"Formato de entrada esperado: {self.modelo.input_shape}")
            else:
                print(f"❌ Modelo não encontrado: {modelo_path}")
        except Exception as e:
            print(f"❌ Erro ao carregar modelo: {e}")
    
    def extrair_landmarks(self, landmarks):
        """Extrai landmarks como array numpy"""
        coords = []
        for landmark in landmarks.landmark:
            coords.extend([landmark.x, landmark.y, landmark.z])
        return np.array(coords)
    
    def adicionar_ao_buffer(self, landmarks_array):
        """Adiciona landmarks ao buffer temporal"""
        self.sequence_buffer.append(landmarks_array)
        
        # Mantém apenas os últimos sequence_length frames
        if len(self.sequence_buffer) > self.sequence_length:
            self.sequence_buffer.pop(0)
    
    def preparar_dados_modelo(self):
        """Prepara sequência temporal para o modelo"""
        if self.modelo is None or len(self.sequence_buffer) == 0:
            return None
        
        # Cria sequência de 200 frames
        if len(self.sequence_buffer) >= self.sequence_length:
            # Usa os últimos 200 frames
            sequence = np.array(self.sequence_buffer[-self.sequence_length:])
        else:
            # Preenche com zeros se não tiver frames suficientes
            sequence = np.zeros((self.sequence_length, 99))
            for i, frame in enumerate(self.sequence_buffer):
                sequence[i] = frame
        
        # Adiciona dimensão do batch: (1, 200, 99)
        return np.expand_dims(sequence, axis=0)
    
    def classificar_postura(self, landmarks):
        """Obtém saída bruta do modelo sem interpretação"""
        if self.modelo is None:
            return "Modelo não carregado", []
            
        try:
            # Extrai landmarks
            landmarks_array = self.extrair_landmarks(landmarks)
            
            # Adiciona ao buffer temporal
            self.adicionar_ao_buffer(landmarks_array)
            
            # Prepara sequência temporal
            dados = self.preparar_dados_modelo()
            if dados is None:
                return "Buffer insuficiente", []
            
            # Faz predição - APENAS RETORNA A SAÍDA DO MODELO
            predicao = self.modelo.predict(dados, verbose=0)[0]
            
            # Retorna a predição bruta sem interpretação
            return "PREDICAO", predicao
            
        except Exception as e:
            print(f"Erro na classificação: {e}")
            return "Erro", []
    
    def executar(self):
        """Loop principal"""
        print("Iniciando captura de postura...")
        print("Pressione 'q' para sair")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Espelha horizontalmente
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            # Converte para RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Processa com MediaPipe
            results = self.pose.process(rgb_frame)
            
            # Desenha landmarks
            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
                )
                
                # Obtém saída do modelo
                status, predicao = self.classificar_postura(results.pose_landmarks)
                
                # Cor fixa para visualização
                cor = (255, 255, 255)  # Branco
                
                # Mostra saída bruta do modelo
                cv2.putText(frame, "SAIDA DO MODELO:", (20, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor, 2)
                
                if isinstance(predicao, np.ndarray):
                    # Mostra cada valor da predição
                    for i, valor in enumerate(predicao[:5]):  # Máximo 5 valores
                        texto = f"[{i}]: {valor:.6f}"
                        cv2.putText(frame, texto, (20, 80 + i * 25),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 1)
                    
                    # Mostra forma e tamanho
                    cv2.putText(frame, f"Shape: {predicao.shape}", (20, 200),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 1)
                    cv2.putText(frame, f"Min: {predicao.min():.6f}", (20, 220),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 1)
                    cv2.putText(frame, f"Max: {predicao.max():.6f}", (20, 240),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 1)
                
            else:
                # Sem pessoa detectada
                cv2.putText(frame, "NENHUMA PESSOA DETECTADA", (20, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            
            # Instruções
            cv2.putText(frame, "Pressione 'q' para sair", (20, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Mostra frame
            cv2.imshow('Classificacao de Postura - MediaPipe + Keras', frame)
            
            # Verifica tecla pressionada
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print("Programa encerrado.")


def main():
    """Função principal"""
    try:
        classifier = PosturaClassifier()
        classifier.executar()
    except KeyboardInterrupt:
        print("\nPrograma interrompido pelo usuário")
    except Exception as e:
        print(f"Erro: {e}")


if __name__ == "__main__":
    main()