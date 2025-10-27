#!/usr/bin/env python3
"""
Script para mostrar APENAS a saída do modelo Keras em tempo real
"""

import cv2
import mediapipe as mp
import numpy as np
from tensorflow import keras
import os


def main():
    # Configuração MediaPipe
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    # Carrega modelo
    modelo_path = "facesense_posture/models/stress.keras"
    try:
        modelo = keras.models.load_model(modelo_path)
        print(f"✅ Modelo carregado: {modelo.input_shape}")
    except Exception as e:
        print(f"❌ Erro: {e}")
        return
    
    # Buffer para sequência temporal (200 frames, 99 features cada)
    buffer = []
    
    # Câmera
    cap = cv2.VideoCapture(0)
    
    print("=== MOSTRANDO SAÍDA BRUTA DO MODELO ===")
    print("Pressione 'q' para sair")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Espelha e processa
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        
        # Desenha landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Extrai coordenadas (33 landmarks × 3 coords = 99 features)
            coords = []
            for lm in results.pose_landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])
            
            # Adiciona ao buffer
            buffer.append(coords)
            if len(buffer) > 200:
                buffer.pop(0)
            
            # Faz predição quando tiver dados suficientes
            if len(buffer) >= 50:  # Mínimo de frames
                # Cria sequência (200, 99)
                sequence = np.zeros((200, 99))
                for i, frame_coords in enumerate(buffer[-200:]):
                    sequence[i] = frame_coords
                
                # Predição
                input_data = np.expand_dims(sequence, axis=0)  # (1, 200, 99)
                pred = modelo.predict(input_data, verbose=0)[0]
                
                # CLASSE PREDITA (PRINCIPAL)
                classe_predita = np.argmax(pred)
                confianca_max = np.max(pred)
                
                # Título principal
                cv2.putText(frame, "=== CLASSE PREDITA ===", (10, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Classe predita em destaque
                cv2.putText(frame, f"CLASSE: {classe_predita}", (10, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
                
                # Confiança
                cv2.putText(frame, f"CONFIANCA: {confianca_max:.1%}", (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                cv2.putText(frame, f"VALOR: {confianca_max:.6f}", (10, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                
                # TOP 3 CLASSES
                cv2.putText(frame, "TOP 3 CLASSES:", (10, 190), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 2)
                
                # Ordena classes por probabilidade
                indices_ordenados = np.argsort(pred)[::-1]  # Do maior para menor
                
                for i, idx in enumerate(indices_ordenados[:3]):
                    y_pos = 220 + i * 25
                    prob = pred[idx]
                    cv2.putText(frame, f"{i+1}. Classe {idx}: {prob:.1%}", (10, y_pos), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Info adicional no canto
                cv2.putText(frame, f"Total classes: {len(pred)}", (350, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
                cv2.putText(frame, f"Frames: {len(buffer)}/200", (350, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
                
            else:
                cv2.putText(frame, f"Coletando... {len(buffer)}/50", (10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        else:
            cv2.putText(frame, "NENHUMA PESSOA DETECTADA", (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        # Instrução
        cv2.putText(frame, "Pressione 'q' para sair", (10, frame.shape[0]-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('SAIDA BRUTA DO MODELO KERAS', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Finalizado.")


if __name__ == "__main__":
    main()