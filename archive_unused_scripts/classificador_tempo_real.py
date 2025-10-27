#!/usr/bin/env python3
"""
Script final: Mostra classe predita do modelo em tempo real
"""

import cv2
import mediapipe as mp
import numpy as np
from tensorflow import keras
import time


def main():
    # MediaPipe
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    # Modelo
    try:
    modelo = keras.models.load_model("facesense_posture/models/stress.keras")
    print(f"✅ Modelo stress.keras carregado - output shape={modelo.output_shape}")
    except Exception as e:
        print(f"❌ Erro: {e}")
        return
    
    # Buffer temporal - AUMENTADO PARA MAIS FRAMES
    buffer = []
    max_frames = 500  # Aumentado de 200 para 500 frames
    min_frames = 50   # Mínimo para começar predições
    
    cap = cv2.VideoCapture(0)
    
    print(f"Buffer aumentado para {max_frames} frames (mín: {min_frames})")
    print("Pressione 'q' para sair")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        
        if results.pose_landmarks:
            # Desenha skeleton
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Extrai landmarks (33 pontos x 3 coordenadas = 99 features)
            coords = []
            for lm in results.pose_landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])
            
            buffer.append(coords)
            if len(buffer) > max_frames:
                buffer.pop(0)
            
            # Predição quando tiver dados suficientes
            if len(buffer) >= min_frames:
                # Sequência temporal - usa os últimos 200 frames para o modelo
                sequence = np.zeros((200, 99))
                
                # Seleciona os 200 frames mais recentes do buffer maior
                frames_para_modelo = buffer[-200:] if len(buffer) >= 200 else buffer
                
                for i, frame_coords in enumerate(frames_para_modelo):
                    if i < 200:  # Garante que não exceda o limite do modelo
                        sequence[i] = frame_coords
                
                # Predição com cronometragem
                inicio_pred = time.time()
                pred = modelo.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                tempo_pred = (time.time() - inicio_pred) * 1000  # em milissegundos
                
                # === INFORMAÇÕES PRINCIPAIS ===
                classe = np.argmax(pred)
                confianca = np.max(pred)
                
                # Fundo semi-transparente para texto
                overlay = frame.copy()
                cv2.rectangle(overlay, (10, 10), (400, 200), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                
                # CLASSE PREDITA (destaque)
                cv2.putText(frame, f"CLASSE PREDITA: {classe}", (20, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
                
                # Confiança com cor baseada no valor
                cor_conf = (0, 255, 0) if confianca > 0.7 else (0, 255, 255) if confianca > 0.4 else (0, 0, 255)
                cv2.putText(frame, f"CONFIANCA: {confianca:.1%}", (20, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor_conf, 2)
                
                # Top 3 classes
                cv2.putText(frame, "TOP 3:", (20, 130),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                top_indices = np.argsort(pred)[-3:][::-1]
                for i, idx in enumerate(top_indices):
                    y_pos = 150 + i * 20
                    cv2.putText(frame, f"{i+1}. Classe {idx}: {pred[idx]:.1%}", (25, y_pos),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Status no canto superior direito
                cv2.putText(frame, f"Buffer: {len(buffer)}/{max_frames}", (w-200, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
                cv2.putText(frame, f"Classes: {len(pred)}", (w-200, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
                
                # Indicador de qualidade baseado no buffer
                if len(buffer) >= 200:
                    quality = "OTIMO"
                    color = (0, 255, 0)
                elif len(buffer) >= 100:
                    quality = "BOM"
                    color = (0, 255, 255)
                else:
                    quality = "BASICO"
                    color = (0, 165, 255)
                
                cv2.putText(frame, f"Qualidade: {quality}", (w-200, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
            else:
                # Ainda coletando dados
                cv2.putText(frame, f"Coletando dados... {len(buffer)}/{min_frames}", (20, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                # Barra de progresso
                progress = len(buffer) / min_frames
                bar_width = 300
                cv2.rectangle(frame, (20, 70), (20 + int(bar_width * progress), 90), (0, 255, 0), -1)
                cv2.rectangle(frame, (20, 70), (20 + bar_width, 90), (255, 255, 255), 2)
        
        else:
            cv2.putText(frame, "NENHUMA PESSOA DETECTADA", (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        # Instruções
        cv2.putText(frame, "Pressione 'q' para sair", (20, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow('CLASSIFICACAO DE POSTURA EM TEMPO REAL', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Finalizado!")


if __name__ == "__main__":
    main()