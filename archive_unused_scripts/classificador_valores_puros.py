#!/usr/bin/env python3
"""
Classificador que mostra apenas os valores PUROS da predição do modelo
SEM interpretação de postura boa/ruim
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
    
    # Modelo de postura (best_model.keras)
    try:
        modelo = keras.models.load_model("facesense_posture/models/best_model.keras")
        print(f"✅ Modelo best_model.keras carregado - output shape={modelo.output_shape}")
    except Exception as e:
        print(f"❌ Erro: {e}")
        return
    
    # Buffers
    buffer = []
    max_frames = 500
    min_frames = 50
    tempos_predicao = []
    
    cap = cv2.VideoCapture(0)
    
    print("🎯 CLASSIFICADOR - VALORES PUROS DO MODELO")
    print("  • Mostra apenas a saída bruta do modelo")
    print("  • SEM interpretação de postura")
    print("  • 29 classes disponíveis (0-28)")
    print("\nPressione 'q' para sair")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        
        if results.pose_landmarks:
            # Desenha skeleton básico
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
                # Prepara sequência temporal
                sequence = np.zeros((200, 99))
                frames_recentes = buffer[-200:] if len(buffer) >= 200 else buffer
                
                for i, frame_coords in enumerate(frames_recentes):
                    if i < 200:
                        sequence[i] = frame_coords
                
                # Predição com cronometragem
                inicio = time.time()
                pred = modelo.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                tempo_pred = (time.time() - inicio) * 1000
                
                tempos_predicao.append(tempo_pred)
                if len(tempos_predicao) > 30:
                    tempos_predicao.pop(0)
                
                # === VALORES PUROS DO MODELO ===
                classe_maxima = np.argmax(pred)
                confianca_maxima = np.max(pred)
                
                # Fundo para informações
                overlay = frame.copy()
                cv2.rectangle(overlay, (10, 10), (500, 350), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
                
                # CABEÇALHO
                cv2.putText(frame, "VALORES PUROS DO MODELO", (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                # CLASSE COM MAIOR PROBABILIDADE
                cv2.putText(frame, f"CLASSE MAXIMA: {classe_maxima}", (20, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 3)
                cv2.putText(frame, f"PROBABILIDADE: {confianca_maxima:.4f} ({confianca_maxima:.2%})", 
                           (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # TOP 10 CLASSES COM MAIORES PROBABILIDADES
                cv2.putText(frame, "TOP 10 CLASSES:", (20, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                top_indices = np.argsort(pred)[-10:][::-1]  # Top 10
                for i, idx in enumerate(top_indices):
                    y_pos = 175 + i * 18
                    probabilidade = pred[idx]
                    
                    # Cor baseada na probabilidade
                    if probabilidade > 0.1:
                        cor = (0, 255, 0)  # Verde para probabilidades altas
                    elif probabilidade > 0.05:
                        cor = (0, 255, 255)  # Amarelo para médias
                    else:
                        cor = (255, 255, 255)  # Branco para baixas
                    
                    cv2.putText(frame, f"{i+1:2d}. Classe {idx:2d}: {probabilidade:.4f} ({probabilidade:.1%})",
                               (25, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, cor, 1)
                
                # INFORMAÇÕES TÉCNICAS
                cv2.putText(frame, f"Buffer: {len(buffer)}/{max_frames}", (w-200, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
                cv2.putText(frame, f"Frames usados: {len(frames_recentes)}/200", (w-200, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
                cv2.putText(frame, f"Tempo predição: {tempo_pred:.1f}ms", (w-200, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
                
                if tempos_predicao:
                    tempo_medio = np.mean(tempos_predicao)
                    cv2.putText(frame, f"Tempo médio: {tempo_medio:.1f}ms", (w-200, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                # ESTATÍSTICAS DA DISTRIBUIÇÃO
                cv2.putText(frame, f"Soma total: {np.sum(pred):.4f}", (w-200, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                cv2.putText(frame, f"Média: {np.mean(pred):.4f}", (w-200, 140),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                cv2.putText(frame, f"Desvio: {np.std(pred):.4f}", (w-200, 160),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                
            else:
                # Ainda coletando dados
                progress = len(buffer) / min_frames
                cv2.putText(frame, f"Coletando dados... {len(buffer)}/{min_frames}", (20, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                # Barra de progresso
                bar_width = 400
                cv2.rectangle(frame, (20, 80), (20 + int(bar_width * progress), 100), (0, 255, 0), -1)
                cv2.rectangle(frame, (20, 80), (20 + bar_width, 100), (255, 255, 255), 2)
                cv2.putText(frame, f"{progress:.1%}", (230, 95), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        else:
            cv2.putText(frame, "NENHUMA PESSOA DETECTADA", (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.putText(frame, "Posicione-se na frente da camera", (20, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Instruções
        cv2.putText(frame, "VALORES PUROS - SEM INTERPRETACAO | Pressione 'q' para sair", 
                   (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('CLASSIFICADOR - VALORES PUROS DO MODELO', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Finalizado!")


if __name__ == "__main__":
    main()