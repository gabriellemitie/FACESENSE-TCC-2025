#!/usr/bin/env python3
"""
Versão melhorada com tratamento de oclusão e perda de landmarks
"""

import cv2
import mediapipe as mp
import numpy as np
from tensorflow import keras
import time


def avaliar_qualidade_landmarks(landmarks):
    """
    Avalia a qualidade dos landmarks detectados
    Retorna: (qualidade_geral, problemas_detectados)
    """
    problemas = []
    landmarks_validos = 0
    landmarks_criticos_perdidos = 0
    
    # Índices dos landmarks críticos para postura
    landmarks_criticos = [
        11, 12,  # Ombros
        13, 14,  # Cotovelos
        15, 16,  # Punhos/mãos
        23, 24,  # Quadris
        0,       # Nariz
    ]
    
    # Índices específicos das mãos
    landmarks_maos = [15, 16]  # Punhos (representam as mãos)
    
    maos_visiveis = 0
    
    for i, lm in enumerate(landmarks.landmark):
        # Verifica se está dentro da tela
        dentro_tela = 0 <= lm.x <= 1 and 0 <= lm.y <= 1
        
        # Verifica visibilidade se disponível
        visivel = True
        if hasattr(lm, 'visibility'):
            visivel = lm.visibility > 0.5
        
        if dentro_tela and visivel:
            landmarks_validos += 1
            
            # Conta mãos visíveis
            if i in landmarks_maos:
                maos_visiveis += 1
        
        # Verifica landmarks críticos perdidos
        if i in landmarks_criticos and (not dentro_tela or not visivel):
            landmarks_criticos_perdidos += 1
    
    # Calcula qualidade geral
    qualidade_geral = landmarks_validos / 33
    
    # Detecta problemas específicos
    if maos_visiveis == 0:
        problemas.append("Ambas as mãos ocultas")
    elif maos_visiveis == 1:
        problemas.append("Uma mão oculta")
    
    if landmarks_criticos_perdidos > 2:
        problemas.append("Muitos pontos críticos perdidos")
    
    if qualidade_geral < 0.6:
        problemas.append("Qualidade geral baixa")
    
    return qualidade_geral, problemas, maos_visiveis


def interpolar_landmarks(buffer_atual, landmarks_problema):
    """
    Interpola/corrige landmarks problemáticos usando histórico
    """
    if len(buffer_atual) < 3:
        return landmarks_problema
    
    # Pega os últimos 3 frames válidos
    frames_recentes = buffer_atual[-3:]
    
    # Calcula média dos últimos frames para suavizar
    coords_interpolados = []
    for i in range(0, len(landmarks_problema), 3):  # x,y,z por landmark
        x_vals = [frame[i] for frame in frames_recentes]
        y_vals = [frame[i+1] for frame in frames_recentes]
        z_vals = [frame[i+2] for frame in frames_recentes]
        
        # Média móvel simples
        x_interp = np.mean(x_vals)
        y_interp = np.mean(y_vals)
        z_interp = np.mean(z_vals)
        
        coords_interpolados.extend([x_interp, y_interp, z_interp])
    
    return coords_interpolados


def main():
    # MediaPipe setup
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    # Modelo
    try:
        modelo = keras.models.load_model("facesense_posture/models/best_model.keras")
        print(f"✅ Modelo carregado - {modelo.output_shape[1]} classes")
    except Exception as e:
        print(f"❌ Erro: {e}")
        return
    
    # Buffers
    buffer = []
    buffer_validos = []  # Só frames de boa qualidade
    tempos_predicao = []
    max_frames = 500
    min_frames = 50
    
    cap = cv2.VideoCapture(0)
    
    print("=== SISTEMA COM TRATAMENTO DE OCLUSÃO ===")
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
            
            # Avalia qualidade dos landmarks
            qualidade, problemas, maos_visiveis = avaliar_qualidade_landmarks(results.pose_landmarks)
            
            # Extrai coordenadas básicas
            coords = []
            for lm in results.pose_landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])
            
            # Decide como processar baseado na qualidade
            if qualidade >= 0.7:
                # Qualidade boa - usa dados diretos
                buffer.append(coords)
                buffer_validos.append(coords)
                status_frame = "QUALIDADE BOA"
                cor_status = (0, 255, 0)
                
            elif qualidade >= 0.5 and len(buffer_validos) > 0:
                # Qualidade média - interpola com histórico
                coords_corrigidos = interpolar_landmarks(buffer_validos, coords)
                buffer.append(coords_corrigidos)
                status_frame = "INTERPOLADO"
                cor_status = (0, 255, 255)
                
            else:
                # Qualidade ruim - usa último frame válido
                if buffer_validos:
                    buffer.append(buffer_validos[-1])
                    status_frame = "USANDO ÚLTIMO VÁLIDO"
                    cor_status = (0, 165, 255)
                else:
                    buffer.append(coords)  # Sem escolha
                    status_frame = "SEM DADOS VÁLIDOS"
                    cor_status = (0, 0, 255)
            
            # Limita buffers
            if len(buffer) > max_frames:
                buffer.pop(0)
            if len(buffer_validos) > 100:  # Mantém histórico menor de frames válidos
                buffer_validos.pop(0)
            
            # Predição
            if len(buffer) >= min_frames:
                sequence = np.zeros((200, 99))
                frames_para_modelo = buffer[-200:] if len(buffer) >= 200 else buffer
                
                for i, frame_coords in enumerate(frames_para_modelo):
                    if i < 200:
                        sequence[i] = frame_coords
                
                # Classifica
                inicio_pred = time.time()
                pred = modelo.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                tempo_pred = (time.time() - inicio_pred) * 1000
                
                tempos_predicao.append(tempo_pred)
                if len(tempos_predicao) > 100:
                    tempos_predicao.pop(0)
                
                # Resultados
                classe = np.argmax(pred)
                confianca = np.max(pred)
                
                # Interface
                overlay = frame.copy()
                cv2.rectangle(overlay, (10, 10), (450, 250), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                
                # Classe predita
                cv2.putText(frame, f"CLASSE PREDITA: {classe}", (20, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
                
                # Confiança
                cor_conf = (0, 255, 0) if confianca > 0.7 else (0, 255, 255) if confianca > 0.4 else (0, 0, 255)
                cv2.putText(frame, f"CONFIANCA: {confianca:.1%}", (20, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor_conf, 2)
                
                # Status da detecção
                cv2.putText(frame, f"Status: {status_frame}", (20, 130),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor_status, 2)
                
                # Qualidade e problemas
                cv2.putText(frame, f"Qualidade: {qualidade:.1%}", (20, 160),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor_status, 1)
                
                cv2.putText(frame, f"Maos visiveis: {maos_visiveis}/2", (20, 185),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Problemas detectados
                if problemas:
                    prob_texto = ", ".join(problemas[:2])  # Máximo 2 problemas
                    cv2.putText(frame, f"Problemas: {prob_texto}", (20, 210),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 100), 1)
                
                # Info técnica no canto
                cv2.putText(frame, f"Buffer: {len(buffer)}/{max_frames}", (w-200, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
                cv2.putText(frame, f"Validos: {len(buffer_validos)}", (w-200, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
                cv2.putText(frame, f"Tempo: {tempo_pred:.1f}ms", (w-200, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
                
            else:
                cv2.putText(frame, f"Coletando dados... {len(buffer)}/{min_frames}", (20, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                cv2.putText(frame, f"Status: {status_frame}", (20, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor_status, 1)
                
                if problemas:
                    prob_texto = ", ".join(problemas[:2])
                    cv2.putText(frame, f"Problemas: {prob_texto}", (20, 120),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 1)
        
        else:
            cv2.putText(frame, "NENHUMA PESSOA DETECTADA", (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        cv2.putText(frame, "Pressione 'q' para sair", (20, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow('CLASSIFICADOR COM TRATAMENTO DE OCLUSAO', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Finalizado!")


if __name__ == "__main__":
    main()