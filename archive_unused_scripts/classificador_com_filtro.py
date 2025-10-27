#!/usr/bin/env python3
"""
VERSÃO MELHORADA: Lida com oclusão das mãos e landmarks perdidos
"""

import cv2
import mediapipe as mp
import numpy as np
from tensorflow import keras
import time


class LandmarkFilter:
    """Filtro para suavizar e corrigir landmarks problemáticos"""
    
    def __init__(self, suavizacao=0.7):
        self.ultimo_valido = None
        self.alpha = suavizacao  # Fator de suavização
        
    def filtrar(self, landmarks_atuais, qualidades):
        """Aplica filtro temporal nos landmarks"""
        if self.ultimo_valido is None:
            self.ultimo_valido = landmarks_atuais.copy()
            return landmarks_atuais
            
        # Suavização temporal
        landmarks_filtrados = []
        for i, (atual, qualidade) in enumerate(zip(landmarks_atuais, qualidades)):
            if qualidade > 0.5:  # Landmark confiável
                # Suavização: novo = alpha * atual + (1-alpha) * anterior
                novo = self.alpha * atual + (1 - self.alpha) * self.ultimo_valido[i]
                landmarks_filtrados.append(novo)
            else:  # Landmark problemático
                # Usa valor anterior com pequena correção
                landmarks_filtrados.append(self.ultimo_valido[i] * 0.98)  # Leve decay
        
        self.ultimo_valido = landmarks_filtrados.copy()
        return landmarks_filtrados


def main():
    # MediaPipe
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    # Modelo
    try:
    modelo = keras.models.load_model("facesense_posture/models/stress.keras")
        print(f"✅ Modelo carregado - {modelo.output_shape[1]} classes")
    except Exception as e:
        print(f"❌ Erro: {e}")
        return
    
    # Filtro de landmarks
    filtro = LandmarkFilter(suavizacao=0.8)
    
    # Buffers
    buffer = []
    max_frames = 500
    min_frames = 50
    tempos_predicao = []
    
    cap = cv2.VideoCapture(0)
    
    print("🔧 MELHORIAS IMPLEMENTADAS:")
    print("  • Filtro temporal para landmarks")
    print("  • Correção automática de oclusões")  
    print("  • Detecção visual de mãos problemáticas")
    print("  • Suavização de transições")
    print("\n Pressione 'q' para sair")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        
        if results.pose_landmarks:
            # === ANÁLISE DE QUALIDADE DOS LANDMARKS ===
            landmarks_raw = []
            qualidades = []
            problemas_maos = 0
            
            for i, lm in enumerate(results.pose_landmarks.landmark):
                landmarks_raw.extend([lm.x, lm.y, lm.z])
                
                # Avalia qualidade do landmark
                visibility = getattr(lm, 'visibility', 1.0)
                
                # Landmarks das mãos (15-22)
                if i in [15, 16, 17, 18, 19, 20, 21, 22]:
                    if visibility < 0.6 or lm.z > 0.8:
                        qualidades.extend([0.2, 0.2, 0.2])  # Baixa qualidade
                        problemas_maos += 1
                    else:
                        qualidades.extend([visibility, visibility, visibility])
                else:
                    # Outros landmarks
                    qualidades.extend([visibility, visibility, visibility])
            
            # === APLICAÇÃO DO FILTRO ===
            landmarks_filtrados = filtro.filtrar(landmarks_raw, qualidades)
            
            # === DESENHO VISUAL ===
            # Skeleton básico
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Indicadores visuais das mãos
            landmarks_maos = [15, 16, 17, 18, 19, 20, 21, 22]
            for i in landmarks_maos:
                lm = results.pose_landmarks.landmark[i]
                x, y = int(lm.x * w), int(lm.y * h)
                visibility = getattr(lm, 'visibility', 1.0)
                
                if 0 <= x < w and 0 <= y < h:
                    if visibility < 0.6 or lm.z > 0.8:
                        # Landmark problemático
                        cv2.circle(frame, (x, y), 10, (0, 0, 255), 2)
                        cv2.putText(frame, "!", (x-5, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    else:
                        # Landmark bom
                        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            
            # === ADIÇÃO AO BUFFER ===
            qualidade_geral = sum(1 for q in qualidades if q > 0.5) / len(qualidades)
            
            buffer.append(landmarks_filtrados)
            if len(buffer) > max_frames:
                buffer.pop(0)
            
            # === PREDIÇÃO ===
            if len(buffer) >= min_frames:
                # Prepara sequência
                sequence = np.zeros((200, 99))
                frames_recentes = buffer[-200:] if len(buffer) >= 200 else buffer
                
                for i, frame_coords in enumerate(frames_recentes):
                    if i < 200:
                        sequence[i] = frame_coords
                
                # Predição com tempo
                inicio = time.time()
                pred = modelo.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                tempo_pred = (time.time() - inicio) * 1000
                
                tempos_predicao.append(tempo_pred)
                if len(tempos_predicao) > 50:
                    tempos_predicao.pop(0)
                
                # === INFORMAÇÕES NA TELA ===
                classe = np.argmax(pred)
                confianca = np.max(pred)
                
                # Fundo para texto
                overlay = frame.copy()
                cv2.rectangle(overlay, (10, 10), (400, 220), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                
                # Informações principais
                cv2.putText(frame, f"CLASSE: {classe}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
                cv2.putText(frame, f"CONF: {confianca:.1%}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Qualidade dos landmarks
                cor_qual = (0, 255, 0) if qualidade_geral > 0.8 else (0, 255, 255) if qualidade_geral > 0.6 else (0, 0, 255)
                cv2.putText(frame, f"Qualidade landmarks: {qualidade_geral:.1%}", (20, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor_qual, 1)
                
                # Status das mãos
                if problemas_maos > 0:
                    cv2.putText(frame, f"Maos ocultas: {problemas_maos}/8", (20, 140), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                    cv2.putText(frame, "FILTRO ATIVO", (20, 165), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)
                else:
                    cv2.putText(frame, "Landmarks OK", (20, 140), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                
                # Top 3 classes
                top_indices = np.argsort(pred)[-3:][::-1]
                for i, idx in enumerate(top_indices):
                    cv2.putText(frame, f"{i+1}. Classe {idx}: {pred[idx]:.1%}", (20, 190 + i*15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                
                # Info lateral
                cv2.putText(frame, f"Tempo: {tempo_pred:.1f}ms", (w-180, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
                cv2.putText(frame, f"Buffer: {len(buffer)}/{max_frames}", (w-180, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
                
                if tempos_predicao:
                    tempo_medio = np.mean(tempos_predicao)
                    cv2.putText(frame, f"Media: {tempo_medio:.1f}ms", (w-180, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            else:
                # Coletando dados
                progress = len(buffer) / min_frames
                cv2.putText(frame, f"Coletando... {len(buffer)}/{min_frames}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.rectangle(frame, (20, 70), (20 + int(300 * progress), 90), (0, 255, 0), -1)
                cv2.rectangle(frame, (20, 70), (320, 90), (255, 255, 255), 2)
        
        else:
            cv2.putText(frame, "NENHUMA PESSOA DETECTADA", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        # Instruções
        cv2.putText(frame, "Pressione 'q' para sair", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow('CLASSIFICADOR COM FILTRO DE OCLUSAO', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Finalizado!")


if __name__ == "__main__":
    main()