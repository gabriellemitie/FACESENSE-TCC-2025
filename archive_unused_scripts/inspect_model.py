#!/usr/bin/env python3
"""
Script para inspecionar a estrutura do modelo Keras
"""

from tensorflow import keras
import numpy as np

def inspecionar_modelo():
    modelo_path = "facesense_posture/models/stress.keras"
    
    try:
        # Carrega o modelo
        modelo = keras.models.load_model(modelo_path)
        
        print("=== INFORMAÇÕES DO MODELO ===")
        print(f"Input shape: {modelo.input_shape}")
        print(f"Output shape: {modelo.output_shape}")
        print()
        
        # Mostra resumo do modelo
        print("=== RESUMO DA ARQUITETURA ===")
        modelo.summary()
        print()
        
        # Testa com dados dummy
        print("=== TESTE COM DADOS DUMMY ===")
        # Cria entrada fake no formato esperado (1, 200, 99)
        dummy_input = np.random.random((1, 200, 99))
        
        # Faz predição
        predicao = modelo.predict(dummy_input, verbose=1)
        
        print(f"Shape da predição: {predicao.shape}")
        print(f"Tipo da predição: {type(predicao)}")
        print(f"Valores da predição: {predicao}")
        print(f"Número de saídas: {predicao.shape[1] if len(predicao.shape) > 1 else 1}")
        
        # Se for classificação, mostra probabilidades
        if len(predicao.shape) > 1 and predicao.shape[1] > 1:
            print(f"\n=== ANÁLISE DE CLASSIFICAÇÃO ===")
            print(f"Número de classes: {predicao.shape[1]}")
            for i, prob in enumerate(predicao[0]):
                print(f"Classe {i}: {prob:.6f} ({prob*100:.2f}%)")
            print(f"Classe predita: {np.argmax(predicao[0])}")
            print(f"Confiança máxima: {np.max(predicao[0]):.6f}")
        
    except Exception as e:
        print(f"Erro ao carregar modelo: {e}")

if __name__ == "__main__":
    inspecionar_modelo()