#!/usr/bin/env python3
"""
Script para inspecionar o modelo best_model.keras
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np

def inspecionar_modelo():
    print("🔍 Inspecionando modelo best_model.keras...")
    
    try:
        modelo = keras.models.load_model("facesense_posture/models/best_model.keras")
        
        print(f"✅ Modelo carregado com sucesso!")
        print(f"📊 Resumo do modelo:")
        modelo.summary()
        
        print(f"\n📝 Detalhes:")
        print(f"Input shape: {modelo.input_shape}")
        print(f"Output shape: {modelo.output_shape}")
        print(f"Número de parâmetros: {modelo.count_params()}")
        
        # Testa com dados dummy
        if modelo.input_shape[1:] == (200, 99):
            print(f"\n🧪 Testando com input (200, 99):")
            dummy_input = np.random.random((1, 200, 99))
            prediction = modelo.predict(dummy_input, verbose=0)
            print(f"Prediction shape: {prediction.shape}")
            print(f"Prediction sample: {prediction[0]}")
            
        return modelo
        
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    inspecionar_modelo()