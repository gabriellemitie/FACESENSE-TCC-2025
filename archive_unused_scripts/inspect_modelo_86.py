#!/usr/bin/env python3
"""
Inspeção do modelo _86.keras para verificar suas características
"""

import numpy as np
from tensorflow import keras

def inspecionar_modelo(model_path):
    print(f"=== INSPEÇÃO DO MODELO: {model_path} ===\n")
    
    try:
        # Carrega o modelo
        modelo = keras.models.load_model(model_path)
        print(f"✅ Modelo carregado com sucesso!")
        
        # Informações básicas
        print(f"\n📊 INFORMAÇÕES BÁSICAS:")
        print(f"   • Input shape: {modelo.input_shape}")
        print(f"   • Output shape: {modelo.output_shape}")
        print(f"   • Número de classes: {modelo.output_shape[1]}")
        print(f"   • Número de parâmetros: {modelo.count_params():,}")
        
        # Arquitetura resumida
        print(f"\n🏗️ ARQUITETURA:")
        modelo.summary()
        
        # Teste com dados dummy
        print(f"\n🧪 TESTE COM DADOS DUMMY:")
        if len(modelo.input_shape) == 3:  # (None, timesteps, features)
            timesteps = modelo.input_shape[1] or 200
            features = modelo.input_shape[2] or 99
            dummy_input = np.random.random((1, timesteps, features))
            print(f"   • Input teste: {dummy_input.shape}")
        else:
            print(f"   • Input shape não reconhecido: {modelo.input_shape}")
            return
        
        # Predição teste
        pred = modelo.predict(dummy_input, verbose=0)
        print(f"   • Output teste: {pred.shape}")
        print(f"   • Valores de output: {pred[0][:5]}... (primeiros 5)")
        print(f"   • Soma das probabilidades: {np.sum(pred[0]):.4f}")
        print(f"   • Classe predita: {np.argmax(pred[0])}")
        print(f"   • Confiança máxima: {np.max(pred[0]):.4f}")
        
        # Comparação com o modelo anterior
        print(f"\n🔄 COMPARAÇÃO:")
        try:
            modelo_antigo = keras.models.load_model("facesense_posture/models/stress.keras")
            print(f"   • Modelo anterior: {modelo_antigo.input_shape} -> {modelo_antigo.output_shape}")
            print(f"   • Novo modelo: {modelo.input_shape} -> {modelo.output_shape}")
            
            if modelo.input_shape == modelo_antigo.input_shape:
                print(f"   • ✅ Input shapes compatíveis")
            else:
                print(f"   • ⚠️ Input shapes diferentes - pode precisar ajustar código")
                
            if modelo.output_shape == modelo_antigo.output_shape:
                print(f"   • ✅ Output shapes compatíveis")
            else:
                print(f"   • ⚠️ Output shapes diferentes - número de classes mudou")
        except:
            print(f"   • ⚠️ Não foi possível carregar modelo anterior para comparação")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        return False

if __name__ == "__main__":
    # Inspeciona o novo modelo
    sucesso = inspecionar_modelo("facesense_posture/models/best_model.keras")
    
    if sucesso:
        print(f"\n🎯 RESUMO:")
        print(f"   • Modelo best_model.keras carregado e funcionando")
        print(f"   • Pronto para uso no sistema FACESENSE")
        print(f"   • Execute o app.py para testar!")
    else:
        print(f"\n❌ PROBLEMAS ENCONTRADOS:")
        print(f"   • Verifique se o arquivo existe e está íntegro")
        print(f"   • Pode ser necessário usar o modelo anterior")