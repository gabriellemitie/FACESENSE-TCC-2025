"""Script de teste standalone para PostureMonitor

Use este script para testar o monitor de postura sem abrir a UI.
Ele faz:
 - instancia PostureMonitor
 - calibra por alguns segundos (configurável)
 - reduz janelas (fila_postura) para acelerar detecção em testes
 - diminui ALERTA_SEGUNDOS_POSTURA para testes rápidos
 - executa um loop chamando processar_frame() e imprimindo status/dica

Rode no mesmo venv usado pelo app:
python3 scripts/test_posture_standalone.py

Interrompa com Ctrl+C.
"""

import time
from collections import deque
import importlib.util
from pathlib import Path

# Resolve o caminho do módulo com base na raiz do repositório (um nível acima deste scripts/)
repo_root = Path(__file__).resolve().parent.parent
module_path = repo_root / 'facesense_posture' / 'modules' / 'posture_model.py'
if not module_path.exists():
    raise FileNotFoundError(f"posture_model.py não encontrado em: {module_path}")

spec = importlib.util.spec_from_file_location('pm', str(module_path))
pm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pm)

def main():
    Mon = pm.PostureMonitor
    monitor = Mon()

    print('-> Calibrando por 3s. Mantenha postura neutra (reto).')
    try:
        monitor.calibrar(duracao=3)
    except Exception as e:
        print('Falha na calibração (talvez a câmera esteja em uso):', e)
        return

    # Ajustes para teste rápido
    monitor.ALERTA_SEGUNDOS_POSTURA = 3
    monitor.fila_postura = deque(maxlen=10)  # janela menor para testes

    print('-> Iniciando captura (CTRL+C para parar).')
    monitor.iniciar()

    try:
        while True:
            img, status, dica, cor = monitor.processar_frame()
            now = time.time()
            print(f'[{now:.1f}] status={status} dica="{dica}"')
            # o próprio monitor já emite DEBUGs periodicamente
            time.sleep(0.12)
    except KeyboardInterrupt:
        print('\nInterrompido pelo usuário. Parando monitor...')
    except Exception as e:
        print('Erro no loop:', e)
    finally:
        try:
            monitor.parar()
        except Exception:
            pass

if __name__ == '__main__':
    main()
