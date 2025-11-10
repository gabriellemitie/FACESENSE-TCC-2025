## Documentação: Tecnologias e Metodologias utilizadas no projeto FaceSense (TCC)

Este documento descreve, em detalhe, todas as tecnologias, componentes, dependências e metodologias relevantes usadas ao longo do projeto. O objetivo é permitir reprodução, manutenção e evolução do trabalho.

---

## 1. Visão geral do projeto

FaceSense é uma prova-de-conceito que realiza classificação de postura e inferência de stress a partir de vídeo/câmera, com processamento local (on-device) e componentes para geração de relatórios de sessão.

O repositório contém:
- Código Python para experimentos e treinamento (`classificador_robusto.py`, scripts antigos em `archive_unused_scripts/`).
- PoC móvel/cliente Flutter em `flutter_mobile/` e uma aplicação de demonstração em `facesense_posture/`.
- Modelos e artefatos em `facesense_posture/models/`.
- Dados de exportação / históricos em CSV/JSON (`historico_postura.csv`, `historico_sessoes.csv`, `historico_sessoes.json`).
- Script gerador de relatório em `scripts/report_generator.py`.
- Workflow CI para build de APK em `.github/workflows/build_apk.yml`.

## 2. Tecnologias principais (stack)

- Python 3.x
  - Utilizado para pré-processamento, treinamento experimental, scripts utilitários e geração de relatórios.
  - Bibliotecas principais: pandas, numpy, matplotlib, seaborn, python-dateutil. (ver `scripts/requirements.txt`).

- TensorFlow / Keras
  - Treinamento experimental e exportação de modelos (ex.: `best_model.keras` em `facesense_posture/models/`).
  - Modelos convertidos para TensorFlow Lite para execução em dispositivos móveis.

- TensorFlow Lite (TFLite)
  - Versão alvo usada em Android (`org.tensorflow:tensorflow-lite` e, quando necessário, `tensorflow-lite-select-tf-ops`).
  - Obs: Durante desenvolvimento houve problemas de resolução de artefatos em versões mais novas (por ex. 2.19.1) em alguns runners CI; fallback bem sucedido para 2.12.0 foi usado para resolver fetch/compatibilidade.

- Flutter / Dart
  - Interface móvel front-end PoC. Projeto Flutter está sob `flutter_mobile/`.
  - Foi decidido evitar plugins Dart que exigiam legacy Android embedding (ex.: `tflite` e `camera`) para prevenir incompatibilidades, e usar em vez disso uma ponte nativa.

- Kotlin (Android)
  - Ponte nativa implementada usando MethodChannel: `FlexBridge.kt` e `MainActivity.kt` (v2 embedding) dentro do módulo Android do app Flutter.
  - Responsável por carregar o modelo TFLite, configurar (opcionalmente) o delegate Select TF Ops/Flex via reflexão, e executar inferência em tempo real a pedido do Dart.

- Android SDK / NDK / Gradle
  - Build do app Android usa gradle. NDK/CMake instalados em runners quando necessário para empacotar `.so` de delegates.
  - Ajustes em `android/app/build.gradle` adicionam `packagingOptions` para resolver conflitos de bibliotecas nativas.

- GitHub Actions
  - Pipeline criado em `.github/workflows/build_apk.yml` para construir APK debug e publicar artefato (upload-artifact). O workflow também tenta opções de entrega direta (ex.: upload para transfer.sh) como fallback quando o usuário não pode usar Releases/Secrets.

- transfer.sh / curl
  - Usado como fallback para entregar um link público com o APK sem necessidade de configurar secrets de repositório. Observação: disponibilidade pode variar conforme rede do runner.

- Ferramentas de sistema / utilitários
  - `open` (macOS) usado localmente para abrir relatórios.
  - Ferramentas de build Flutter e Android (flutter CLI, sdkmanager).

## 3. Arquitetura e componentes - alto nível

- Camada de aquisição de dados
  - Câmera (no PoC Flutter) → frames → pipeline de extração de features (local ou no dispositivo).

- Modelos
  - Modelos Keras treinados (arquivo `.keras`) convertidos para TFLite (arquivo `.tflite`) para execução on-device.
  - Para operações não suportadas pelo TFLite runtime em algumas builds, o projeto considera o uso do pacote Select TF Ops e o Flex delegate (carregado via Kotlin quando disponível).

- Ponte nativa Android
  - `FlexBridge.kt`: implementa MethodChannel entre Dart e Kotlin para carregar modelo TFLite, executar inferência e devolver resultados ao Flutter.

- Aplicativo Flutter
  - UI e orquestração. Em vez de depender de plugins de terceiros (camera/tflite), o app chama métodos nativos via MethodChannel para reduzir dependências e problemas de embedding.

- Geração de relatórios
  - Script Python `scripts/report_generator.py`: lê CSVs de histórico (`historico_postura.csv`, `historico_sessoes.csv`), calcula métricas (estresse, contagem de alertas, taxa de correção), gera PNGs e um relatório Markdown.

- CI/CD
  - Workflow GitHub Actions para build/debug APK, instalação de SDK/NDK e upload de artefatos.

## 4. Formato de dados (persistência)

- CSV/JSON
  - `historico_postura.csv`: timeline de frames/amostras com colunas de timestamp, rótulos de postura, possivelmente probabilidade de estresse por frame.
  - `historico_sessoes.csv` / `.json`: metadados por sessão (duração, métricas agregadas, timestamps de início/fim, etc.).

- Modelos e assets
  - `facesense_posture/models/` contém `best_model.keras` e `features.json` (descrição de features / mapeamento).
  - Arquivo empacotado `assets/best_model_select.tflite` incluído no Flutter (asset) para inferência móvel.

## 5. Metodologias de desenvolvimento e engenharia

- Remoção de dependências problemáticas
  - Razão: plugins Dart como `tflite` e `camera` arrastavam código ou exigências do Android embedding v1, o que causou builds falhos. Solução: remover dependências e implementar ponte nativa (Kotlin) que usa TFLite diretamente.

- Reprodutibilidade e builds limpos
  - Para evitar inconsistências de dependências locais, foi testado um scaffold Flutter limpo em `/tmp/facesense_clean` para garantir que o projeto pode ser construído em uma máquina limpa e por runners CI.

- Gestão de versões e compatibilidade de TF
  - Testes com diferentes versões do TensorFlow Lite (ex.: problemas com 2.19.1 em alguns runners → fallback para 2.12.0). Registrar versão exata usada em cada ambiente (local/CI) é recomendável.

## 6. Metodologias de ML e avaliação (treinamento / inferência)

- Pipeline experimental (tipicamente):
  1. Coleta de dados (vídeo / frames) com rótulos de postura e, quando disponível, rótulos de estresse (auto-relato, questionário, proxy).
  2. Preprocessamento: detecção de rosto/postura, normalização, extração de features (coord. articulares, ângulos, percentis temporais).
  3. Treinamento: modelos Keras (classificação multi/classe, possivelmente modelos temporais) com divisão treino/validação/teste.
  4. Avaliação: métricas usadas incluem acurácia, matriz de confusão, precision/recall/F1 para classes críticas (alertas posturais). Para a predição de estresse (probabilística), avaliar calibração (calibration curve / Brier score), ROC/AUC quando aplicável.
  5. Conversão: exportar para TFLite (com ou sem optimizações). Se operações custom foram usadas, converter usando Select TF Ops.
  6. Deploy: empacotar `.tflite` no app e carregar via Kotlin para inferência on-device.

- Estratégias para problemas comuns
  - Desbalanceamento de classes: oversampling, undersampling, pesos de perda.
  - Ruído de rótulos: limpeza manual, rótulos fracos com validação cruzada, ensembles.
  - Temporal smoothing: aplicar janelas móveis, médias móveis ou filtros para reduzir falsos positivos de alertas posturais.

## 7. Lógica de alertas e regra de negócio (UX)

- Postura: eventos são tratados como alertas baseados em rótulos ou thresholds; sistema registra timestamp do alerta e aguarda janela de correção (p.ex. 30s) para verificar correção.
- Estresse: o modelo fornece uma probabilidade contínua; UX decide limite operacional (p.ex. 0.6) e também pode usar indicadores temporais (p.ex. percentil médio em 5 minutos) para reduzir ruído.

Recomendações de apresentação ao usuário:
- Mostrar probabilidade, não decisão binária, com explicação breve ("Probabilidade estimada de stress: 74% — considere pausa breve").
- Mostrar histórico/linha do tempo com markers para alertas e correções.

## 8. Privacidade e ética

- Dados sensíveis: video e inferências relacionadas a stress são dados sensíveis; recomenda-se:
  - Armazenar localmente quando possível e criptografar exportações sensíveis.
  - Pedir consentimento explícito para qualquer coleta/exportação/uso de dados para pesquisa.
  - Documentar qualquer dado enviado para servidores externos (neste PoC, processamento é local sempre que possível).

## 9. Reprodutibilidade e como executar o projeto localmente (resumo)

- Pré-requisitos (exemplos):
  - macOS/linux/Windows com Flutter SDK instalado (versão compatível com o projeto), Java JDK (11+ / 17), Android SDK/NDK se for build Android.
  - Python 3.8+ para scripts de análise/treinamento.

- Para gerar relatório localmente:
  - Criar venv, instalar `scripts/requirements.txt` e rodar:
    ```bash
    python3 scripts/report_generator.py --outdir out/report
    ```

- Para construir APK localmente (debug):
  - No diretório do Flutter app (ex.: `flutter_mobile/`) executar:
    ```bash
    flutter pub get
    flutter build apk --debug --no-shrink
    ```
  - OBS: O projeto foi testado com um scaffold limpo; se houver erros de dependência, revisar `pubspec.yaml` e os ajustes no módulo Android (build.gradle) para dependências TFLite.

## 10. Testes, qualidade e CI

- CI: GitHub Actions workflow em `.github/workflows/build_apk.yml`. Funções principais:
  - Instalar Java, Flutter (action), SDK/NDK (sdkmanager) quando necessário;
  - Rodar `flutter build apk --debug --no-shrink` e fazer upload do APK como artifact;
  - Tentativas de upload externo (transfer.sh) como fallback — sujeito a disponibilidade de rede do runner.

- Testes e validações recomendadas:
  - Unit tests Python para scripts utilitários (pandas parsing, métricas).
  - Validações de build Android em runners limpos e verificação de integridade do APK (tamanho, checksum).

## 11. Limitações conhecidas e pontos de atenção

- Dependências nativas (delegates TFLite) podem causar conflitos de empacotamento; usar `packagingOptions` e escolhas de versão cuidadosas.
- Serviços públicos de upload (transfer.sh) podem não ser confiáveis em runners de CI; use Actions artifacts ou Releases (requer PAT/secret) para entrega estável.
- Modelos com operadores não suportados pelo runtime padrão precisam do Select TF Ops / Flex delegate — isso aumenta o tamanho e complexidade do build.

## 12. Arquivos e caminhos relevantes (resumo rápido)

- Raiz:
  - `classificador_robusto.py` — código experimental local.
  - `historico_postura.csv`, `historico_sessoes.csv`, `historico_sessoes.json` — export CSV/JSON.

- `facesense_posture/`:
  - `app.py` — app de demonstração (Flask/Streamlit/CLI conforme conteúdo).
  - `models/` — `best_model.keras`, `features.json`, possivelmente `best_model_select.tflite`.

- `flutter_mobile/`:
  - `pubspec.yaml` — assets/dep.
  - `android/app/src/main/kotlin/.../FlexBridge.kt` e `MainActivity.kt` — ponte nativa Kotlin.

- `scripts/`:
  - `report_generator.py` — gera relatórios Markdown + PNG a partir de CSVs.
  - `requirements.txt` — dependências Python.

- CI:
  - `.github/workflows/build_apk.yml` — build + upload artifact.

## 13. Próximos passos recomendados

1. Documentar versões exatas de todas as dependências críticas (TensorFlow/TFLite, Flutter, Dart, JDK) em um arquivo `ENV.md` ou no `README.md` para garantir reprodutibilidade.
2. Adicionar testes unitários para `scripts/report_generator.py` e um exemplo de CSV de teste em `tests/fixtures/`.
3. Se for necessária entrega pública do APK sem login GitHub, considerar um servidor de arquivos próprio (S3 com links temporários) ou configurar um pipeline de Releases com `REPO_PAT` (secret) quando for possível.
4. Definir e documentar a política de privacidade / consentimento para coleta de vídeo/estresse.

---

Se desejar, eu posso:
- Gerar um `ENV.md` com comandos de instalação e versões testadas (ex.: versão Flutter e TF).
- Inserir links/trechos de código diretamente na documentação para mostrar o `FlexBridge.kt` e `report_generator.py` (resumidos).
- Executar `scripts/report_generator.py` com os CSVs do repositório e anexar as métricas geradas e imagens (se quiser que eu rode agora, diga para executar). 

FIM.
