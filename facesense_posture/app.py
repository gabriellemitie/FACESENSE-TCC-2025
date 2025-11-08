import sys
import time
import platform
import pandas as pd
import cv2
import numpy as np


from PySide6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QHBoxLayout,
    QMessageBox, QTabWidget, QSlider, QFormLayout, QCheckBox
)
from PySide6.QtGui import QImage, QPixmap, QFont
from PySide6.QtCore import QTimer, Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from modules.posture_model import PostureMonitor
# ===== IMPORTA O MÓDULO DE MICROGESTURE =====
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from modules.microgesture_model import MicrogestureMonitor


class PosturaApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(" FACESENSE - Monitor Integrado (Microgesture + Postura)")
        self.setGeometry(200, 100, 1200, 750)
        self.historico_sessoes = []  # sempre conterá 'dados'
        self.historico_csv_path = "historico_sessoes.csv"
        self.historico_json_path = "historico_sessoes.json"
        self.carregar_historico_sessoes()


        # ======== ABAS PRINCIPAIS ========
        self.tabs = QTabWidget()
        self.tab_monitor = QWidget()
        self.tab_historico = QWidget()
        self.tab_config = QWidget()

        self.tabs.addTab(self.tab_monitor, "🧍 Monitoramento")
        self.tabs.addTab(self.tab_historico, "📊 Histórico")
        self.tabs.addTab(self.tab_config, "⚙️ Configurações")
        self.setCentralWidget(self.tabs)

        # ========== MONITORAMENTO ==========
        self.video_label = QLabel("Feed de vídeo")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: #111; border: 2px solid #444; color: #aaa;")
        self.video_label.setMinimumSize(960, 540)
        self.video_label.setMaximumSize(1280, 720)

        self.status_label = QLabel("Status: Aguardando início...")
        self.status_label.setFont(QFont("Arial", 14))
        self.status_label.setStyleSheet("color: #ddd;")

        self.timer_label = QLabel("⏱ Tempo monitorado: 00:00")
        self.timer_label.setFont(QFont("Arial", 14))
        self.timer_label.setStyleSheet("color: #66ccff;")

        # Labels para valores puros (detector de estresse e detector de microgestos)
        self.micro_label = QLabel("ESTRESSE: --")
        self.micro_label.setFont(QFont("Arial", 12))
        self.micro_label.setStyleSheet("color: #ffd27f;")

        self.posture_label = QLabel("MICROGESTO CLASS: -- (conf: --)")
        self.posture_label.setFont(QFont("Arial", 12))
        self.posture_label.setStyleSheet("color: #ffa500;")

        # Por padrão não mostramos os resultados numéricos na UI
        self.micro_label.setVisible(False)
        self.posture_label.setVisible(False)

        # Botões
        self.btn_iniciar = QPushButton("▶️ Iniciar")
        self.btn_pausar = QPushButton("⏸️ Pausar")
        self.btn_encerrar = QPushButton("⏹️ Encerrar")
        self.btn_test_sound = QPushButton("🔊 Testar som")
        for btn in [self.btn_iniciar, self.btn_pausar, self.btn_encerrar]:
            btn.setFixedHeight(45)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #444; color: white; border-radius: 8px; font-size: 14px;
                }
                QPushButton:hover { background-color: #555; }
                QPushButton:pressed { background-color: #222; }
            """)
        # estilo e tamanho para o botão de teste de som
        self.btn_test_sound.setFixedHeight(45)
        self.btn_test_sound.setStyleSheet("background-color: #2b2b2b; color: #fff; border-radius: 8px; font-size: 14px;")

        layout_botoes = QHBoxLayout()
        layout_botoes.addWidget(self.btn_iniciar)
        layout_botoes.addWidget(self.btn_pausar)
        layout_botoes.addWidget(self.btn_encerrar)
        layout_botoes.addWidget(self.btn_test_sound)

        vbox_monitor = QVBoxLayout()
        vbox_monitor.addWidget(self.video_label)
        vbox_monitor.addWidget(self.status_label)
        vbox_monitor.addWidget(self.timer_label)
        vbox_monitor.addWidget(self.micro_label)
        vbox_monitor.addWidget(self.posture_label)
        vbox_monitor.addLayout(layout_botoes)
        self.tab_monitor.setLayout(vbox_monitor)

        # ========= VARIÁVEIS =========
        self.timer = QTimer()
        self.timer.timeout.connect(self.atualizar_frame)
        self.start_time = None
        self.paused = False
        self.alertas = 0
        self.historico = []
        # FPS tracking: last timestamp, recent history for smoothing, and last display value
        self.last_frame_time = None
        self.fps_history = []  # list of recent fps samples for smoothing
        self.fps_display = 0.0

        # ========= SOM =========
        def beep_callback():
            """Callback simples para som de alerta"""
            try:
                # Respeita a checkbox de som
                if not self.checkbox_som.isChecked():
                    return
                if platform.system() == "Windows":
                    import winsound
                    winsound.Beep(1000, 400)
                else:
                    # Som simples usando simpleaudio se disponível
                    try:
                        import numpy as np
                        import simpleaudio as sa
                        fs = 44100
                        t = np.linspace(0, 0.3, int(fs * 0.3), False)
                        tone = np.sin(1000 * t * 2 * np.pi)
                        audio = (tone * (32767 * 0.8)).astype(np.int16)
                        play_obj = sa.play_buffer(audio, 1, 2, fs)
                        play_obj.wait_done()
                    except ImportError:
                        print("🔊 ALERTA: Estresse detectado!")
            except Exception as e:
                print(f"🔊 ALERTA: {e}")
        
        self.beep = beep_callback

        # conecta o botão de teste de som
        try:
            self.btn_test_sound.clicked.connect(self._on_test_sound)
        except Exception:
            pass

        # ========= MÓDULO INTEGRADO (MICROGESTURE + POSTURA) ==========
        self.monitor_postura = MicrogestureMonitor(
            model_path="models/model_face_temporal.pkl",
            features_json_path="models/features.json",
            # Usar o classificador de microgestos 'best_model.keras' (24 classes)
            posture_model_path="models/best_model.keras",
            beep_callback=self.beep,
            update_callback=self.atualizar_status_ui
        )
        # Não desenhar resultados sobre o frame (somente gravação em histórico)
        self.monitor_postura.show_on_screen = False

        # ========= MONITOR DE POSTURA LEGACY (opcional) =========
        # Instanciação do monitor legacy é feita mais abaixo, após criação dos
        # controles de configuração (sliders, checkboxes). Isso evita tentar
        # ler valores de widgets ainda não inicializados.
        self.posture_monitor = None
        # Usa o MicrogestureMonitor (integrado) por padrão — ele combina
        # detector facial temporal e o classificador Keras (best_model.keras).
        # Mantemos a instância legacy em `self.posture_monitor` apenas como
        # fallback/manual, mas `self.active_monitor` aponta para o integrado.
        self.active_monitor = self.monitor_postura

        # ========= ABA HISTÓRICO =========
        self.hist_label = QLabel("📊 Relatórios de monitoramento")
        self.hist_label.setFont(QFont("Arial", 14))

        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvas(self.fig)

        vbox_hist = QVBoxLayout()
        vbox_hist.addWidget(self.hist_label)
        vbox_hist.addWidget(self.canvas)
        self.tab_historico.setLayout(vbox_hist)

        # ========= ABA CONFIGURAÇÕES =========
        form = QFormLayout()

        # ========== SLIDER TEMPO DE ALERTA ==========
        self.slider_alerta = QSlider(Qt.Orientation.Horizontal)
        self.slider_alerta.setRange(3, 10)
        self.slider_alerta.setValue(5)
        self.label_alerta_valor = QLabel(f"{self.slider_alerta.value()} s")
        self.label_alerta_valor.setFixedWidth(50)
        self.label_alerta_valor.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout_alerta = QHBoxLayout()
        layout_alerta.addWidget(self.slider_alerta)
        layout_alerta.addWidget(self.label_alerta_valor)
        self.slider_alerta.valueChanged.connect(lambda v: self.label_alerta_valor.setText(f"{v} s"))

        # ========== SLIDER LEMBRETE DE ALONGAMENTO ==========
        self.slider_alongamento = QSlider(Qt.Orientation.Horizontal)
        self.slider_alongamento.setRange(15, 90)
        self.slider_alongamento.setValue(45)
        self.slider_alongamento.setTickInterval(15)
        self.slider_alongamento.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.label_alongamento_valor = QLabel(f"{self.slider_alongamento.value()} min")
        self.label_alongamento_valor.setFixedWidth(60)
        self.label_alongamento_valor.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout_alongamento = QHBoxLayout()
        layout_alongamento.addWidget(self.slider_alongamento)
        layout_alongamento.addWidget(self.label_alongamento_valor)
        self.slider_alongamento.valueChanged.connect(lambda v: self.label_alongamento_valor.setText(f"{v} min"))

        # ========== SLIDER DE VOLUME ==========
        self.slider_volume = QSlider(Qt.Orientation.Horizontal)
        self.slider_volume.setRange(0, 100)
        self.slider_volume.setValue(80)
        self.slider_volume.setTickInterval(10)
        self.slider_volume.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.label_volume_valor = QLabel(f"{self.slider_volume.value()}%")
        self.label_volume_valor.setFixedWidth(50)
        self.label_volume_valor.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout_volume = QHBoxLayout()
        layout_volume.addWidget(self.slider_volume)
        layout_volume.addWidget(self.label_volume_valor)
        self.slider_volume.valueChanged.connect(lambda v: self.label_volume_valor.setText(f"{v}%"))

        # ========== CHECKBOX SOM ==========
        self.checkbox_som = QCheckBox("Ativar alerta sonoro")
        self.checkbox_som.setChecked(True)

        # ========== CHECKBOX PARA MOSTRAR PREDIÇÕES NA TELA ==========
        self.checkbox_show_predictions = QCheckBox("Mostrar predições na tela")
        self.checkbox_show_predictions.setChecked(False)

        # ========== ADICIONA AO FORM ==========
        form.addRow("⏱ Tempo de alerta (s):", layout_alerta)
        form.addRow("🕐 Lembrete de alongar (min):", layout_alongamento)
        form.addRow("🔊 Volume dos alertas (%):", layout_volume)
        form.addRow(self.checkbox_som)
        form.addRow(self.checkbox_show_predictions)

        # Conecta toggle para mostrar/ocultar labels numéricas
        self.checkbox_show_predictions.stateChanged.connect(self._on_toggle_show_predictions)

        self.tab_config.setLayout(form)

        # ===== Instancia o PostureMonitor legacy AGORA que os widgets existem =====
        try:
            if PostureMonitor is not None:
                # Usa os valores iniciais dos sliders para configurar o monitor
                self.posture_monitor = PostureMonitor(
                    alerta_segundos=self.slider_alerta.value(),
                    lembrete_alongar_min=self.slider_alongamento.value(),
                    beep_callback=self.beep,
                    update_callback=self.atualizar_status_ui
                )
                try:
                    print(f"[App] PostureMonitor legacy instantiated: {self.posture_monitor is not None}, debounce={getattr(self.posture_monitor, 'DEBOUNCE_FRAMES', 'n/a')}")
                except Exception:
                    pass
        except Exception as e:
            print("Falha ao instanciar PostureMonitor legacy:", e)
            self.posture_monitor = None

        # ========= CONEXÕES =========
        self.btn_iniciar.clicked.connect(self.iniciar_monitoramento)
        self.btn_pausar.clicked.connect(self.pausar_monitoramento)
        self.btn_encerrar.clicked.connect(self.encerrar_monitoramento)

    # ========== FUNÇÕES ==========
    def iniciar_monitoramento(self):
        try:
            # Inicia ambos os monitores automaticamente (integrado + legacy se disponível)
            # Configura parâmetros comuns e inicia o monitor integrado (MicrogestureMonitor)
            self.monitor_postura.ALERTA_SEGUNDOS_POSTURA = self.slider_alerta.value()
            self.monitor_postura.VOLUME = self.slider_volume.value() / 100
            self.monitor_postura.LEMBRETE_ALONGAR_MIN = self.slider_alongamento.value()
            self.monitor_postura.iniciar()

            # Tenta iniciar o PostureMonitor legacy (se instanciado). Falhas aqui não impedem o integrado.
            if self.posture_monitor is not None:
                try:
                    self.posture_monitor.ALERTA_SEGUNDOS_POSTURA = self.slider_alerta.value()
                    self.posture_monitor.VOLUME = self.slider_volume.value() / 100
                    self.posture_monitor.LEMBRETE_ALONGAR_MIN = self.slider_alongamento.value()
                    self.posture_monitor.iniciar()
                except Exception as e_post:
                    print("Falha ao iniciar PostureMonitor legacy:", e_post)

            # Mantemos compatibilidade: aponta active_monitor para o integrado
            self.active_monitor = self.monitor_postura
        except Exception as e:
            QMessageBox.warning(self, "Erro", str(e))
            return

        self.status_label.setText("🟢 Monitorando postura...")
        self.start_time = time.time()
        self.paused = False
        self.alertas = 0
        self.historico = []
        self.timer.start(30)

    def _on_test_sound(self):
        """Handler do botão 'Testar som' — reproduz o som de alerta (respeita checkbox de som)."""
        try:
            # Reutiliza o callback de beep central (aplica mesma lógica da UI)
            if callable(self.beep):
                print("[App] Teste de som: acionando beep callback")
                try:
                    self.beep()
                except Exception as e:
                    print(f"[App] Erro ao chamar beep(): {e}")
            else:
                print("[App] Beep callback não disponível.")
        except Exception as e:
            print(f"[App] Erro no _on_test_sound: {e}")

    def pausar_monitoramento(self):
        if self.paused:
            self.paused = False
            self.status_label.setText("🟢 Retomando monitoramento...")
            self.timer.start(30)
        else:
            self.paused = True
            self.status_label.setText("⏸️ Monitoramento pausado.")
            self.timer.stop()

    def encerrar_monitoramento(self):
        """Finaliza o monitoramento e salva sessão"""
        self.timer.stop()
        # Para ambos os monitores se estiverem rodando
        for m in [getattr(self, 'monitor_postura', None), getattr(self, 'posture_monitor', None)]:
            if m is not None:
                try:
                    m.parar()
                except Exception:
                    pass
        self.video_label.clear()
        self.status_label.setText("🛑 Monitoramento encerrado.")

        inicio = time.strftime("%d/%m/%Y %H:%M:%S", time.localtime(self.start_time))
        fim = time.strftime("%d/%m/%Y %H:%M:%S")

        # Salva sessão atual
        self.salvar_sessao(inicio, fim, self.historico)

        # Atualiza gráfico geral
        self.gerar_graficos_multiplos()

        tempo_total = time.time() - self.start_time
        minutos = int(tempo_total // 60)
        segundos = int(tempo_total % 60)
        QMessageBox.information(
            self,
            "Resumo do Monitoramento",
            f"⏱ Tempo total: {minutos:02d}:{segundos:02d}\n"
            f"🔔 Alertas emitidos: {self.alertas}\n"
            f"📅 Início: {inicio}\n"
            f"📅 Fim: {fim}"
        )

    def salvar_sessao(self, inicio, fim, dados):
        """Salva a sessão atual no JSON (com 'dados') e no CSV (resumo). Mantém só as 8 últimas."""
        import json, os
    # Normaliza dados: aceita formato antigo [tempo, 'boa'/'ruim'] ou novo [tempo, {"micro":.., ...}]
        dados_norm = []
        for item in dados:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                t = float(item[0])
                s = item[1]
                # Se for dicionário (valores puros), mantém como está (json serializável)
                if isinstance(s, dict):
                    # Converte valores numéricos quando possível; mantém inteiros (classes) sem alteração
                    clean = {}
                    for k, v in s.items():
                        if v is None:
                            clean[k] = None
                        else:
                            try:
                                # tenta converter para int quando chave for posture_class
                                if k == 'posture_class':
                                    clean[k] = int(v)
                                else:
                                    clean[k] = float(v)
                            except Exception:
                                clean[k] = v
                    dados_norm.append([t, clean])
                else:
                    # valor legacy, grava como string
                    dados_norm.append([t, str(s)])

        # Adiciona ao histórico em memória
        nova = {"inicio": inicio, "fim": fim, "dados": dados_norm}
        self.historico_sessoes.append(nova)
        self.historico_sessoes = self.historico_sessoes[-8:]

        # --- Persistência JSON (completo, inclui 'dados') ---
        try:
            with open(self.historico_json_path, "w", encoding="utf-8") as f:
                json.dump(self.historico_sessoes, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print("Erro ao salvar JSON de sessões:", e)

        # --- CSV (resumo) ---
        try:
            df_csv = pd.DataFrame([{
                "inicio": s["inicio"],
                "fim": s["fim"],
                # Estimativa de duração pelos frames coletados nesta sessão
                "duracao_min": round((len(s["dados"]) / 30) / 60, 2),  # supondo ~30fps de amostragem lógica
                "frames": len(s["dados"])
            } for s in self.historico_sessoes])
            df_csv.to_csv(self.historico_csv_path, index=False)
        except Exception as e:
            print("Erro ao salvar CSV de sessões:", e)


    def carregar_historico_sessoes(self):
        """Carrega sessões anteriores do JSON completo. Se não existir, inicia vazio."""
        import os, json
        if os.path.exists(self.historico_json_path):
            try:
                with open(self.historico_json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Garante formato
                self.historico_sessoes = []
                for s in data:
                    self.historico_sessoes.append({
                        "inicio": s.get("inicio"),
                        "fim": s.get("fim"),
                        "dados": s.get("dados", [])  # <- garante que 'dados' exista
                    })
                # Mantém no máximo 8
                self.historico_sessoes = self.historico_sessoes[-8:]
            except Exception as e:
                print("Falha ao carregar JSON de sessões:", e)
                self.historico_sessoes = []
        else:
            self.historico_sessoes = []




    def atualizar_status_ui(self, status, dica):
        # status can be: 'raw_microgesture' or alert types 'alert_corner'/'alert_center'
        if status == 'alert_corner':
            # Exibe notificação discreta no canto inferior direito
            self._show_corner_popup(dica)
            # também atualiza a status bar
            self.status_label.setText(f"⚠️ {dica}")
        elif status == 'alert_center':
            # Exibe alerta central chamativo
            self._show_center_popup(dica)
            self.status_label.setText(f"⚠️ {dica}")
        else:
            self.status_label.setText(f"{'⚠️' if status == 'ruim' else '✅'} {dica}")

    def _on_toggle_show_predictions(self, state):
        """Mostra ou oculta as labels com os valores das predições."""
        try:
            show = state == Qt.Checked
            self.micro_label.setVisible(show)
            self.posture_label.setVisible(show)
        except Exception:
            pass

    def gerar_graficos_multiplos(self):
        """Gera até 8 gráficos, um por sessão (linha temporal de postura)"""
        if not self.historico_sessoes:
            self.ax.clear()
            self.ax.text(0.5, 0.5, "Nenhum histórico encontrado",
                         ha='center', va='center', fontsize=12, color='gray')
            self.canvas.draw()
            return

        self.ax.clear()

        # Gera um gráfico por sessão (cores gradientes)
        cores = plt.cm.viridis(np.linspace(0, 1, len(self.historico_sessoes)))
        for i, sessao in enumerate(self.historico_sessoes):
            df = pd.DataFrame(sessao["dados"], columns=["tempo", "status"])
            if df.empty:
                continue

            # Extrai valor numérico: prioriza o resultado do detector facial ('micro')
            def _to_val(s):
                if isinstance(s, dict):
                    v = s.get("micro") if s.get("micro") is not None else None
                    return float(v) if v is not None else np.nan
                if isinstance(s, str):
                    return 1.0 if s == "boa" else 0.0 if s == "ruim" else np.nan
                return np.nan

            # Agrupa por minuto (tempo está em segundos desde o início)
            df["minuto"] = (df["tempo"] // 60).astype(int)
            df["valor"] = df["status"].apply(_to_val)

            # Extrai rótulo predito pelo modelo Keras (se disponível)
            def _to_label(s):
                if isinstance(s, dict):
                    return s.get("posture_label")
                return None

            df["label"] = df["status"].apply(_to_label)

            # Média da probabilidade por minuto
            df_min = df.groupby("minuto")["valor"].mean().reset_index()
            # Rótulo mais frequente por minuto (modo) para anotar no gráfico
            df_label = (
                df.groupby("minuto")["label"]
                .agg(lambda x: x.dropna().mode().iloc[0] if not x.dropna().empty else None)
                .reset_index()
            )

            df_min = df_min.merge(df_label, on="minuto", how="left")

            self.ax.plot(
                df_min["minuto"],
                df_min["valor"],
                marker="o",
                color=cores[i],
                label=f"{sessao['inicio'].split()[0]} {sessao['fim'].split()[1]}"
            )

            # Anota rótulos de classe (quando disponíveis) acima de cada ponto
            for _, r in df_min.iterrows():
                lbl = r.get("label")
                if lbl is not None and lbl is not pd.NA:
                    try:
                        self.ax.text(r["minuto"], r["valor"] + 0.02, str(lbl),
                                     fontsize=8, ha="center", color=cores[i])
                    except Exception:
                        pass

            self.ax.set_ylim(-0.05, 1.05)
            self.ax.set_yticks([0, 0.5, 1])
            self.ax.set_yticklabels(["0.0", "0.5", "1.0"])
            self.ax.set_xlabel("Tempo (minutos)")
            self.ax.set_ylabel("Probabilidade (detector facial)")
            self.ax.set_title("Evolução da prob. de estresse (microgesture) nas sessões")
            self.ax.legend(loc="upper right", fontsize=8)
            self.ax.grid(True, linestyle="--", alpha=0.4)

            self.canvas.draw()


    # === POPUPS ===
    def _show_corner_popup(self, text, timeout=5000):
        """Mostra uma notificação discreta no canto inferior direito."""
        try:
            w = QMessageBox(self)
            w.setWindowTitle("Sugestão")
            w.setText(text)
            w.setStandardButtons(QMessageBox.NoButton)
            w.setStyleSheet("QMessageBox { background-color: #222; color: #ffd27f; }")
            # posiciona no canto inferior direito da janela
            geo = self.geometry()
            w.setGeometry(geo.x() + geo.width() - 360, geo.y() + geo.height() - 140, 340, 100)
            w.show()
            QTimer.singleShot(timeout, w.close)
        except Exception as e:
            print("Erro ao mostrar popup de canto:", e)

    def _show_center_popup(self, text, timeout=7000):
        """Mostra uma notificação central mais chamativa."""
        try:
            w = QMessageBox(self)
            w.setWindowTitle("Alerta de Estresse")
            w.setText(text)
            w.setIcon(QMessageBox.Icon.Warning)
            w.setStandardButtons(QMessageBox.StandardButton.Ok)
            w.setStyleSheet("QMessageBox { background-color: #2b2b2b; color: #ffb3b3; }")
            w.setModal(False)
            w.show()
            # Fecha automaticamente após timeout se o usuário não interagir
            QTimer.singleShot(timeout, w.close)
        except Exception as e:
            print("Erro ao mostrar popup central:", e)



    def atualizar_frame(self):
        # Se pausado, não processa
        if self.paused:
            return

        # Processa ambos os monitores (integrado + legacy) quando disponíveis.
        frame_micro = None
        frame_posture_legacy = None
        try:
            if getattr(self, 'monitor_postura', None) and getattr(self.monitor_postura, 'running', False):
                frame_micro, status_micro, dica_micro, cor_micro = self.monitor_postura.processar_frame()
        except Exception as e:
            print("Erro ao processar frame MicrogestureMonitor:", e)

        try:
            if getattr(self, 'posture_monitor', None) and getattr(self.posture_monitor, 'running', False):
                frame_posture_legacy, status_legacy, dica_legacy, cor_legacy = self.posture_monitor.processar_frame()
        except Exception as e:
            print("Erro ao processar frame PostureMonitor legacy:", e)

        # Escolhe qual frame mostrar (preferência: microgesture integrado -> legacy)
        frame = frame_micro if frame_micro is not None else frame_posture_legacy
        if frame is None:
            return

        # ----- FPS calculation and overlay -----
        try:
            now = time.time()
            if self.last_frame_time is None:
                dt = None
            else:
                dt = now - self.last_frame_time
            self.last_frame_time = now

            if dt is None or dt <= 0:
                fps_current = 0.0
            else:
                fps_current = 1.0 / dt

            # append and keep last N samples for smoothing
            self.fps_history.append(fps_current)
            if len(self.fps_history) > 10:
                self.fps_history.pop(0)

            # smoothed fps
            if self.fps_history:
                fps = sum(self.fps_history) / len(self.fps_history)
            else:
                fps = 0.0
            self.fps_display = fps

            # draw overlay at bottom-left of frame
            try:
                h, w = frame.shape[:2]
                text = f"FPS: {self.fps_display:.2f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 0.7
                thickness = 2
                (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
                pad_x = 8
                pad_y = 6

                x1 = 10
                y1 = h - 10 - text_h - pad_y
                x2 = x1 + text_w + pad_x * 2
                y2 = h - 10

                # background rectangle (slightly transparent look via dark fill)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (30, 30, 30), -1)
                # text (light color)
                cv2.putText(frame, text, (x1 + pad_x, y2 - pad_y), font, scale, (230, 230, 230), thickness, cv2.LINE_AA)
            except Exception:
                # If any drawing error occurs, ignore and continue
                pass
        except Exception:
            # don't let FPS calculation break frame processing
            pass

        # Coleta valores do monitor integrado (microgesture + Keras)
        micro_val = getattr(self.monitor_postura, 'last_proba', None)
        posture_class = getattr(self.monitor_postura, 'last_posture_class', None)
        posture_conf = getattr(self.monitor_postura, 'last_posture_confidence', None)
        posture_label = getattr(self.monitor_postura, 'last_posture_label', None)
        posture_pred = getattr(self.monitor_postura, 'last_posture_prediction', None)

        # Coleta valores do monitor legacy (se houver)
        legacy_status = None
        legacy_dica = None
        if getattr(self, 'posture_monitor', None):
            legacy_status = locals().get('status_legacy', None)
            legacy_dica = locals().get('dica_legacy', None)

        # Armazena histórico agregando ambas as saídas
        entry = {
            "micro": float(micro_val) if micro_val is not None else None,
            "posture_class": int(posture_class) if posture_class is not None else None,
            "posture_confidence": float(posture_conf) if posture_conf is not None else None,
            "posture_label": posture_label,
            "posture_prediction": posture_pred,
            "legacy_posture_status": legacy_status,
            "legacy_posture_dica": legacy_dica
        }
        self.historico.append([time.time() - self.start_time, entry])

        # AtualIZA LABELS DE PREDIÇÃO (quando habilitado pelo usuário)
        try:
            if getattr(self, 'checkbox_show_predictions', None) and self.checkbox_show_predictions.isChecked():
                try:
                    micro_s = f"ESTRESSE: {float(micro_val):.2f}"
                except Exception:
                    micro_s = "ESTRESSE: --"

                try:
                    if posture_label is not None:
                        label_str = str(posture_label)
                    elif posture_class is not None:
                        label_str = str(int(posture_class))
                    else:
                        label_str = "--"
                except Exception:
                    label_str = "--"

                try:
                    conf_s = f"{float(posture_conf):.2f}" if posture_conf is not None else "--"
                except Exception:
                    conf_s = "--"

                self.micro_label.setText(micro_s)
                self.posture_label.setText(f"MICROGESTO CLASS: {label_str} (conf: {conf_s})")
                self.micro_label.setVisible(True)
                self.posture_label.setVisible(True)
            else:
                # mantém oculto quando desativado
                try:
                    self.micro_label.setVisible(False)
                    self.posture_label.setVisible(False)
                except Exception:
                    pass
        except Exception as e:
            print("Erro ao atualizar labels visuais:", e)

        # Atualiza timer UI
        tempo_decorrido = time.time() - self.start_time
        minutos = int(tempo_decorrido // 60)
        segundos = int(tempo_decorrido % 60)
        self.timer_label.setText(f"⏱ Tempo monitorado: {minutos:02d}:{segundos:02d}")

        # Mostra frame (convertendo para QImage)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qt_img = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img).scaled(960, 540, Qt.AspectRatioMode.KeepAspectRatio)
        self.video_label.setPixmap(pixmap)


# ========== MAIN ==========
if __name__ == "__main__":
    app = QApplication(sys.argv)
    janela = PosturaApp()
    janela.show()
    sys.exit(app.exec())
