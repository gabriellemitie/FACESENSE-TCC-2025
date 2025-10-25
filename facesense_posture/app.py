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

# ===== IMPORTA O MÓDULO DE POSTURA =====
from modules.posture_model import PostureMonitor


class PosturaApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🧍 FACESENSE - Monitor de Postura")
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

        # Botões
        self.btn_iniciar = QPushButton("▶️ Iniciar")
        self.btn_pausar = QPushButton("⏸️ Pausar")
        self.btn_encerrar = QPushButton("⏹️ Encerrar")
        for btn in [self.btn_iniciar, self.btn_pausar, self.btn_encerrar]:
            btn.setFixedHeight(45)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #444; color: white; border-radius: 8px; font-size: 14px;
                }
                QPushButton:hover { background-color: #555; }
                QPushButton:pressed { background-color: #222; }
            """)

        layout_botoes = QHBoxLayout()
        layout_botoes.addWidget(self.btn_iniciar)
        layout_botoes.addWidget(self.btn_pausar)
        layout_botoes.addWidget(self.btn_encerrar)

        vbox_monitor = QVBoxLayout()
        vbox_monitor.addWidget(self.video_label)
        vbox_monitor.addWidget(self.status_label)
        vbox_monitor.addWidget(self.timer_label)
        vbox_monitor.addLayout(layout_botoes)
        self.tab_monitor.setLayout(vbox_monitor)

        # ========= VARIÁVEIS =========
        self.timer = QTimer()
        self.timer.timeout.connect(self.atualizar_frame)
        self.start_time = None
        self.paused = False
        self.alertas = 0
        self.historico = []

        # ========= SOM =========
        if platform.system() == "Windows":
            import winsound
            self.beep = lambda: winsound.Beep(1000, 400)
        else:
            from playsound import playsound
            self.beep = lambda: playsound("assets/alerta_postura.mp3")

        # ========= MÓDULO DE POSTURA =========
        self.monitor_postura = PostureMonitor(
            beep_callback=self.beep,
            update_callback=self.atualizar_status_ui
        )

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

        # ========== ADICIONA AO FORM ==========
        form.addRow("⏱ Tempo de alerta (s):", layout_alerta)
        form.addRow("🕐 Lembrete de alongar (min):", layout_alongamento)
        form.addRow("🔊 Volume dos alertas (%):", layout_volume)
        form.addRow(self.checkbox_som)

        self.tab_config.setLayout(form)


        # ========= CONEXÕES =========
        self.btn_iniciar.clicked.connect(self.iniciar_monitoramento)
        self.btn_pausar.clicked.connect(self.pausar_monitoramento)
        self.btn_encerrar.clicked.connect(self.encerrar_monitoramento)

    # ========== FUNÇÕES ==========
    def iniciar_monitoramento(self):
        try:
            self.monitor_postura.ALERTA_SEGUNDOS_POSTURA = self.slider_alerta.value()
            self.monitor_postura.VOLUME = self.slider_volume.value() / 100 
            self.monitor_postura.LEMBRETE_ALONGAR_MIN = self.slider_alongamento.value()
            self.monitor_postura.iniciar()
        except Exception as e:
            QMessageBox.warning(self, "Erro", str(e))
            return

        self.status_label.setText("🟢 Monitorando postura...")
        self.start_time = time.time()
        self.paused = False
        self.alertas = 0
        self.historico = []
        self.timer.start(30)

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
        self.monitor_postura.parar()
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

        # Normaliza dados: lista de [tempo, status]
        dados_norm = []
        for item in dados:
            # item esperado: [tempo_segundos, "boa"/"ruim"]
            if isinstance(item, (list, tuple)) and len(item) == 2:
                dados_norm.append([float(item[0]), str(item[1])])

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
        self.status_label.setText(f"{'⚠️' if status == 'ruim' else '✅'} {dica}")

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

            df["minuto"] = (df["tempo"] // 30).astype(int)
            df["valor"] = df["status"].map({"boa": 1, "ruim": 0})
            df_min = df.groupby("minuto")["valor"].mean().reset_index()

            self.ax.plot(
                df_min["minuto"],
                df_min["valor"],
                marker="o",
                color=cores[i],
                label=f"{sessao['inicio'].split()[0]} {sessao['fim'].split()[1]}"
            )

        self.ax.set_ylim(-0.1, 1.1)
        self.ax.set_yticks([0, 0.5, 1])
        self.ax.set_yticklabels(["Ruim", "Média", "Boa"])
        self.ax.set_xlabel("Tempo (minutos)")
        self.ax.set_ylabel("Qualidade da postura")
        self.ax.set_title("Evolução das últimas sessões")
        self.ax.legend(loc="upper right", fontsize=8)
        self.ax.grid(True, linestyle="--", alpha=0.4)

        self.canvas.draw()



    def atualizar_frame(self):
        if self.paused or not self.monitor_postura.running:
            return

        frame, status, dica, cor = self.monitor_postura.processar_frame()
        if frame is None:
            return

        self.historico.append([time.time() - self.start_time, status])
        self.status_label.setText(f"Postura {status.upper()}")
        self.status_label.setStyleSheet(f"color: rgb({cor[0]}, {cor[1]}, {cor[2]});")

        tempo_decorrido = time.time() - self.start_time
        minutos = int(tempo_decorrido // 60)
        segundos = int(tempo_decorrido % 60)
        self.timer_label.setText(f"⏱ Tempo monitorado: {minutos:02d}:{segundos:02d}")

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
