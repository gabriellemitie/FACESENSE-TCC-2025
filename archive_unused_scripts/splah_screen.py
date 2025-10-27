import sys
import time
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QPushButton, QWidget, QProgressBar
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QPixmap
from app import PosturaApp  # importa o app principal


class SplashScreen(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FACESENSE - Iniciando...")
        self.setFixedSize(600, 400)
        self.setStyleSheet("background-color: #101010; color: #fff;")

        # ======== LAYOUT ========
        self.container = QWidget()
        self.layout = QVBoxLayout()
        self.container.setLayout(self.layout)
        self.setCentralWidget(self.container)

        # ======== LOGO ========
        self.logo_label = QLabel()
        self.logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        try:
            pixmap = QPixmap("logo_facesense.png").scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatio)
            self.logo_label.setPixmap(pixmap)
        except:
            self.logo_label.setText("🧍")
            self.logo_label.setFont(QFont("Arial", 100))

        # ======== TÍTULO ========
        self.title_label = QLabel("FACESENSE Stress Monitor")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setFont(QFont("Arial", 18, QFont.Weight.Bold))

        # ======== TEXTO DE STATUS ========
        self.status_label = QLabel("Carregando módulos...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setFont(QFont("Arial", 12))
        self.status_label.setStyleSheet("color: #aaa;")

        # ======== BARRA DE PROGRESSO ========
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setStyleSheet("""
            QProgressBar {
                border: 2px solid #444; 
                border-radius: 8px;
                text-align: center;
                height: 20px;
                color: #fff;
            }
            QProgressBar::chunk {
                background-color: #0078d7;
                border-radius: 8px;
            }
        """)

        # ======== BOTÃO ========
        self.btn_start = QPushButton("Entrar no aplicativo")
        self.btn_start.setFixedHeight(40)
        self.btn_start.setStyleSheet("""
            QPushButton {
                background-color: #0078d7; color: white;
                border-radius: 10px; font-size: 14px;
            }
            QPushButton:hover { background-color: #339cff; }
            QPushButton:pressed { background-color: #005fa3; }
        """)
        self.btn_start.clicked.connect(self.abrir_principal)

        # ======== ORGANIZAÇÃO ========
        self.layout.addStretch()
        self.layout.addWidget(self.logo_label)
        self.layout.addWidget(self.title_label)
        self.layout.addWidget(self.status_label)
        self.layout.addWidget(self.progress)
        self.layout.addWidget(self.btn_start)
        self.layout.addStretch()

        # ======== ANIMAÇÃO ========
        self.timer = QTimer()
        self.timer.timeout.connect(self.carregar)
        self.timer.start(50)  # atualização a cada 50ms
        self.valor = 0

    def carregar(self):
        """Simula carregamento visual"""
        self.valor += 2
        self.progress.setValue(self.valor)
        if self.valor < 30:
            self.status_label.setText("Iniciando módulos de vídeo...")
        elif self.valor < 60:
            self.status_label.setText("Configurando MediaPipe...")
        elif self.valor < 90:
            self.status_label.setText("Carregando interface...")
        else:
            self.status_label.setText("Pronto para iniciar!")
        if self.valor >= 100:
            self.timer.stop()

    def abrir_principal(self):
        """Abre o painel principal"""
        self.timer.stop()
        self.close()
        self.main_window = PosturaApp()
        self.main_window.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    splash = SplashScreen()
    splash.show()
    sys.exit(app.exec())
