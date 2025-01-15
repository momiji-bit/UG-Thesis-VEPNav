import sys
import socket
import threading
import pickle
import time

import numpy as np
from ultralytics import YOLO
from utils.models import RoadBoundGetter
import cv2
import asyncio
from PyQt5.QtWidgets import *
from PyQt5.QtCore import pyqtSignal, QThread, Qt
from PyQt5.QtGui import QPixmap, QImage
import logging

# 设置服务器参数
HOST = '127.0.0.1'
PORT = 65432

# 创建一个套接字
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen()

class ServerThread(QThread):
    update_signal = pyqtSignal(str, np.ndarray, float, float, float)
    log_signal = pyqtSignal(str)
    connection_closed_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.road_model = RoadBoundGetter(scale=0.3, density=10, pretrain="utils/road_model_maxmIOU75.pth")
        self.YOLO_model = YOLO('utils/yolov8s-seg.pt')
        self.paused = False
        self.pause_cond = threading.Condition()

    async def handle_client(self, client_socket, addr):
        self.log_signal.emit(f"Connected by {addr}")
        while True:
            try:
                t0 = time.time()

                with self.pause_cond:
                    while self.paused:
                        self.pause_cond.wait()
                data = b''
                while True:
                    packet = await self.loop.sock_recv(client_socket, 4096)
                    if packet.endswith(b'$END#'):
                        data += packet[:-5]
                        break
                    data += packet

                t1 = time.time()

                if data:
                    received_data = pickle.loads(data)
                    rgb_image_compressed = received_data['rgb']
                    rgb_image = cv2.imdecode(np.frombuffer(rgb_image_compressed, np.uint8), cv2.IMREAD_COLOR)

                    result = self.YOLO_model.track(rgb_image, persist=True, verbose=False)[0]
                    road_masked = self.road_model(rgb_image)[0].cpu()
                    data = {'yolo': result, 'road': road_masked}

                    t2 = time.time()

                    await self.loop.sock_sendall(client_socket, pickle.dumps(data) + b'#END$')

                    t3 = time.time()

                    self.update_signal.emit(f"Received data from {addr}", rgb_image, t1-t0, t2-t1, t3-t2)
            except Exception as e:
                self.log_signal.emit(f"Error with {addr}: {e}")
                break
        self.log_signal.emit(f"Connection closed by {addr}")
        self.connection_closed_signal.emit()
        client_socket.close()

    async def accept_clients(self):
        while True:
            client_socket, addr = await self.loop.sock_accept(server_socket)
            asyncio.create_task(self.handle_client(client_socket, addr))

    def run(self):
        self.loop.run_until_complete(self.accept_clients())

    def pause(self):
        self.paused = True
        self.log_signal.emit("Server paused")

    def resume(self):
        self.paused = False
        with self.pause_cond:
            self.pause_cond.notify_all()
        self.log_signal.emit("Server resumed")

class LogHandler(logging.Handler):
    def __init__(self, log_widget):
        super().__init__()
        self.log_widget = log_widget

    def emit(self, record):
        msg = self.format(record)
        self.log_widget.append(msg)

class ServerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.server_thread = ServerThread()
        self.server_thread.update_signal.connect(self.update_display)
        self.server_thread.log_signal.connect(self.log_message)
        self.server_thread.connection_closed_signal.connect(self.reset_image_label)
        self.server_thread.start()

        self.setup_logger()

    def initUI(self):
        self.setWindowTitle('云端服务器')
        self.setFixedSize(660, 660)

        self.image_label = QLabel(self)
        self.image_label.setGeometry(10, 10, 640, 360)
        self.image_label.setText("RGB Data")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #FAF5E4;")

        self.log_text = QTextEdit(self)
        self.log_text.setGeometry(10, 380, 640, 220)
        self.log_text.setReadOnly(True)

        self.btn_run = QPushButton("服务器运行", self)
        self.btn_run.setGeometry(10, 610, 200, 40)
        self.btn_pause = QPushButton("服务器暂停", self)
        self.btn_pause.setGeometry(230, 610, 200, 40)
        self.btn_resume = QPushButton("服务器恢复", self)
        self.btn_resume.setGeometry(450, 610, 200, 40)

        self.btn_run.clicked.connect(self.clicked_btn_run)
        self.btn_pause.clicked.connect(self.clicked_btn_pause)
        self.btn_resume.clicked.connect(self.clicked_btn_resume)

    def setup_logger(self):
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        log_handler = LogHandler(self.log_text)
        log_handler.setLevel(logging.DEBUG)
        log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        log_handler.setFormatter(log_formatter)
        logger.addHandler(log_handler)

    def clicked_btn_run(self):
        logging.info("Server is listening...")
        self.server_thread.start()

    def clicked_btn_pause(self):
        self.server_thread.pause()

    def clicked_btn_resume(self):
        self.server_thread.resume()

    def update_display(self, log_message, image, time1, time2, time3):
        self.log_text.append(log_message)
        cv2.putText(image, f"{int(time1 * 1000)}ms+{int(time2 * 1000)}ms+{int(time3 * 1000)}ms", (0, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.8,
                    (0, 0, 255))
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qt_image))

    def log_message(self, message):
        self.log_text.append(message)

    def reset_image_label(self):
        self.image_label.setStyleSheet("background-color: #FAF5E4;")
        self.image_label.setText("RGB Data")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = ServerApp()
    ex.show()
    sys.exit(app.exec_())
