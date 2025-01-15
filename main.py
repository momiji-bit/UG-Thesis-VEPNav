import sys
import time

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtWebEngineWidgets import *
from utils.OAK_Info import oak_device
from utils.models import RoadBoundGetter
import sounddevice as sd
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from utils.tools import *
import numpy as np
from ultralytics import YOLO
import speech_recognition as sr
import librosa
import requests
import threading
import cv2
import pyaudio
import logging
import random

# 设置日志的配置信息
logging.basicConfig(level=logging.INFO)

objs = [labels_dict[i] for i in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,56,57,58,59,60,61,62,63,67,68,72,73]]


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
        self.YOLO_model = YOLO('utils/yolov8l-seg.pt')
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


class SecondWindow(QWidget):
    windowClosed = pyqtSignal()
    signal_label_collision_probability = pyqtSignal(str)
    signal_update_label_object_detection = pyqtSignal(QPixmap)
    signal_update_label_semantic_segmentation = pyqtSignal(QPixmap)
    signal_update_label_depth = pyqtSignal(QPixmap)
    signal_update_label_BEV = pyqtSignal(QPixmap)
    signal_update_label_collision_probability_data = pyqtSignal(QPixmap)
    signal_update_textEdit_terminal = pyqtSignal(str)

    signal_clean_label_collision_probability = pyqtSignal()
    signal_clean_label_object_detection = pyqtSignal()
    signal_clean_label_semantic_segmentation = pyqtSignal()
    signal_clean_label_depth = pyqtSignal()
    signal_clean_label_BEV = pyqtSignal()
    signal_clean_label_collision_probability_data = pyqtSignal()
    signal_clean_textEdit_terminal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.azimuth_pitch_s = None
        self.threads_running = False
        self.threads_3D_playing = False
        self.clicked_btn_run_is_clicked = False
        self.clicked_btn_play_3D_audio_is_clicked = False
        self.playing = False
        self.initUI()
        self.initFunc()
        self.initdata()
        self.initDevice()

    def initUI(self):
        self.setWindowTitle("轨迹追踪及碰撞预警模块")
        self.setFixedSize(1600, 900)

        self.label_collision_probability = QLabel(self)
        self.label_collision_probability.setGeometry(10, 10, 1580, 40)  # x, y, width, height
        self.label_collision_probability.setText("Collision Probability")
        self.label_collision_probability.setStyleSheet("background-color: #FAF5E4;")

        self.label_object_detection = QLabel(self)
        self.label_object_detection.setGeometry(10, 60, 480, 270)  # x, y, width, height
        self.label_object_detection.setText("Object Detection")
        self.label_object_detection.setStyleSheet("background-color: #FAF5E4;")

        self.label_semantic_segmentation = QLabel(self)
        self.label_semantic_segmentation.setGeometry(10, 340, 480, 270)  # x, y, width, height
        self.label_semantic_segmentation.setText("Semantic Segmentation")
        self.label_semantic_segmentation.setStyleSheet("background-color: #FAF5E4;")

        self.label_depth = QLabel(self)
        self.label_depth.setGeometry(10, 620, 480, 270)  # x, y, width, height
        self.label_depth.setText("Depth")
        self.label_depth.setStyleSheet("background-color: #FAF5E4;")

        self.label_BEV = QLabel(self)
        self.label_BEV.setGeometry(500, 60, 830, 830)  # x, y, width, height
        self.label_BEV.setText("BEV (Bird's-eye view)")
        self.label_BEV.setStyleSheet("background-color: #FAF5E4;")

        self.label_collision_probability_data = QLabel(self)
        self.label_collision_probability_data.setGeometry(1340, 60, 250, 550)  # x, y, width, height
        self.label_collision_probability_data.setText("Collision Probability")
        self.label_collision_probability_data.setStyleSheet("background-color: #FAF5E4;")

        self.textEdit_terminal = QTextEdit(self)
        self.textEdit_terminal.setGeometry(1340, 620, 250, 230)
        self.textEdit_terminal.setReadOnly(True)  # 设置文本框为只读

        self.btn_run = QPushButton("开始", self)
        self.btn_run.setGeometry(1340, 860, 80, 30)
        self.btn_play_3D_audio = QPushButton("3D 音频", self)
        self.btn_play_3D_audio.setGeometry(1425, 860, 80, 30)
        self.btn_stop = QPushButton("停止", self)
        self.btn_stop.setGeometry(1510, 860, 80, 30)

    def initFunc(self):
        self.signal_label_collision_probability.connect(self.label_collision_probability.setStyleSheet)
        self.signal_clean_label_collision_probability.connect(
            lambda: self.label_collision_probability.setText("Collision Probability"))
        self.signal_clean_label_collision_probability.connect(
            lambda: self.label_collision_probability.setStyleSheet("background-color: #FAF5E4;"))

        self.signal_update_label_object_detection.connect(self.label_object_detection.setPixmap)
        self.signal_clean_label_object_detection.connect(
            lambda: self.label_object_detection.setText("Object Detection"))
        self.signal_clean_label_object_detection.connect(
            lambda: self.label_object_detection.setStyleSheet("background-color: #FAF5E4;"))

        self.signal_update_label_semantic_segmentation.connect(self.label_semantic_segmentation.setPixmap)
        self.signal_clean_label_semantic_segmentation.connect(
            lambda: self.label_semantic_segmentation.setText("Semantic Segmentation"))
        self.signal_clean_label_semantic_segmentation.connect(
            lambda: self.label_semantic_segmentation.setStyleSheet("background-color: #FAF5E4;"))

        self.signal_update_label_depth.connect(self.label_depth.setPixmap)
        self.signal_clean_label_depth.connect(
            lambda: self.label_depth.setText("Depth"))
        self.signal_clean_label_depth.connect(
            lambda: self.label_depth.setStyleSheet("background-color: #FAF5E4;"))

        self.signal_update_label_BEV.connect(self.label_BEV.setPixmap)
        self.signal_clean_label_BEV.connect(
            lambda: self.label_BEV.setText("BEV (Bird's-eye view)"))
        self.signal_clean_label_BEV.connect(
            lambda: self.label_BEV.setStyleSheet("background-color: #FAF5E4;"))

        self.signal_update_label_collision_probability_data.connect(self.label_collision_probability_data.setPixmap)
        self.signal_clean_label_collision_probability_data.connect(
            lambda: self.label_collision_probability_data.setText("Collision Probability"))
        self.signal_clean_label_collision_probability_data.connect(
            lambda: self.label_collision_probability_data.setStyleSheet("background-color: #FAF5E4;"))

        self.signal_update_textEdit_terminal.connect(
            lambda text: self.textEdit_terminal.append(f"{text}"))
        self.signal_clean_textEdit_terminal.connect(self.textEdit_terminal.clear)

        self.btn_run.clicked.connect(self.clicked_btn_run)
        self.btn_play_3D_audio.clicked.connect(self.clicked_btn_play_3D_audio)
        self.btn_stop.clicked.connect(self.clicked_btn_stop)

    def initdata(self):
        self.msgGrp = None
        self.img_BGR = None
        self.img_D = None
        self.imu = None

        self.azimuth_pitch_s = None

    def initDevice(self):
        self.queue = oak_device.getOutputQueue("xoutGrp", 10, False)
        self.threads_running = True
        threading.Thread(target=self.thread_refresh_msgGrp).start()

    def thread_refresh_msgGrp(self):
        while self.threads_running:
            self.msgGrp = self.queue.get()
            self.img_BGR = self.msgGrp['color'].getCvFrame()
            self.img_D = self.msgGrp['depth'].getCvFrame()
            self.imu = self.msgGrp['imu'].packets[0]

    def thread_play_audio(self, stereo_signal, sample_rate, delay):
        time.sleep(delay)
        # 播放合并后的音频信号
        sd.play(stereo_signal, sample_rate)
        sd.wait()  # 等待音频播放完毕

    def thread_play_3D_audio(self):
        def conv_audio(hrtf_left, hrtf_right, dist, wave):
            convolved_left = fftconvolve(wave[0, :], hrtf_left, mode='same') * (1 / ((dist / 1000) ** 2))
            convolved_right = fftconvolve(wave[1, :], hrtf_right, mode='same') * (1 / ((dist / 1000) ** 2))
            stereo_signal = np.vstack((convolved_left, convolved_right)).T
            return stereo_signal

        hrft = HRFT('utils/hrtf_nh94.sofa')
        wave, sample_rate = librosa.load('utils/sounds/beep2.wav', sr=None, mono=False)

        azimuth_pitchs = None
        while self.threads_running and self.threads_3D_playing:
            if self.azimuth_pitch_s is not None and azimuth_pitchs != self.azimuth_pitch_s:
                azimuth_pitchs = self.azimuth_pitch_s.copy()
                if len(azimuth_pitchs) > 3:
                    azimuth_pitchs.sort(key=lambda x: x[3])  # 根据距离排序
                    azimuth_pitchs = azimuth_pitchs[:3]  # 取前5个最近的
                threads = []
                for i, ap in enumerate(azimuth_pitchs):
                    name, azimuth_angle, pitch_angle, dist = ap
                    hrtf_left, hrtf_right = hrft.get_LR_HRFT(pitch_angle, azimuth_angle)
                    stereo_signal = conv_audio(hrtf_left, hrtf_right, dist, wave)
                    thread = threading.Thread(target=self.thread_play_audio,
                                              args=(stereo_signal, sample_rate, i * 0.1))
                    threads.append(thread)
                    thread.start()

                for thread in threads:
                    thread.join()

                self.signal_update_textEdit_terminal.emit(f'{time.time()}')
                self.signal_update_textEdit_terminal.emit(f'{azimuth_pitchs}')
                self.signal_update_textEdit_terminal.emit(f'{len(azimuth_pitchs)} 个音频已播放\n')
            else:
                time.sleep(0.5)

    def clicked_btn_run(self):
        self.clicked_btn_run_is_clicked = True
        if self.clicked_btn_run_is_clicked:
            self.btn_run.setStyleSheet("QPushButton {background-color: #8DBF8B;}")
            QApplication.processEvents()
            def generate_random_color(seed=None):
                """生成并返回一个随机颜色"""
                random.seed(seed)  # 设置随机数种子
                red = random.randint(0, 255)
                green = random.randint(0, 255)
                blue = random.randint(0, 255)
                return red, green, blue

            road_model = RoadBoundGetter(scale=0.3, density=10, pretrain="utils/road_model_maxmIOU75.pth")
            YOLO_model = YOLO('utils/yolov8l-seg.pt')

            tracks = dict()
            speeds = dict()
            collision_probability = dict()
            # 更新每帧画面
            while self.clicked_btn_run_is_clicked:
                t0 = time.time()
                img_BGR = self.img_BGR.copy()
                img_D = self.img_D.copy()
                BEV = BEV_Background_2()
                # seg = np.ones((360, 640, 3), np.uint8) * 100
                seg = img_BGR.copy()
                # 创建Rotation对象
                quat = np.array([self.imu.rotationVector.i,
                                 self.imu.rotationVector.j,
                                 self.imu.rotationVector.k,
                                 self.imu.rotationVector.real])
                rotation = R.from_quat(quat)

                # 绘制相机姿态
                center = (800, 800)
                tan_vfov = math.tan(math.radians(vfov / 2))
                tan_hfov = math.tan(math.radians(hfov / 2))
                vectors = np.array([
                    [100 * tan_vfov, 100 * tan_hfov, 100],  # 相机正前方100cm距离，左下的点
                    [-100 * tan_vfov, 100 * tan_hfov, 100],  # 相机正前方100cm距离，左上的点
                    [100 * tan_vfov, -100 * tan_hfov, 100],  # 相机正前方100cm距离，左上的点
                    [-100 * tan_vfov, -100 * tan_hfov, 100]  # 相机正前方100cm距离，右上的点
                ])
                down_left, up_left, down_right, up_right = rotation.apply(vectors)
                cv2.line(BEV, center, (int(down_left[0] + 0.5) + center[0], -int(down_left[1] + 0.5) + center[1]),
                         (8, 75, 30), 3)  # 左下角
                cv2.line(BEV, center, (int(down_right[0] + 0.5) + center[0], -int(down_right[1] + 0.5) + center[1]),
                         (8, 75, 30), 3)  # 右下角
                cv2.line(BEV, center, (int(up_left[0] + 0.5) + center[0], -int(up_left[1] + 0.5) + center[1]),
                         (79, 160, 39), 3)  # 左上角
                cv2.line(BEV, center, (int(up_right[0] + 0.5) + center[0], -int(up_right[1] + 0.5) + center[1]),
                         (79, 160, 39), 3)  # 右上角

                # 道路识别
                # road_masked = road_model(img_BGR)[0].cpu()
                # seg[road_masked == 1] = (233, 233, 233)
                # rows, cols = np.where((road_masked == 1) & (img_D != 0))
                # if len(rows) > 0 and len(cols) > 0:
                #     distances = img_D[rows, cols]
                #     bboxes = np.column_stack([cols, rows, cols, rows])
                #     Xs = getSpatialCoordinates(distances, bboxes, 'x') / 10
                #     Ys = getSpatialCoordinates(distances, bboxes, 'y') / 10
                #     Zs = distances / 10
                #     points = np.column_stack((Ys, Xs, Zs))
                #     # 四元数旋转的向量化应用
                #     rotated_points = rotation.apply(points)
                #     # 转换和调整坐标
                #     coords = rotated_points[:, :2]  # 获取前两个维度
                #     coords[:, 0] += 800  # X坐标偏移
                #     coords[:, 1] = -coords[:, 1] + 800  # Y坐标偏移并反向
                #     # 将浮点坐标转换为整数
                #     coords = coords.astype(int)
                #     # 过滤出在BEV图像边界内的点
                #     mask = (coords[:, 0] >= 0) & (coords[:, 0] < BEV.shape[1]) & \
                #            (coords[:, 1] >= 0) & (coords[:, 1] < BEV.shape[0])
                #     # 应用mask过滤坐标
                #     filtered_coords = coords[mask]
                #     # 为过滤后的坐标赋值颜色
                #     BEV[filtered_coords[:, 1], filtered_coords[:, 0]] = (233, 233, 233)

                # 环境感知
                object_result = YOLO_model.track(img_BGR, persist=True, verbose=False)[0]
                azimuth_pitch = []
                for i in range(len(object_result)):
                    box = object_result.boxes[i]
                    mask = object_result.masks[i].data[0][12: -12, :].int().cpu()
                    bbox = box.xyxy.cpu().round().int().tolist()[0]
                    name = object_result.names[box.cls.cpu().round().int().tolist()[0]]
                    if name not in ['person']:
                        continue

                    color = generate_random_color(int(box.cls.cpu()))
                    color = generate_random_color(int(box.cls.cpu()))
                    color_expanded = np.zeros_like(img_BGR)
                    color_expanded[:, :] = color
                    alpha = 0.5
                    seg[mask == 1] = (1 - alpha) * img_BGR[mask == 1] + alpha * color_expanded[mask == 1]

                    rows, cols = np.where((mask == 1) & (img_D != 0))
                    if len(rows) > 0 and len(cols) > 0:
                        distances = img_D[rows, cols]
                        # 计算百分位数
                        percentile_low = 5
                        percentile_high = 85
                        low_threshold = np.percentile(distances, percentile_low)
                        high_threshold = np.percentile(distances, percentile_high)
                        # 选择符合条件的数据
                        selected_rows = rows[(distances >= low_threshold) & (distances <= high_threshold)]
                        selected_cols = cols[(distances >= low_threshold) & (distances <= high_threshold)]
                        selected_distances = distances[(distances >= low_threshold) & (distances <= high_threshold)]

                        col = np.median(selected_cols)
                        row = np.median(selected_rows)
                        dist = np.median(selected_distances)
                        azimuth_angle, pitch_angle = calculate_azimuth_pitch(col, row, dist)
                        azimuth_pitch.append([name, azimuth_angle, pitch_angle, dist])

                        bboxes = np.column_stack([selected_cols, selected_rows, selected_cols, selected_rows])
                        Xs = getSpatialCoordinates(selected_distances, bboxes, 'x') / 10
                        Ys = getSpatialCoordinates(selected_distances, bboxes, 'y') / 10
                        Zs = selected_distances / 10
                        points = np.column_stack((Ys, Xs, Zs))
                        # 四元数旋转的向量化应用
                        rotated_points = rotation.apply(points)
                        # 转换和调整坐标
                        coords = rotated_points[:, :2]  # 获取前两个维度
                        coords[:, 0] += 800  # X坐标偏移
                        coords[:, 1] = -coords[:, 1] + 800  # Y坐标偏移并反向
                        # 将浮点坐标转换为整数
                        coords = coords.astype(int)
                        # 过滤出在BEV图像边界内的点
                        mask = (coords[:, 0] >= 0) & (coords[:, 0] < BEV.shape[1]) & \
                               (coords[:, 1] >= 0) & (coords[:, 1] < BEV.shape[0])
                        # 应用mask过滤坐标
                        obj_filtered_coords = coords[mask]
                        # 确保过滤后的坐标有效且不为空
                        if obj_filtered_coords.size > 0:
                            BEV[obj_filtered_coords[:, 1], obj_filtered_coords[:, 0]] = color
                            # BEV[filtered_coords[:, 1], filtered_coords[:, 0]] = img_BGR[selected_rows, selected_cols]
                            ids = object_result.boxes.id
                            if ids is not None:
                                ids = ids.int().cpu().tolist()
                                id = ids[i]
                                if id not in tracks.keys():
                                    tracks[id] = list()

                                tracks[id].append((np.median(obj_filtered_coords[:, 0]),  # X
                                                   np.median(obj_filtered_coords[:, 1]),  # Y
                                                   t0))  # T
                                cv2.putText(img_BGR, f"{id} {name} {np.median(selected_distances):.1f}mm",
                                            (bbox[0] + 5, bbox[1] + 20),
                                            cv2.FONT_HERSHEY_TRIPLEX, 0.8, color)
                            else:
                                cv2.putText(img_BGR, f"-1 {name} {np.median(selected_distances):.1f}mm",
                                            (bbox[0] + 5, bbox[1] + 20),
                                            cv2.FONT_HERSHEY_TRIPLEX, 0.8, color)

                self.azimuth_pitch_s = azimuth_pitch.copy()
                azimuth_pitch.clear()

                tracks_draw = tracks.copy()
                for id in tracks_draw:
                    track = tracks_draw[id]
                    if track:
                        t_old = track[0][2]
                        t_new = track[-1][2]
                        if t0 - t_new > 3:  # 有陈放3秒以上的轨迹删除
                            del tracks[id]
                        elif len(track) > 90 or t0 - t_old > 3:
                            tracks[id].pop(0)

                for id in tracks:
                    track = tracks[id]
                    for i in range(len(track) - 1):  # 遍历轨迹中的每个点，但不包括最后一个点
                        x1, y1, t1 = track[i]
                        x2, y2, t2 = track[i + 1]
                        cv2.line(BEV, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 200), thickness=2)
                    if len(track) >= 5:
                        track_5 = tracks[id][-5:]  # 获取最后5个点
                        velocities_x = []
                        velocities_y = []
                        for i in range(len(track_5) - 1):
                            x1, y1, t1 = track_5[i]
                            x2, y2, t2 = track_5[i + 1]
                            distance_x = x2 - x1
                            distance_y = y2 - y1
                            time_difference = max(0.01, t2 - t1)
                            velocity_x = distance_x / time_difference
                            velocity_y = distance_y / time_difference
                            velocities_x.append(velocity_x)
                            velocities_y.append(velocity_y)
                        speeds[id] = (np.mean(velocities_x), np.mean(velocities_y),
                                      math.sqrt(np.mean(velocities_x) ** 2 + np.mean(velocities_y) ** 2))  # 计算平均速度
                        pro = -1
                        for t in range(1, 16):
                            predicted_tracks = (
                                speeds[id][0] * t / 10 + tracks[id][-1][0], speeds[id][1] * t / 10 + tracks[id][-1][1])
                            dist = math.sqrt((predicted_tracks[0] - 800) ** 2 + (predicted_tracks[1] - 800) ** 2)
                            if dist < 50:
                                pro = t
                                break
                        if pro == -1:
                            collision_probability[id] = 0
                        else:
                            collision_probability[id] = (16 - pro) / 15

                        cv2.line(BEV, (int(tracks[id][-1][0]), int(tracks[id][-1][1])),
                                 (int(predicted_tracks[0]), int(predicted_tracks[1])), (233, 0, 0), thickness=3)

                speed_ = speeds.copy()
                for id in speed_:
                    if id not in tracks.keys():
                        del speeds[id], collision_probability[id]

                # 按照速度对ID进行排序
                sorted_ids = sorted(speeds.keys(), key=lambda x: speeds[x][2], reverse=True)
                fig, ax = plt.subplots(figsize=(2.5, 5.5))
                ax.set_xlim(0, 200)
                if sorted_ids:
                    labels = []  # 创建一个空列表来存储标签
                    for i, id in enumerate(reversed(sorted_ids)):
                        if collision_probability[id] < 0.1:
                            color = '#8DBF8B'
                            label = 'Low Risk'
                        elif collision_probability[id] < 0.3:
                            color = '#FDB15D'
                            label = 'Moderate Risk'
                        elif collision_probability[id] < 0.5:
                            color = '#F95A37'
                            label = 'High Risk'
                        else:
                            color = '#E04255'
                            label = 'Severe Risk'
                        ax.barh(i, speeds[id][2], height=0.8, color=color, label=label if label not in labels else "")
                        labels.append(label)  # 将这个标签添加到列表中
                    ax.set_yticks(range(len(sorted_ids)))
                    ax.set_yticklabels(reversed(sorted_ids))
                    ax.legend()
                fig.canvas.draw()
                img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
                img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                plt.close()

                height, width, channel = img.shape  # 例如 360, 640
                q_img = QImage(img.data, width, height, channel * width, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
                scaled_pixmap_label_collision_probability_data = pixmap.scaled(
                    self.label_collision_probability_data.width(),
                    self.label_collision_probability_data.height(),
                    Qt.KeepAspectRatio)

                img_BGR = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
                height, width, channel = img_BGR.shape  # 例如 360, 640
                q_img = QImage(img_BGR.data, width, height, channel * width, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
                scaled_pixmap_label_object_detection = pixmap.scaled(self.label_object_detection.width(),
                                                                     self.label_object_detection.height(),
                                                                     Qt.KeepAspectRatio)

                seg = cv2.cvtColor(seg, cv2.COLOR_BGR2RGB)
                height, width, channel = seg.shape  # 例如 360, 640
                q_img = QImage(seg.data, width, height, channel * width, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
                scaled_pixmap_label_semantic_segmentation = pixmap.scaled(self.label_semantic_segmentation.width(),
                                                                          self.label_semantic_segmentation.height(),
                                                                          Qt.KeepAspectRatio)

                img_D[img_D == 0] = 10000
                depth255 = (img_D * (255.0 / 10000)).astype(np.uint8)
                depthCOLORMAP = cv2.applyColorMap((255 - depth255), cv2.COLORMAP_MAGMA)
                depthCOLORMAP = cv2.cvtColor(depthCOLORMAP, cv2.COLOR_BGR2RGB)
                height, width, channel = depthCOLORMAP.shape  # 例如 360, 640
                q_img = QImage(depthCOLORMAP.data, width, height, channel * width, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
                scaled_pixmap_label_depth = pixmap.scaled(self.label_depth.width(), self.label_depth.height(),
                                                          Qt.KeepAspectRatio)

                # X = int(800/2)
                BEV = cv2.cvtColor(BEV, cv2.COLOR_BGR2RGB)
                # height, width, _ = BEV.shape
                # start_y = height // 2 - X
                # end_y = height // 2 + X
                # start_x = width // 2 - X
                # end_x = width // 2 + X
                # BEV = BEV[start_y:end_y, start_x:end_x]
                height, width, channel = BEV.shape  # 例如 360, 640
                q_img = QImage(BEV.data, width, height, channel * width, QImage.Format_RGB888)
                # q_img = QImage(bytes(BEV.data), width, height, channel * width, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
                scaled_pixmap_label_BEV = pixmap.scaled(self.label_BEV.width(), self.label_BEV.height(),
                                                        Qt.KeepAspectRatio)

                if collision_probability:
                    cp = max(collision_probability.values())
                    if cp < 0.1:
                        color = '#8DBF8B'
                    elif cp < 0.3:
                        color = '#FDB15D'
                    elif cp < 0.5:
                        color = '#F95A37'
                    else:
                        color = '#E04255'
                    self.label_collision_probability.setStyleSheet(f"background-color: {color};")

                self.signal_update_label_object_detection.emit(scaled_pixmap_label_object_detection)
                self.signal_update_label_semantic_segmentation.emit(scaled_pixmap_label_semantic_segmentation)
                self.signal_update_label_depth.emit(scaled_pixmap_label_depth)
                self.signal_update_label_BEV.emit(scaled_pixmap_label_BEV)
                self.signal_update_label_collision_probability_data.emit(scaled_pixmap_label_collision_probability_data)
                QApplication.processEvents()

            self.signal_clean_label_collision_probability.emit()
            self.signal_clean_label_object_detection.emit()
            self.signal_clean_label_semantic_segmentation.emit()
            self.signal_clean_label_depth.emit()
            self.signal_clean_label_BEV.emit()
            self.signal_clean_label_collision_probability_data.emit()
            self.signal_clean_textEdit_terminal.emit()
            QApplication.processEvents()


    def clicked_btn_play_3D_audio(self):
        self.clicked_btn_play_3D_audio_is_clicked = ~self.clicked_btn_play_3D_audio_is_clicked
        if self.clicked_btn_play_3D_audio_is_clicked and self.clicked_btn_run_is_clicked:
            self.threads_3D_playing = True
            self.btn_play_3D_audio.setStyleSheet("QPushButton {background-color: #8DBF8B;}")
            QApplication.processEvents()
            threading.Thread(target=self.thread_play_3D_audio).start()
        else:
            self.threads_3D_playing = False
            self.btn_play_3D_audio.setStyleSheet("QPushButton {}")

    def clicked_btn_stop(self):
        self.clicked_btn_run_is_clicked = False
        self.clicked_btn_play_3D_audio_is_clicked = False
        self.threads_3D_playing = False
        self.btn_run.setStyleSheet("QPushButton {}")
        self.btn_play_3D_audio.setStyleSheet("QPushButton {}")
        self.btn_stop.setStyleSheet("QPushButton {background-color: #E04255;}")
        QApplication.processEvents()
        time.sleep(1)
        self.btn_stop.setStyleSheet("QPushButton {}")

    def closeEvent(self, event):
        self.threads_running = False
        self.windowClosed.emit()
        event.accept()


class cloud_service_window(QWidget):
    windowClosed = pyqtSignal()
    signal_label_collision_probability = pyqtSignal(str)
    signal_update_label_object_detection = pyqtSignal(QPixmap)
    signal_update_label_semantic_segmentation = pyqtSignal(QPixmap)
    signal_update_label_depth = pyqtSignal(QPixmap)
    signal_update_label_BEV = pyqtSignal(QPixmap)
    signal_update_label_collision_probability_data = pyqtSignal(QPixmap)
    signal_update_textEdit_terminal = pyqtSignal(str)

    signal_clean_label_collision_probability = pyqtSignal()
    signal_clean_label_object_detection = pyqtSignal()
    signal_clean_label_semantic_segmentation = pyqtSignal()
    signal_clean_label_depth = pyqtSignal()
    signal_clean_label_BEV = pyqtSignal()
    signal_clean_label_collision_probability_data = pyqtSignal()
    signal_clean_textEdit_terminal = pyqtSignal()

    def __init__(self):
        super().__init__()
        # 设置客户端参数
        HOST = '127.0.0.1'
        PORT = 65432
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((HOST, PORT))

        self.azimuth_pitch_s = None
        self.threads_running = False
        self.threads_3D_playing = False
        self.clicked_btn_run_is_clicked = False
        self.clicked_btn_play_3D_audio_is_clicked = False
        self.playing = False
        self.initUI()
        self.initFunc()
        self.initdata()
        self.initDevice()

        self.ServerApp = ServerApp()
        # self.ServerApp.windowClosed.connect(self.on_second_window_closed)
        self.ServerApp.setAttribute(Qt.WA_DeleteOnClose)  # 确保窗口关闭时释放资源
        # self.ServerApp.setWindowModality(Qt.ApplicationModal)  # 设置窗口为应用程序级别的模态
        self.ServerApp.show()

    def initUI(self):
        self.setWindowTitle("轨迹追踪及碰撞预警模块")
        self.setFixedSize(1600, 900)

        self.label_collision_probability = QLabel(self)
        self.label_collision_probability.setGeometry(10, 10, 1580, 40)  # x, y, width, height
        self.label_collision_probability.setText("Collision Probability")
        self.label_collision_probability.setStyleSheet("background-color: #FAF5E4;")

        self.label_object_detection = QLabel(self)
        self.label_object_detection.setGeometry(10, 60, 480, 270)  # x, y, width, height
        self.label_object_detection.setText("Object Detection")
        self.label_object_detection.setStyleSheet("background-color: #FAF5E4;")

        self.label_semantic_segmentation = QLabel(self)
        self.label_semantic_segmentation.setGeometry(10, 340, 480, 270)  # x, y, width, height
        self.label_semantic_segmentation.setText("Semantic Segmentation")
        self.label_semantic_segmentation.setStyleSheet("background-color: #FAF5E4;")

        self.label_depth = QLabel(self)
        self.label_depth.setGeometry(10, 620, 480, 270)  # x, y, width, height
        self.label_depth.setText("Depth")
        self.label_depth.setStyleSheet("background-color: #FAF5E4;")

        self.label_BEV = QLabel(self)
        self.label_BEV.setGeometry(500, 60, 830, 830)  # x, y, width, height
        self.label_BEV.setText("BEV (Bird's-eye view)")
        self.label_BEV.setStyleSheet("background-color: #FAF5E4;")

        self.label_collision_probability_data = QLabel(self)
        self.label_collision_probability_data.setGeometry(1340, 60, 250, 550)  # x, y, width, height
        self.label_collision_probability_data.setText("Collision Probability")
        self.label_collision_probability_data.setStyleSheet("background-color: #FAF5E4;")

        self.textEdit_terminal = QTextEdit(self)
        self.textEdit_terminal.setGeometry(1340, 620, 250, 230)
        self.textEdit_terminal.setReadOnly(True)  # 设置文本框为只读

        self.btn_run = QPushButton("开始", self)
        self.btn_run.setGeometry(1340, 860, 80, 30)
        self.btn_play_3D_audio = QPushButton("3D 音频", self)
        self.btn_play_3D_audio.setGeometry(1425, 860, 80, 30)
        self.btn_stop = QPushButton("停止", self)
        self.btn_stop.setGeometry(1510, 860, 80, 30)

    def initFunc(self):
        self.signal_label_collision_probability.connect(self.label_collision_probability.setStyleSheet)
        self.signal_clean_label_collision_probability.connect(
            lambda: self.label_collision_probability.setText("Collision Probability"))
        self.signal_clean_label_collision_probability.connect(
            lambda: self.label_collision_probability.setStyleSheet("background-color: #FAF5E4;"))

        self.signal_update_label_object_detection.connect(self.label_object_detection.setPixmap)
        self.signal_clean_label_object_detection.connect(
            lambda: self.label_object_detection.setText("Object Detection"))
        self.signal_clean_label_object_detection.connect(
            lambda: self.label_object_detection.setStyleSheet("background-color: #FAF5E4;"))

        self.signal_update_label_semantic_segmentation.connect(self.label_semantic_segmentation.setPixmap)
        self.signal_clean_label_semantic_segmentation.connect(
            lambda: self.label_semantic_segmentation.setText("Semantic Segmentation"))
        self.signal_clean_label_semantic_segmentation.connect(
            lambda: self.label_semantic_segmentation.setStyleSheet("background-color: #FAF5E4;"))

        self.signal_update_label_depth.connect(self.label_depth.setPixmap)
        self.signal_clean_label_depth.connect(
            lambda: self.label_depth.setText("Depth"))
        self.signal_clean_label_depth.connect(
            lambda: self.label_depth.setStyleSheet("background-color: #FAF5E4;"))

        self.signal_update_label_BEV.connect(self.label_BEV.setPixmap)
        self.signal_clean_label_BEV.connect(
            lambda: self.label_BEV.setText("BEV (Bird's-eye view)"))
        self.signal_clean_label_BEV.connect(
            lambda: self.label_BEV.setStyleSheet("background-color: #FAF5E4;"))

        self.signal_update_label_collision_probability_data.connect(self.label_collision_probability_data.setPixmap)
        self.signal_clean_label_collision_probability_data.connect(
            lambda: self.label_collision_probability_data.setText("Collision Probability"))
        self.signal_clean_label_collision_probability_data.connect(
            lambda: self.label_collision_probability_data.setStyleSheet("background-color: #FAF5E4;"))

        self.signal_update_textEdit_terminal.connect(
            lambda text: self.textEdit_terminal.append(f"{text}"))
        self.signal_clean_textEdit_terminal.connect(self.textEdit_terminal.clear)

        self.btn_run.clicked.connect(self.clicked_btn_run)
        self.btn_play_3D_audio.clicked.connect(self.clicked_btn_play_3D_audio)
        self.btn_stop.clicked.connect(self.clicked_btn_stop)

    def initdata(self):
        self.msgGrp = None
        self.img_BGR = None
        self.img_D = None
        self.imu = None

        self.azimuth_pitch_s = None

    def initDevice(self):
        self.queue = oak_device.getOutputQueue("xoutGrp", 10, False)
        self.threads_running = True
        threading.Thread(target=self.thread_refresh_msgGrp).start()

    def thread_refresh_msgGrp(self):
        while self.threads_running:
            self.msgGrp = self.queue.get()
            self.img_BGR = self.msgGrp['color'].getCvFrame()
            self.img_D = self.msgGrp['depth'].getCvFrame()
            self.imu = self.msgGrp['imu'].packets[0]

    def thread_play_audio(self, stereo_signal, sample_rate, delay):
        time.sleep(delay)
        # 播放合并后的音频信号
        sd.play(stereo_signal, sample_rate)
        sd.wait()  # 等待音频播放完毕

    def thread_play_3D_audio(self):
        def conv_audio(hrtf_left, hrtf_right, dist, wave):
            convolved_left = fftconvolve(wave[0, :], hrtf_left, mode='same') * (1 / ((dist / 1000) ** 2))
            convolved_right = fftconvolve(wave[1, :], hrtf_right, mode='same') * (1 / ((dist / 1000) ** 2))
            stereo_signal = np.vstack((convolved_left, convolved_right)).T
            return stereo_signal

        hrft = HRFT('utils/hrtf_nh94.sofa')
        wave, sample_rate = librosa.load('utils/sounds/beep2.wav', sr=None, mono=False)

        azimuth_pitchs = None
        while self.threads_running and self.threads_3D_playing:
            if self.azimuth_pitch_s is not None and azimuth_pitchs != self.azimuth_pitch_s:
                azimuth_pitchs = self.azimuth_pitch_s.copy()
                if len(azimuth_pitchs) > 3:
                    azimuth_pitchs.sort(key=lambda x: x[3])  # 根据距离排序
                    azimuth_pitchs = azimuth_pitchs[:3]  # 取前5个最近的
                threads = []
                for i, ap in enumerate(azimuth_pitchs):
                    name, azimuth_angle, pitch_angle, dist = ap
                    hrtf_left, hrtf_right = hrft.get_LR_HRFT(pitch_angle, azimuth_angle)
                    stereo_signal = conv_audio(hrtf_left, hrtf_right, dist, wave)
                    thread = threading.Thread(target=self.thread_play_audio,
                                              args=(stereo_signal, sample_rate, i * 0.1))
                    threads.append(thread)
                    thread.start()

                for thread in threads:
                    thread.join()

                self.signal_update_textEdit_terminal.emit(f'{time.time()}')
                self.signal_update_textEdit_terminal.emit(f'{azimuth_pitchs}')
                self.signal_update_textEdit_terminal.emit(f'{len(azimuth_pitchs)} 个音频已播放\n')
            else:
                time.sleep(0.5)

    def clicked_btn_run(self):
        if not self.clicked_btn_run_is_clicked:
            self.clicked_btn_run_is_clicked = True

            def generate_random_color(seed=None):
                """生成并返回一个随机颜色"""
                random.seed(seed)  # 设置随机数种子
                red = random.randint(0, 255)
                green = random.randint(0, 255)
                blue = random.randint(0, 255)
                return (red, green, blue)

            async def send_images(rgb_image):
                # 压缩RGB图像
                _, rgb_image_compressed = cv2.imencode('.jpg', rgb_image, [cv2.IMWRITE_JPEG_QUALITY, 80])
                rgb_image_compressed = rgb_image_compressed.tobytes()

                # 构建数据包
                # data = {'rgb': rgb_image_compressed, 'depth': depth_image}
                data = {'rgb': rgb_image_compressed}
                serialized_data = pickle.dumps(data)
                serialized_data += b'$END#'  # 添加结束标记

                loop = asyncio.get_event_loop()
                await loop.sock_sendall(self.client_socket, serialized_data)

                # 接收结果
                received_data = b''
                while True:
                    packet = await loop.sock_recv(self.client_socket, 4096)
                    if packet.endswith(b'#END$'):
                        received_data += packet[:-5]
                        break
                    received_data += packet
                    QApplication.processEvents()
                # 反序列化结果
                result = pickle.loads(received_data)
                return result

            async def capture_and_send():
                try:
                    tracks = dict()
                    speeds = dict()
                    collision_probability = dict()
                    # 更新每帧画面
                    while self.clicked_btn_run_is_clicked and self.threads_running:
                        t0 = time.time()
                        img_BGR = self.img_BGR.copy()
                        img_D = self.img_D.copy()
                        BEV = BEV_Background_2()
                        seg = np.ones((360, 640, 3), np.uint8) * 100
                        # 创建Rotation对象
                        quat = np.array([self.imu.rotationVector.i,
                                         self.imu.rotationVector.j,
                                         self.imu.rotationVector.k,
                                         self.imu.rotationVector.real])
                        rotation = R.from_quat(quat)
                        # 绘制相机姿态
                        center = (800, 800)
                        tan_vfov = math.tan(math.radians(vfov / 2))
                        tan_hfov = math.tan(math.radians(hfov / 2))
                        vectors = np.array([
                            [100 * tan_vfov, 100 * tan_hfov, 100],  # 相机正前方100cm距离，左下的点
                            [-100 * tan_vfov, 100 * tan_hfov, 100],  # 相机正前方100cm距离，左上的点
                            [100 * tan_vfov, -100 * tan_hfov, 100],  # 相机正前方100cm距离，左上的点
                            [-100 * tan_vfov, -100 * tan_hfov, 100]  # 相机正前方100cm距离，右上的点
                        ])
                        down_left, up_left, down_right, up_right = rotation.apply(vectors)
                        cv2.line(BEV, center,
                                 (int(down_left[0] + 0.5) + center[0], -int(down_left[1] + 0.5) + center[1]),
                                 (8, 75, 30), 3)  # 左下角
                        cv2.line(BEV, center,
                                 (int(down_right[0] + 0.5) + center[0], -int(down_right[1] + 0.5) + center[1]),
                                 (8, 75, 30), 3)  # 右下角
                        cv2.line(BEV, center, (int(up_left[0] + 0.5) + center[0], -int(up_left[1] + 0.5) + center[1]),
                                 (79, 160, 39), 3)  # 左上角
                        cv2.line(BEV, center, (int(up_right[0] + 0.5) + center[0], -int(up_right[1] + 0.5) + center[1]),
                                 (79, 160, 39), 3)  # 右上角


                        # 发送图像并接收结果
                        received_data = await send_images(img_BGR)

                        road_masked = received_data['road']
                        seg[road_masked == 1] = (233, 233, 233)
                        rows, cols = np.where((road_masked == 1) & (img_D != 0))
                        if len(rows) > 0 and len(cols) > 0:
                            distances = img_D[rows, cols]
                            bboxes = np.column_stack([cols, rows, cols, rows])
                            Xs = getSpatialCoordinates(distances, bboxes, 'x') / 10
                            Ys = getSpatialCoordinates(distances, bboxes, 'y') / 10
                            Zs = distances / 10
                            points = np.column_stack((Ys, Xs, Zs))
                            # 四元数旋转的向量化应用
                            rotated_points = rotation.apply(points)
                            # 转换和调整坐标
                            coords = rotated_points[:, :2]  # 获取前两个维度
                            coords[:, 0] += 800  # X坐标偏移
                            coords[:, 1] = -coords[:, 1] + 800  # Y坐标偏移并反向
                            # 将浮点坐标转换为整数
                            coords = coords.astype(int)
                            # 过滤出在BEV图像边界内的点
                            mask = (coords[:, 0] >= 0) & (coords[:, 0] < BEV.shape[1]) & \
                                   (coords[:, 1] >= 0) & (coords[:, 1] < BEV.shape[0])
                            # 应用mask过滤坐标
                            filtered_coords = coords[mask]
                            # 为过滤后的坐标赋值颜色
                            BEV[filtered_coords[:, 1], filtered_coords[:, 0]] = (233, 233, 233)

                        object_result = received_data['yolo']
                        azimuth_pitch = []
                        for i in range(len(object_result)):
                            box = object_result.boxes[i]
                            mask = object_result.masks[i].data[0][12: -12, :].int().cpu()
                            bbox = box.xyxy.cpu().round().int().tolist()[0]
                            name = object_result.names[box.cls.cpu().round().int().tolist()[0]]
                            if name not in objs:
                                continue

                            color = generate_random_color(int(box.cls.cpu()))
                            seg[mask == 1] = color
                            cv2.rectangle(img_BGR, bbox[:2], bbox[2:], color, 2)

                            rows, cols = np.where((mask == 1) & (img_D != 0))
                            if len(rows) > 0 and len(cols) > 0:
                                distances = img_D[rows, cols]
                                # 计算百分位数
                                percentile_low = 5
                                percentile_high = 85
                                low_threshold = np.percentile(distances, percentile_low)
                                high_threshold = np.percentile(distances, percentile_high)
                                # 选择符合条件的数据
                                selected_rows = rows[(distances >= low_threshold) & (distances <= high_threshold)]
                                selected_cols = cols[(distances >= low_threshold) & (distances <= high_threshold)]
                                selected_distances = distances[
                                    (distances >= low_threshold) & (distances <= high_threshold)]

                                col = np.median(selected_cols)
                                row = np.median(selected_rows)
                                dist = np.median(selected_distances)
                                azimuth_angle, pitch_angle = calculate_azimuth_pitch(col, row, dist)
                                azimuth_pitch.append([name, azimuth_angle, pitch_angle, dist])

                                bboxes = np.column_stack([selected_cols, selected_rows, selected_cols, selected_rows])
                                Xs = getSpatialCoordinates(selected_distances, bboxes, 'x') / 10
                                Ys = getSpatialCoordinates(selected_distances, bboxes, 'y') / 10
                                Zs = selected_distances / 10
                                points = np.column_stack((Ys, Xs, Zs))
                                # 四元数旋转的向量化应用
                                rotated_points = rotation.apply(points)
                                # 转换和调整坐标
                                coords = rotated_points[:, :2]  # 获取前两个维度
                                coords[:, 0] += 800  # X坐标偏移
                                coords[:, 1] = -coords[:, 1] + 800  # Y坐标偏移并反向
                                # 将浮点坐标转换为整数
                                coords = coords.astype(int)
                                # 过滤出在BEV图像边界内的点
                                mask = (coords[:, 0] >= 0) & (coords[:, 0] < BEV.shape[1]) & \
                                       (coords[:, 1] >= 0) & (coords[:, 1] < BEV.shape[0])
                                # 应用mask过滤坐标
                                obj_filtered_coords = coords[mask]
                                # 确保过滤后的坐标有效且不为空
                                if obj_filtered_coords.size > 0:
                                    BEV[obj_filtered_coords[:, 1], obj_filtered_coords[:, 0]] = color
                                    # BEV[filtered_coords[:, 1], filtered_coords[:, 0]] = img_BGR[selected_rows, selected_cols]
                                    ids = object_result.boxes.id
                                    if ids is not None:
                                        ids = ids.int().cpu().tolist()
                                        id = ids[i]
                                        if id not in tracks.keys():
                                            tracks[id] = list()

                                        tracks[id].append((np.median(obj_filtered_coords[:, 0]),  # X
                                                           np.median(obj_filtered_coords[:, 1]),  # Y
                                                           t0))  # T
                                        cv2.putText(img_BGR, f"{id} {name} {np.median(selected_distances):.1f}mm",
                                                    (bbox[0] + 5, bbox[1] + 20),
                                                    cv2.FONT_HERSHEY_TRIPLEX, 0.8, color)
                                    else:
                                        cv2.putText(img_BGR, f"-1 {name} {np.median(selected_distances):.1f}mm",
                                                    (bbox[0] + 5, bbox[1] + 20),
                                                    cv2.FONT_HERSHEY_TRIPLEX, 0.8, color)

                        self.azimuth_pitch_s = azimuth_pitch.copy()
                        azimuth_pitch.clear()

                        tracks_draw = tracks.copy()
                        for id in tracks_draw:
                            track = tracks_draw[id]
                            if track:
                                t_old = track[0][2]
                                t_new = track[-1][2]
                                if t0 - t_new > 3:  # 有陈放3秒以上的轨迹删除
                                    del tracks[id]
                                elif len(track) > 90 or t0 - t_old > 3:
                                    tracks[id].pop(0)

                        for id in tracks:
                            track = tracks[id]
                            for i in range(len(track) - 1):  # 遍历轨迹中的每个点，但不包括最后一个点
                                x1, y1, t1 = track[i]
                                x2, y2, t2 = track[i + 1]
                                cv2.line(BEV, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 200), thickness=2)
                            if len(track) >= 5:
                                track_5 = tracks[id][-5:]  # 获取最后5个点
                                velocities_x = []
                                velocities_y = []
                                for i in range(len(track_5) - 1):
                                    x1, y1, t1 = track_5[i]
                                    x2, y2, t2 = track_5[i + 1]
                                    distance_x = x2 - x1
                                    distance_y = y2 - y1
                                    time_difference = max(0.01, t2 - t1)
                                    velocity_x = distance_x / time_difference
                                    velocity_y = distance_y / time_difference
                                    velocities_x.append(velocity_x)
                                    velocities_y.append(velocity_y)
                                speeds[id] = (np.mean(velocities_x), np.mean(velocities_y),
                                              math.sqrt(
                                                  np.mean(velocities_x) ** 2 + np.mean(velocities_y) ** 2))  # 计算平均速度
                                pro = -1
                                for t in range(1, 16):
                                    predicted_tracks = (
                                        speeds[id][0] * t / 10 + tracks[id][-1][0],
                                        speeds[id][1] * t / 10 + tracks[id][-1][1])
                                    dist = math.sqrt(
                                        (predicted_tracks[0] - 800) ** 2 + (predicted_tracks[1] - 800) ** 2)
                                    if dist < 50:
                                        pro = t
                                        break
                                if pro == -1:
                                    collision_probability[id] = 0
                                else:
                                    collision_probability[id] = (16 - pro) / 15

                                cv2.line(BEV, (int(tracks[id][-1][0]), int(tracks[id][-1][1])),
                                         (int(predicted_tracks[0]), int(predicted_tracks[1])), (233, 0, 0), thickness=3)

                        speed_ = speeds.copy()
                        for id in speed_:
                            if id not in tracks.keys():
                                del speeds[id], collision_probability[id]

                        # 按照速度对ID进行排序
                        sorted_ids = sorted(speeds.keys(), key=lambda x: speeds[x][2], reverse=True)
                        fig, ax = plt.subplots(figsize=(2.5, 5.5))
                        ax.set_xlim(0, 200)
                        if sorted_ids:
                            labels = []  # 创建一个空列表来存储标签
                            for i, id in enumerate(reversed(sorted_ids)):
                                if collision_probability[id] < 0.1:
                                    color = '#8DBF8B'
                                    label = 'Low Risk'
                                elif collision_probability[id] < 0.3:
                                    color = '#FDB15D'
                                    label = 'Moderate Risk'
                                elif collision_probability[id] < 0.5:
                                    color = '#F95A37'
                                    label = 'High Risk'
                                else:
                                    color = '#E04255'
                                    label = 'Severe Risk'
                                ax.barh(i, speeds[id][2], height=0.8, color=color,
                                        label=label if label not in labels else "")
                                labels.append(label)  # 将这个标签添加到列表中
                            ax.set_yticks(range(len(sorted_ids)))
                            ax.set_yticklabels(reversed(sorted_ids))
                            ax.legend()
                        fig.canvas.draw()
                        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
                        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
                        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                        plt.close()

                        height, width, channel = img.shape  # 例如 360, 640
                        q_img = QImage(img.data, width, height, channel * width, QImage.Format_RGB888)
                        pixmap = QPixmap.fromImage(q_img)
                        scaled_pixmap_label_collision_probability_data = pixmap.scaled(
                            self.label_collision_probability_data.width(),
                            self.label_collision_probability_data.height(),
                            Qt.KeepAspectRatio)

                        img_BGR = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
                        height, width, channel = img_BGR.shape  # 例如 360, 640
                        q_img = QImage(img_BGR.data, width, height, channel * width, QImage.Format_RGB888)
                        pixmap = QPixmap.fromImage(q_img)
                        scaled_pixmap_label_object_detection = pixmap.scaled(self.label_object_detection.width(),
                                                                             self.label_object_detection.height(),
                                                                             Qt.KeepAspectRatio)

                        seg = cv2.cvtColor(seg, cv2.COLOR_BGR2RGB)
                        height, width, channel = seg.shape  # 例如 360, 640
                        q_img = QImage(seg.data, width, height, channel * width, QImage.Format_RGB888)
                        pixmap = QPixmap.fromImage(q_img)
                        scaled_pixmap_label_semantic_segmentation = pixmap.scaled(
                            self.label_semantic_segmentation.width(),
                            self.label_semantic_segmentation.height(),
                            Qt.KeepAspectRatio)

                        img_D[img_D == 0] = 10000
                        depth255 = (img_D * (255.0 / 10000)).astype(np.uint8)
                        depthCOLORMAP = cv2.applyColorMap((255 - depth255), cv2.COLORMAP_MAGMA)
                        depthCOLORMAP = cv2.cvtColor(depthCOLORMAP, cv2.COLOR_BGR2RGB)
                        height, width, channel = depthCOLORMAP.shape  # 例如 360, 640
                        q_img = QImage(depthCOLORMAP.data, width, height, channel * width, QImage.Format_RGB888)
                        pixmap = QPixmap.fromImage(q_img)
                        scaled_pixmap_label_depth = pixmap.scaled(self.label_depth.width(), self.label_depth.height(),
                                                                  Qt.KeepAspectRatio)

                        # X = int(800/2)
                        BEV = cv2.cvtColor(BEV, cv2.COLOR_BGR2RGB)
                        # height, width, _ = BEV.shape
                        # start_y = height // 2 - X
                        # end_y = height // 2 + X
                        # start_x = width // 2 - X
                        # end_x = width // 2 + X
                        # BEV = BEV[start_y:end_y, start_x:end_x]
                        height, width, channel = BEV.shape  # 例如 360, 640
                        q_img = QImage(BEV.data, width, height, channel * width, QImage.Format_RGB888)
                        # q_img = QImage(bytes(BEV.data), width, height, channel * width, QImage.Format_RGB888)
                        pixmap = QPixmap.fromImage(q_img)
                        scaled_pixmap_label_BEV = pixmap.scaled(self.label_BEV.width(), self.label_BEV.height(),
                                                                Qt.KeepAspectRatio)

                        if collision_probability:
                            cp = max(collision_probability.values())
                            if cp < 0.1:
                                color = '#8DBF8B'
                            elif cp < 0.3:
                                color = '#FDB15D'
                            elif cp < 0.5:
                                color = '#F95A37'
                            else:
                                color = '#E04255'
                            self.label_collision_probability.setStyleSheet(f"background-color: {color};")

                        self.signal_update_label_object_detection.emit(scaled_pixmap_label_object_detection)
                        self.signal_update_label_semantic_segmentation.emit(scaled_pixmap_label_semantic_segmentation)
                        self.signal_update_label_depth.emit(scaled_pixmap_label_depth)
                        self.signal_update_label_BEV.emit(scaled_pixmap_label_BEV)
                        self.signal_update_label_collision_probability_data.emit(
                            scaled_pixmap_label_collision_probability_data)
                        QApplication.processEvents()

                except KeyboardInterrupt:
                    print("Client stopped.")
                finally:
                    # 释放摄像头并关闭窗口
                    self.client_socket.close()

                self.signal_clean_label_collision_probability.emit()
                self.signal_clean_label_object_detection.emit()
                self.signal_clean_label_semantic_segmentation.emit()
                self.signal_clean_label_depth.emit()
                self.signal_clean_label_BEV.emit()
                self.signal_clean_label_collision_probability_data.emit()
                self.signal_clean_textEdit_terminal.emit()
                QApplication.processEvents()

            # 启动捕获和发送任务
            asyncio.run(capture_and_send())

    def clicked_btn_play_3D_audio(self):
        if not self.clicked_btn_play_3D_audio_is_clicked and self.clicked_btn_run_is_clicked:
            self.clicked_btn_play_3D_audio_is_clicked = True
            threading.Thread(target=self.thread_play_3D_audio).start()

    def clicked_btn_stop(self):
        self.clicked_btn_run_is_clicked = False
        self.clicked_btn_play_3D_audio_is_clicked = False
        self.btn_stop.setStyleSheet("QPushButton {background-color: #E04255;}")
        QApplication.processEvents()
        time.sleep(1)
        self.btn_stop.setStyleSheet("QPushButton {}")


    def closeEvent(self, event):
        self.threads_running = False
        self.windowClosed.emit()
        if self.ServerApp is not None:
            self.ServerApp.close()
        event.accept()

class Window(QMainWindow):
    signal_update_label_L_MONO = pyqtSignal(QPixmap)
    signal_clean_label_L_MONO = pyqtSignal()
    signal_update_label_R_MONO = pyqtSignal(QPixmap)
    signal_clean_label_R_MONO = pyqtSignal()
    signal_update_label_RGB = pyqtSignal(QPixmap)
    signal_clean_label_RGB = pyqtSignal()
    signal_update_label_D = pyqtSignal(QPixmap)
    signal_clean_label_D = pyqtSignal()
    signal_update_label_pose = pyqtSignal(QPixmap)
    signal_clean_label_pose = pyqtSignal()
    signal_update_textEdit_pose = pyqtSignal(str)
    signal_clean_textEdit_pose = pyqtSignal()
    signal_update_label_Perception = pyqtSignal(QPixmap)
    signal_update_label_BEV = pyqtSignal(QPixmap)
    signal_clean_label_Perception_BEV = pyqtSignal()
    signal_update_lineEdit_navigation_destination = pyqtSignal(str)
    signal_back_lineEdit_navigation_destination = pyqtSignal()
    signal_update_textEdit_terminal = pyqtSignal(str)

    def __init__(self):
        logging.info("Starting...")
        self.threads_running = True
        self.system_pause = False
        self.opened_other_widge = False
        self.use_cloud_computing = False
        super().__init__()
        self.initUI()
        self.initFunc()
        self.initDevice()
        self.initdata()
        logging.info("Start Running...")

    def initUI(self):
        self.setFixedSize(1600, 900)
        self.setWindowTitle("[后台界面] 基于视觉环境感知的视障人士出行导航系统 V1.0")
        # self.current_address = [116.349594, 40.042233]
        self.current_address = [117.204424,31.769715]

        self.label_L_MONO = QLabel(self)
        self.label_L_MONO.setGeometry(10, 10, 320, 200)  # x, y, width, height
        self.label_L_MONO.setText("Cam: RectifiedLeft")
        self.label_L_MONO.setStyleSheet("background-color: #FAF5E4;")

        self.label_R_MONO = QLabel(self)
        self.label_R_MONO.setGeometry(10, 220, 320, 200)
        self.label_R_MONO.setText("Cam: RectifiedRight")
        self.label_R_MONO.setStyleSheet("background-color: #FAF5E4;")

        self.label_RGB = QLabel(self)
        self.label_RGB.setGeometry(10, 430, 320, 180)
        self.label_RGB.setText("Cam: Color")
        self.label_RGB.setStyleSheet("background-color: #FAF5E4;")

        self.label_D = QLabel(self)
        self.label_D.setGeometry(10, 620, 320, 180)
        self.label_D.setText("Cam: Depth")
        self.label_D.setStyleSheet("background-color: #FAF5E4;")

        self.label_pose = QLabel(self)
        self.label_pose.setGeometry(10, 810, 160, 80)
        self.label_pose.setText("Camera Pose")
        self.label_pose.setStyleSheet("background-color: #FAF5E4;")

        self.textEdit_pose = QTextEdit(self)
        self.textEdit_pose.setGeometry(180, 810, 80, 80)
        self.textEdit_pose.setReadOnly(True)  # 设置文本框为只读

        self.btn_open_cam = QPushButton("相机", self)
        self.btn_open_cam.setGeometry(270, 810, 60, 24)
        self.btn_open_depth = QPushButton("深度图", self)
        self.btn_open_depth.setGeometry(270, 838, 60, 24)
        self.btn_open_pose = QPushButton("角度", self)
        self.btn_open_pose.setGeometry(270, 867, 60, 24)

        self.label_Perception = QLabel(self)
        self.label_Perception.setGeometry(350, 10, 550, 320)
        self.label_Perception.setText("Visual Environmental Perception")
        self.label_Perception.setStyleSheet("background-color: #FAF5E4;")

        self.label_BEV = QLabel(self)
        self.label_BEV.setGeometry(350, 340, 550, 550)
        self.label_BEV.setText("BEV (Bird's-eye view)")
        self.label_BEV.setStyleSheet("background-color: #FAF5E4;")

        self.browser_amap = QWebEngineView(self)
        self.browser_amap.setGeometry(910, 10, 680, 320)
        self.draw_current_amap()

        self.textEdit_navigation_route = QTextEdit(self)
        self.textEdit_navigation_route.setGeometry(910, 340, 680, 120)
        self.textEdit_navigation_route.setReadOnly(True)  # 设置文本框为只读

        self.lineEdit_navigation_destination = QLineEdit(self)
        self.lineEdit_navigation_destination.setGeometry(910, 470, 480, 30)
        self.lineEdit_navigation_destination.setPlaceholderText("输入目的地")
        self.btn_speech_input = QPushButton("语音输入", self)
        self.btn_speech_input.setGeometry(1400, 470, 90, 30)
        self.btn_start_navigation = QPushButton("导航", self)
        self.btn_start_navigation.setGeometry(1500, 470, 90, 30)

        self.label_logo_Chinese = QLabel(self)
        self.label_logo_Chinese.setText('基于环境感知的视障人士出行导航系统 V1.0')
        self.label_logo_Chinese.setGeometry(910, 510, 680, 40)
        self.label_logo_Chinese.setFont(QFont('黑体', 20))
        self.label_logo_English = QLabel(self)
        self.label_logo_English.setText(
            'Navigation System for Visually Impaired People Based on Visual Environmental Perception V1.0')
        self.label_logo_English.setGeometry(910, 550, 680, 30)
        self.label_logo_English.setFont(QFont('Times New Roman', 10))

        # self.textEdit_collision = QTextEdit(self)
        # self.textEdit_collision.setGeometry(910, 590, 680, 30)
        # self.textEdit_collision.setReadOnly(True)  # 设置文本框为只读

        self.textEdit_terminal = QTextEdit(self)
        self.textEdit_terminal.setGeometry(910, 590, 680, 240)
        self.textEdit_terminal.setReadOnly(True)  # 设置文本框为只读

        self.btn_switch_BEV = QPushButton("显示轨迹", self)
        self.btn_switch_BEV.setGeometry(910, 840, 100, 50)
        self.btn_open_cloud_computing = QPushButton("云计算", self)
        self.btn_open_cloud_computing.setGeometry(1020, 840, 100, 50)
        self.btn_system_run = QPushButton("RUN", self)
        self.btn_system_run.setGeometry(1380, 840, 100, 50)
        self.btn_system_stop = QPushButton("STOP", self)
        self.btn_system_stop.setGeometry(1490, 840, 100, 50)

    def initFunc(self):
        self.signal_update_label_L_MONO.connect(self.label_L_MONO.setPixmap)
        self.signal_clean_label_L_MONO.connect(lambda: self.label_L_MONO.setText("Cam: RectifiedLeft"))
        self.signal_clean_label_L_MONO.connect(lambda: self.label_L_MONO.setStyleSheet("background-color: #FAF5E4;"))

        self.signal_update_label_R_MONO.connect(self.label_R_MONO.setPixmap)
        self.signal_clean_label_R_MONO.connect(lambda: self.label_R_MONO.setText("Cam: RectifiedRight"))
        self.signal_clean_label_R_MONO.connect(lambda: self.label_R_MONO.setStyleSheet("background-color: #FAF5E4;"))

        self.signal_update_label_RGB.connect(self.label_RGB.setPixmap)
        self.signal_clean_label_RGB.connect(lambda: self.label_RGB.setText("Cam: Color"))
        self.signal_clean_label_RGB.connect(lambda: self.label_RGB.setStyleSheet("background-color: #FAF5E4;"))

        self.signal_update_label_D.connect(self.label_D.setPixmap)
        self.signal_clean_label_D.connect(lambda: self.label_D.setText("Cam: RectifiedLeft"))
        self.signal_clean_label_D.connect(lambda: self.label_D.setStyleSheet("background-color: #FAF5E4;"))

        self.signal_update_label_pose.connect(self.label_pose.setPixmap)
        self.signal_clean_label_pose.connect(lambda: self.label_pose.setText("Camera Pose"))
        self.signal_clean_label_pose.connect(lambda: self.label_pose.setStyleSheet("background-color: #FAF5E4;"))

        self.signal_update_textEdit_pose.connect(self.textEdit_pose.setText)
        self.signal_update_textEdit_pose.connect(lambda: self.textEdit_pose.setFont(QFont('Times New Roman', 8)))
        self.signal_clean_textEdit_pose.connect(self.textEdit_pose.clear)

        self.btn_open_cam.clicked.connect(self.clicked_btn_open_cams)
        self.btn_open_depth.clicked.connect(self.clicked_btn_open_depth)
        self.btn_open_pose.clicked.connect(self.clicked_btn_open_pose)

        self.signal_update_label_Perception.connect(self.label_Perception.setPixmap)
        self.signal_update_label_BEV.connect(self.label_BEV.setPixmap)
        self.signal_clean_label_Perception_BEV.connect(lambda: self.label_Perception.setText("Visual Environmental Perception"))
        self.signal_clean_label_Perception_BEV.connect(lambda: self.label_Perception.setStyleSheet("background-color: #FAF5E4;"))
        self.signal_clean_label_Perception_BEV.connect(lambda: self.label_BEV.setText("BEV (Bird's-eye view)"))
        self.signal_clean_label_Perception_BEV.connect(lambda: self.label_BEV.setStyleSheet("background-color: #FAF5E4;"))

        self.signal_update_lineEdit_navigation_destination.connect(
            lambda text: self.lineEdit_navigation_destination.setText(f"{text}"))
        self.signal_back_lineEdit_navigation_destination.connect(
            lambda: self.lineEdit_navigation_destination.setPlaceholderText("输入目的地"))
        self.signal_back_lineEdit_navigation_destination.connect(
            lambda: self.lineEdit_navigation_destination.clear())

        self.btn_speech_input.pressed.connect(self.pressed_btn_speech_input)
        self.btn_speech_input.released.connect(self.released_btn_speech_input)
        self.btn_start_navigation.clicked.connect(self.clicked_btn_start_navigation)

        self.signal_update_textEdit_terminal.connect(
            lambda text: self.textEdit_terminal.append(f"{text}"))

        self.btn_switch_BEV.clicked.connect(self.clicked_btn_switch_BEV)
        self.btn_open_cloud_computing.clicked.connect(self.clicked_btn_open_cloud_computing)
        self.btn_system_run.clicked.connect(self.clicked_btn_system_run)
        self.btn_system_stop.clicked.connect(self.clicked_btn_system_stop)

    def initdata(self):
        self.btn_open_cam_is_clicked = False
        self.btn_open_depth_is_clicked = False
        self.btn_open_pose_is_clicked = False
        self.btn_speech_input_is_clicked = False
        self.btn_system_run_is_clicked = False
        self.btn_system_stop_is_clicked = False

        self.frames = []
        self.thread_record_running = False
        self.thread_recognize_running = False

        self.msgGrp = None
        self.img_rectifiedLeft = None
        self.img_rectifiedRight = None
        self.img_BGR = None
        self.img_D = None
        self.imu = None
        self.Quaternion = None

    def initDevice(self):
        self.queue = oak_device.getOutputQueue("xoutGrp", 10, False)
        threading.Thread(target=self.thread_refresh_msgGrp).start()

    def thread_refresh_msgGrp(self):
        while True:
            if not self.threads_running or self.opened_other_widge:
                break
            self.msgGrp = self.queue.get()

    def clicked_btn_open_cams(self):
        if not self.btn_system_run_is_clicked:
            if self.btn_open_cam_is_clicked:
                self.btn_open_cam_is_clicked = False
                self.btn_open_cam.setStyleSheet("QPushButton {}")
            else:
                self.btn_open_cam_is_clicked = True
                self.btn_open_cam.setStyleSheet("QPushButton {background-color: #8DBF8B;}")
                QApplication.processEvents()

                threading.Thread(target=self.thread_update_frame_cams).start()

    def clicked_btn_open_depth(self):
        if not self.btn_system_run_is_clicked:
            if self.btn_open_depth_is_clicked:
                self.btn_open_depth_is_clicked = False
                self.btn_open_depth.setStyleSheet("QPushButton {}")
            else:
                self.btn_open_depth_is_clicked = True
                self.btn_open_depth.setStyleSheet("QPushButton {background-color: #8DBF8B;}")
                QApplication.processEvents()

                threading.Thread(target=self.thread_update_frame_depth).start()

    def clicked_btn_open_pose(self):
        if not self.btn_system_run_is_clicked:
            if self.btn_open_pose_is_clicked:
                self.btn_open_pose_is_clicked = False
                self.btn_open_pose.setStyleSheet("QPushButton {}")
            else:
                self.btn_open_pose_is_clicked = True
                self.btn_open_pose.setStyleSheet("QPushButton {background-color: #8DBF8B;}")
                QApplication.processEvents()

                threading.Thread(target=self.thread_update_frame_pose).start()

    def draw_current_amap(self):
        map_html = f"""
            <!doctype html>
            <html>
            <head>
                <meta charset="utf-8">
                <meta http-equiv="X-UA-Compatible" content="IE=edge">
                <meta name="viewport" content="initial-scale=1.0, user-scalable=no, width=device-width">
                <link rel="stylesheet" href="https://a.amap.com/jsapi_demos/static/demo-center/css/demo-center.css" />
                <title>地图显示</title>
                <style>
                    html,
                    body,
                    #container {{
                      width: 100%;
                      height: 100%;
                    }}
                </style>
            </head>
            <body>
            <div id="container"></div>
            <!-- 加载地图JSAPI脚本 -->
            <script type="text/javascript"> window._AMapSecurityConfig = {{securityJsCode: "aaa4fcbdf20f0034b736d8a507bb9cce",}}; </script>
            <script src="https://webapi.amap.com/maps?v=2.0&key=6a7bb6c35b9a1c699b765d3b415e5e08"></script>
            <script>
                var map = new AMap.Map('container', {{
                    viewMode: '2D', // 默认使用 2D 模式，如果希望使用带有俯仰角的 3D 模式，请设置 viewMode: '3D'
                    zoom:15, // 初始化地图层级
                    center: {self.current_address} // 初始化地图中心点
                }});
                var marker = new AMap.Marker({{
                    position: {self.current_address}, // 标记位置，这里使用的是天安门的坐标
                    map: map
                }});
                AMap.plugin('AMap.Scale',function(){{
                    var tool = new AMap.Scale(); 
                    map.addControl(tool);
                }});
                AMap.plugin('AMap.ControlBar',function(){{
                    var tool = new AMap.ControlBar(); 
                    map.addControl(tool);
                }});
            </script>
            </body>
            </html>
            """
        self.browser_amap.setHtml(map_html, baseUrl=QUrl("https://webapi.amap.com"))

    def pressed_btn_speech_input(self):
        if not self.btn_speech_input_is_clicked:
            self.btn_speech_input_is_clicked = True
            self.signal_update_lineEdit_navigation_destination.emit("正在录音...")
            threading.Thread(target=self.thread_record).start()

    def released_btn_speech_input(self):
        self.btn_speech_input_is_clicked = False
        while self.thread_record_running:
            pass
        self.signal_update_lineEdit_navigation_destination.emit('正在识别...')
        while self.thread_recognize_running:
            pass

    def clicked_btn_start_navigation(self):
        def get_navigation_data(current_address, destination, key="ac5e6845a081b25303b11702c3196f50"):
            # API URL
            url = f"https://restapi.amap.com/v5/direction/walking"
            # 请求参数
            params = {
                "key": key,
                "isindoor": 1,
                "origin": current_address,
                "destination": destination
            }
            # 发送GET请求
            response = requests.get(url, params=params)
            # 检查请求是否成功
            if response.status_code == 200:
                # 打印JSON数据
                return response.json()
            else:
                return f"Error: {response.status_code}"

        def get_geocode_data(address, key="ac5e6845a081b25303b11702c3196f50"):
            # API URL
            url = f"https://restapi.amap.com/v3/geocode/geo"
            # 请求参数
            params = {
                "key": key,
                "address": address,
                "output": "JSON",
            }
            # 发送GET请求
            response = requests.get(url, params=params)
            # 检查请求是否成功
            if response.status_code == 200:
                # 打印JSON数据
                return response.json()
            else:
                return f"Error: {response.status_code}"

        self.textEdit_navigation_route.clear()
        current_address = str(self.current_address[0]) + ',' + str(self.current_address[1])
        # 将用户输入解析为起点和终点
        navigation_destination = self.lineEdit_navigation_destination.text()
        map_html = f"""
        <!doctype html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta http-equiv="X-UA-Compatible" content="IE=edge">
            <meta name="viewport" content="initial-scale=1.0, user-scalable=no, width=device-width">
            <title>地点关键字 + 步行路线规划</title>
            <style type="text/css">
                html,
                body,
                #container {{
                    width: 100%;
                    height: 100%;
                }}
                #panel {{
                    position: fixed;
                    background-color: white;
                    max-height: 90%;
                    overflow-y: auto;
                    top: 10px;
                    right: 10px;
                    width: 280px;
                }}
                #panel .amap-call {{
                    background-color: #009cf9;
                    border-top-left-radius: 4px;
                    border-top-right-radius: 4px;
                }}
                #panel .amap-lib-walking {{
                    border-bottom-left-radius: 4px;
                    border-bottom-right-radius: 4px;
                    overflow: hidden;
                }}
            </style>
            <link rel="stylesheet" href="https://a.amap.com/jsapi_demos/static/demo-center/css/demo-center.css" />
            <script type="text/javascript"> window._AMapSecurityConfig = {{securityJsCode: "aaa4fcbdf20f0034b736d8a507bb9cce",}}; </script>
            <script type="text/javascript" src="https://webapi.amap.com/maps?v=2.0&key=6a7bb6c35b9a1c699b765d3b415e5e08&plugin=AMap.Geocoder,AMap.Walking,AMap.Scale,AMap.ControlBar"></script>
            <script src="https://a.amap.com/jsapi_demos/static/demo-center/js/demoutils.js"></script>
            <script type="text/javascript" src="https://cache.amap.com/lbs/static/addToolbar.js"></script>
        </head>

        <body>
        <div id="container"></div>
        <div id="panel"></div>
            <script type="text/javascript">
                var map = new AMap.Map("container", {{
                    resizeEnable: true,
                    mapStyle: "amap://styles/normal", //设置地图的显示样式
                    center: [116.397428, 39.90923],//地图中心点
                    zoom: 13 //地图显示的缩放级别
                }});

                AMap.plugin('AMap.Scale',function(){{
                    var tool = new AMap.Scale(); 
                    map.addControl(tool);
                }});
                AMap.plugin('AMap.ControlBar',function(){{
                    var tool = new AMap.ControlBar(); 
                    map.addControl(tool);
                }});

                var geocoder = new AMap.Geocoder({{
                    city: "合肥", //城市设为北京，默认：“全国”
                }});
                geocoder.getLocation('{navigation_destination}',
                    function(status, result) {{
                        if (status === 'complete' && result.geocodes.length) {{
                            var lnglat = result.geocodes[0].location;

                            // 步行导航查询放在这里
                            var walking = new AMap.Walking();
                            //根据起终点坐标规划步行路线
                            walking.search(
                                {self.current_address},
                                [lnglat.lng, lnglat.lat],
                                function(status, result) {{
                                    if (status === 'complete') {{
                                        if (result.routes && result.routes.length) {{
                                            drawRoute(result.routes[0]);
                                            log.success('绘制步行路线完成');
                                            // 触发Python端的信号
                                            new QWebChannel(qt.webChannelTransport, function (channel) {{
                                                var myObj = channel.objects.myObj;
                                                myObj.resultReady.emit(result);
                                            }});
                                        }}
                                    }} else {{
                                        log.error('步行路线数据查询失败' + result);
                                    }}
                                }}
                            );
                        }} else {{
                            log.error('根据地址查询位置失败');
                        }}
                    }}
                );

                function drawRoute (route) {{
                    var path = parseRouteToPath(route)

                    var startMarker = new AMap.Marker({{
                        position: path[0],
                        icon: 'https://webapi.amap.com/theme/v2.0/markers/n/start.png',
                        map: map,
                        anchor: 'bottom-center',
                    }})

                    var endMarker = new AMap.Marker({{
                        position: path[path.length - 1],
                        icon: 'https://webapi.amap.com/theme/v2.0/markers/n/end.png',
                        map: map,
                        anchor: 'bottom-center',
                    }})

                    var routeLine = new AMap.Polyline({{
                        path: path,
                        isOutline: true,
                        outlineColor: '#ffeeee',
                        borderWeight: 2,
                        strokeWeight: 5,
                        strokeColor: '#0091ff',
                        strokeOpacity: 0.9,
                        lineJoin: 'round'
                    }})

                    map.add(routeLine);

                    // 调整视野达到最佳显示区域
                    map.setFitView([ startMarker, endMarker, routeLine ])
                }}

                function parseRouteToPath(route) {{
                    var path = []

                    for (var i = 0, l = route.steps.length; i < l; i++) {{
                        var step = route.steps[i]

                        for (var j = 0, n = step.path.length; j < n; j++) {{
                          path.push(step.path[j])
                        }}
                    }}

                    return path
                }}
            </script>
        </body>
        </html>
        """
        self.browser_amap.setHtml(map_html, baseUrl=QUrl("https://webapi.amap.com"))

        destination_data = get_geocode_data(address=navigation_destination)
        if destination_data['status'] == '1':
            geocodes = destination_data['geocodes'][0]
            location = geocodes['location']
            # 获取数据
            data = get_navigation_data(current_address, location)
            if data['status'] == '1':
                route = data['route']['paths'][0]  # 取第一条路径
                distance = route['distance']
                cost = route['cost']['duration']
                steps = route['steps']

                self.textEdit_navigation_route.append("距离: " + str(distance) + " 米")
                # self.resultText.append("成本: " + str(cost) + " 分钟")
                seconds = int(cost)  # 假设cost是以秒为单位的时间
                if seconds >= 3600:  # 如果超过或等于一小时
                    hours = seconds // 3600
                    remaining_seconds = seconds % 3600
                    minutes = remaining_seconds // 60
                    remaining_seconds = remaining_seconds % 60
                    self.textEdit_navigation_route.append(
                        "成本: {}小时{}分钟{}秒".format(hours, minutes, remaining_seconds))
                elif seconds >= 60:  # 如果超过或等于一分钟但少于一小时
                    minutes = seconds // 60
                    remaining_seconds = seconds % 60
                    self.textEdit_navigation_route.append("成本: {}分钟{}秒".format(minutes, remaining_seconds))
                else:  # 如果少于一分钟
                    self.textEdit_navigation_route.append("成本: {}秒".format(seconds))

                for s, step in enumerate(steps):
                    self.textEdit_navigation_route.append(f"步骤{s}: " + step['instruction'])  # 假设步骤信息在instruction键中
            else:
                self.textEdit_navigation_route.append(data["info"])
        else:
            self.textEdit_navigation_route.append(destination_data["info"])

    def clicked_btn_switch_BEV(self):
        self.opened_other_widge = True
        self.clicked_btn_system_stop()

        self.secondWindow = SecondWindow()
        self.secondWindow.windowClosed.connect(self.on_second_window_closed)
        self.secondWindow.setAttribute(Qt.WA_DeleteOnClose)  # 确保窗口关闭时释放资源
        self.secondWindow.setWindowModality(Qt.ApplicationModal)  # 设置窗口为应用程序级别的模态
        self.secondWindow.show()

    def clicked_btn_open_cloud_computing(self):
        self.use_cloud_computing = True
        self.opened_other_widge = True
        self.clicked_btn_system_stop()

        self.cloud_service_window = cloud_service_window()
        self.cloud_service_window.windowClosed.connect(self.on_cloud_service_window_closed)
        self.cloud_service_window.setAttribute(Qt.WA_DeleteOnClose)  # 确保窗口关闭时释放资源
        self.cloud_service_window.setWindowModality(Qt.ApplicationModal)  # 设置窗口为应用程序级别的模态
        self.cloud_service_window.show()

    def on_cloud_service_window_closed(self):
        self.use_cloud_computing = False
        self.opened_other_widge = False
        # 在此处添加任何需要在子窗口关闭时执行的其他操作

    def on_second_window_closed(self):
        self.opened_other_widge = False
        self.initDevice()

    def clicked_btn_system_run(self):
        if self.btn_system_run_is_clicked:
            pass
        else:
            self.btn_system_stop_is_clicked = False
            self.system_pause = False
            self.btn_system_run.setStyleSheet("QPushButton {background-color: #8DBF8B;}")

            self.btn_open_cam_is_clicked = False
            self.btn_open_depth_is_clicked = False
            self.btn_open_pose_is_clicked = False

            self.clicked_btn_open_cams()
            self.clicked_btn_open_depth()
            self.clicked_btn_open_pose()
            QApplication.processEvents()

            threading.Thread(target=self.thread_update_frame_Perception_BEV).start()
            self.btn_system_run_is_clicked = True

    def clicked_btn_system_stop(self):
            self.btn_system_stop_is_clicked = True
            self.system_pause = True
            self.btn_system_stop.setStyleSheet("QPushButton {background-color: #E04255;}")
            self.btn_open_cam.setStyleSheet("QPushButton {}")
            self.btn_open_depth.setStyleSheet("QPushButton {}")
            self.btn_open_pose.setStyleSheet("QPushButton {}")
            self.btn_system_run.setStyleSheet("QPushButton {}")
            self.btn_system_stop.setText("stopping")
            QApplication.processEvents()
            time.sleep(1)
            self.btn_system_stop.setStyleSheet("QPushButton {}")
            self.system_pause = False
            self.btn_system_stop_is_clicked = False
            self.btn_system_run_is_clicked = False
            self.btn_open_cam_is_clicked = False
            self.btn_open_depth_is_clicked = False
            self.btn_open_pose_is_clicked = False
            self.btn_system_stop.setText("STOP")

    def thread_update_frame_cams(self):
        while self.threads_running:
            if not self.btn_open_cam_is_clicked or self.system_pause:
                # self.label_L_MONO.setText("Cam: RectifiedLeft")
                # self.label_L_MONO.setStyleSheet("background-color: #FAF5E4;")
                # self.label_R_MONO.setText("Cam: RectifiedRight")
                # self.label_R_MONO.setStyleSheet("background-color: #FAF5E4;")
                # self.label_RGB.setText("Cam: Color")
                # self.label_RGB.setStyleSheet("background-color: #FAF5E4;")
                self.signal_clean_label_L_MONO.emit()
                self.signal_clean_label_R_MONO.emit()
                self.signal_clean_label_RGB.emit()
                break

            self.img_rectifiedLeft = self.msgGrp['rectifiedLeft'].getCvFrame()
            self.img_rectifiedRight = self.msgGrp['rectifiedRight'].getCvFrame()
            self.img_BGR = self.msgGrp['color'].getCvFrame()

            # 左侧图像处理
            height, width = self.img_rectifiedLeft.shape  # 例如 400, 640
            q_img = QImage(self.img_rectifiedLeft.data, width, height, width, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(q_img)
            scaled_pixmap_label_L_MONO = pixmap.scaled(self.label_L_MONO.width(), self.label_L_MONO.height(),
                                                       Qt.KeepAspectRatio)
            # 右侧图像处理
            height, width = self.img_rectifiedRight.shape  # 例如 400, 640
            q_img = QImage(self.img_rectifiedRight.data, width, height, width, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(q_img)
            scaled_pixmap_label_R_MONO = pixmap.scaled(self.label_R_MONO.width(), self.label_R_MONO.height(),
                                                       Qt.KeepAspectRatio)
            # RGB 图像处理
            img_RGB = cv2.cvtColor(self.img_BGR, cv2.COLOR_BGR2RGB)
            height, width, channel = img_RGB.shape  # 例如 360, 640
            q_img = QImage(img_RGB.data, width, height, channel * width, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            scaled_pixmap_label_RGB = pixmap.scaled(self.label_RGB.width(), self.label_RGB.height(), Qt.KeepAspectRatio)

            # self.label_L_MONO.setPixmap(scaled_pixmap_label_L_MONO)
            # self.label_R_MONO.setPixmap(scaled_pixmap_label_R_MONO)
            # self.label_RGB.setPixmap(scaled_pixmap_label_RGB)

            self.signal_update_label_L_MONO.emit(scaled_pixmap_label_L_MONO)
            self.signal_update_label_R_MONO.emit(scaled_pixmap_label_R_MONO)
            self.signal_update_label_RGB.emit(scaled_pixmap_label_RGB)

            # cv2.waitKey(1)

    def thread_update_frame_depth(self):
        while self.threads_running:
            if self.btn_open_depth_is_clicked == False or self.system_pause:
                # self.label_D.setText("Cam: Depth")
                # self.label_D.setStyleSheet("background-color: #FAF5E4;")
                self.signal_clean_label_D.emit()
                break
            self.img_D = self.msgGrp['depth'].getCvFrame()

            # 深度图处理
            img_depth = self.img_D.copy()
            img_depth[img_depth == 0] = 10000
            depth255 = (img_depth * (255.0 / 10000)).astype(np.uint8)
            depthCOLORMAP = cv2.applyColorMap((255 - depth255), cv2.COLORMAP_MAGMA)
            depthCOLORMAP = cv2.cvtColor(depthCOLORMAP, cv2.COLOR_BGR2RGB)
            height, width, channel = depthCOLORMAP.shape  # 例如 360, 640
            q_img = QImage(depthCOLORMAP.data, width, height, channel * width, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            scaled_pixmap = pixmap.scaled(self.label_D.width(), self.label_D.height(), Qt.KeepAspectRatio)

            # self.label_D.setPixmap(scaled_pixmap)
            self.signal_update_label_D.emit(scaled_pixmap)

            # cv2.waitKey(1)

    def thread_update_frame_pose(self):
        while self.threads_running:
            if self.btn_open_pose_is_clicked == False or self.system_pause:
                # self.label_pose.setText("Camera Pose")
                # self.label_pose.setStyleSheet("background-color: #FAF5E4;")
                self.signal_clean_label_pose.emit()
                self.signal_clean_textEdit_pose.emit()
                break

            background = np.zeros((80, 160, 3), np.uint8)
            center = (80, 40)
            self.imu = self.msgGrp['imu'].packets[0]
            self.Quaternion = Quaternion(self.imu.rotationVector.real, self.imu.rotationVector.i, self.imu.rotationVector.j,
                                         self.imu.rotationVector.k)

            down_left = self.Quaternion.rotate(
                np.array([30 * math.tan(math.radians(vfov / 2)), 30 * math.tan(math.radians(hfov / 2)), 30]))
            up_left = self.Quaternion.rotate(
                np.array([-30 * math.tan(math.radians(vfov / 2)), 30 * math.tan(math.radians(hfov / 2)), 30]))
            down_right = self.Quaternion.rotate(
                np.array([30 * math.tan(math.radians(vfov / 2)), -30 * math.tan(math.radians(hfov / 2)), 30]))
            up_right = self.Quaternion.rotate(
                np.array([-30 * math.tan(math.radians(vfov / 2)), -30 * math.tan(math.radians(hfov / 2)), 30]))

            cv2.line(
                background,
                center,
                (int(down_left[0] + 0.5) + center[0], -int(down_left[1] + 0.5) + center[1]),
                (100, 100, 100),
                1
            )  # 左下角
            cv2.line(
                background,
                center,
                (int(down_right[0] + 0.5) + center[0], -int(down_right[1] + 0.5) + center[1]),
                (100, 100, 100),
                1
            )  # 右下角
            cv2.line(
                background,
                center,
                (int(up_left[0] + 0.5) + center[0], -int(up_left[1] + 0.5) + center[1]),
                (150, 150, 150),
                1
            )  # 左上角
            cv2.line(
                background,
                center,
                (int(up_right[0] + 0.5) + center[0], -int(up_right[1] + 0.5) + center[1]),
                (150, 150, 150),
                1
            )  # 右上角

            cv2.line(
                background,
                (int(down_left[0] + 0.5) + center[0], -int(down_left[1] + 0.5) + center[1]),
                (int(down_right[0] + 0.5) + center[0], -int(down_right[1] + 0.5) + center[1]),
                (8, 75, 30),
                1
            )  # 下边沿
            cv2.line(
                background,
                (int(up_left[0] + 0.5) + center[0], -int(up_left[1] + 0.5) + center[1]),
                (int(down_left[0] + 0.5) + center[0], -int(down_left[1] + 0.5) + center[1]),
                (150, 150, 150),
                1
            )  # 左边沿
            cv2.line(
                background,
                (int(up_right[0] + 0.5) + center[0], -int(up_right[1] + 0.5) + center[1]),
                (int(down_right[0] + 0.5) + center[0], -int(down_right[1] + 0.5) + center[1]),
                (150, 150, 150),
                1
            )  # 右边沿
            cv2.line(
                background,
                (int(up_left[0] + 0.5) + center[0], -int(up_left[1] + 0.5) + center[1]),
                (int(up_right[0] + 0.5) + center[0], -int(up_right[1] + 0.5) + center[1]),
                (79, 160, 39),
                2
            )  # 上边沿

            height, width, channel = background.shape  # 例如 360, 640
            q_img = QImage(background.data, width, height, channel * width, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            scaled_pixmap = pixmap.scaled(self.label_pose.width(), self.label_pose.height(), Qt.KeepAspectRatio)
            # self.label_pose.setPixmap(scaled_pixmap)
            self.signal_update_label_pose.emit(scaled_pixmap)

            pose_data = f"real: {self.imu.rotationVector.real:.4f}\ni   : {self.imu.rotationVector.i:.4f}\nj   : {self.imu.rotationVector.j:.4f}\nk   : {self.imu.rotationVector.k:.4f}"
            self.signal_update_textEdit_pose.emit(pose_data)

            # cv2.waitKey(1)

    def thread_update_frame_Perception_BEV(self):
        def generate_random_color(seed=None):
            """生成并返回一个随机颜色"""
            random.seed(seed)  # 设置随机数种子
            red = random.randint(0, 255)
            green = random.randint(0, 255)
            blue = random.randint(0, 255)
            return (red, green, blue)

        self.road_model = RoadBoundGetter(scale=0.3, density=10, pretrain="utils/road_model_maxmIOU75.pth")
        self.YOLO_model = YOLO('utils/yolov8l-seg.pt')

        while self.threads_running:
            if self.system_pause:
                self.signal_clean_label_Perception_BEV.emit()
                break
            BEV = BEV_Background()
            perception_background = self.img_BGR.copy()
            depth = self.img_D.copy()

            # 人行道识别及处理
            # road_masked = self.road_model(self.img_BGR)[0].cpu()
            # perception_background[road_masked == 1] = (233, 233, 233)
            # rows, cols = np.where((road_masked == 1) & (self.img_D != 0))
            # if len(rows) > 0 and len(cols) > 0:
            #     distances = self.img_D[rows, cols]
            #     # # 计算百分位数
            #     # percentile_low = 5
            #     # percentile_high = 95
            #     # low_threshold = np.percentile(distances, percentile_low)
            #     # high_threshold = np.percentile(distances, percentile_high)
            #     # # 选择符合条件的数据
            #     # selected_rows = rows[(distances >= low_threshold) & (distances <= high_threshold)]
            #     # selected_cols = cols[(distances >= low_threshold) & (distances <= high_threshold)]
            #     # selected_distances = distances[(distances >= low_threshold) & (distances <= high_threshold)]
            #     bboxes = np.column_stack([cols, rows, cols, rows])
            #     BEV_x = np.clip((500 - getSpatialCoordinates(distances, bboxes, 'x') / 10).astype(int), 0, 999)
            #     BEV_y = np.clip((1000 - distances / 10).astype(int), 0, 999)
            #     BEV[BEV_y, BEV_x] = (233, 233, 233)

            # YOLO
            object_result = self.YOLO_model(self.img_BGR, verbose=False)[0]
            for i in range(len(object_result)):
                box = object_result.boxes[i]
                mask = object_result.masks[i].data[0][12: -12, :].int().cpu()
                bbox = box.xyxy.cpu().round().int().tolist()[0]
                name = object_result.names[box.cls.cpu().round().int().tolist()[0]]

                color = generate_random_color(int(box.cls.cpu()))
                perception_background[mask == 1] = color
                # road_masked[mask == 1] = 0
                cv2.rectangle(perception_background, bbox[:2], bbox[2:], color)

                rows, cols = np.where((mask == 1) & (depth != 0))
                if len(rows) > 0 and len(cols) > 0:
                    distances = depth[rows, cols]
                    # 计算百分位数
                    percentile_low = 10
                    percentile_high = 85
                    low_threshold = np.percentile(distances, percentile_low)
                    high_threshold = np.percentile(distances, percentile_high)
                    selected_rows = rows[(distances >= low_threshold) & (distances <= high_threshold)]
                    selected_cols = cols[(distances >= low_threshold) & (distances <= high_threshold)]
                    selected_distances = distances[(distances >= low_threshold) & (distances <= high_threshold)]
                    bboxes = np.column_stack([selected_cols, selected_rows, selected_cols, selected_rows])
                    BEV_x_filtered = np.clip(
                        (500 - getSpatialCoordinates(selected_distances, bboxes, 'x') / 10).astype(int), 0, 999)
                    BEV_y_filtered = np.clip((1000 - selected_distances / 10).astype(int), 0, 999)
                    BEV[BEV_y_filtered, BEV_x_filtered] = color

                    points = np.column_stack((BEV_x_filtered, BEV_y_filtered))
                    # hull = cv2.convexHull(points)
                    # cv2.fillConvexPoly(BEV, hull, color)
                    rect = cv2.minAreaRect(points)
                    box = cv2.boxPoints(rect)
                    box = np.intp(box)
                    cv2.drawContours(BEV, [box], 0, color, 2)

                    cv2.putText(perception_background, f"{name} {np.median(selected_distances):.1f}mm", (bbox[0], bbox[1] + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)

            perception_background = cv2.cvtColor(perception_background, cv2.COLOR_BGR2RGB)
            height, width, channel = perception_background.shape  # 例如 360, 640
            q_img = QImage(perception_background.data, width, height, channel * width, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            scaled_pixmap_label_Perception = pixmap.scaled(self.label_Perception.width(), self.label_Perception.height(),
                                          Qt.KeepAspectRatio)
            BEV = cv2.cvtColor(BEV, cv2.COLOR_BGR2RGB)
            height, width, channel = BEV.shape  # 例如 360, 640
            q_img = QImage(BEV.data, width, height, channel * width, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            scaled_pixmap_label_BEV = pixmap.scaled(self.label_BEV.width(), self.label_BEV.height(),
                                          Qt.KeepAspectRatio)
            # self.label_Perception.setPixmap(scaled_pixmap_label_Perception)
            # self.label_BEV.setPixmap(scaled_pixmap_label_BEV)
            self.signal_update_label_Perception.emit(scaled_pixmap_label_Perception)
            self.signal_update_label_BEV.emit(scaled_pixmap_label_BEV)

    def thread_record(self):
        self.thread_record_running = True
        # 录制音频参数
        chunk = 1024  # 每次读取的音频数据大小
        format = pyaudio.paInt16  # 音频格式
        channels = 1  # 单声道
        rate = 16000  # 采样率
        p = pyaudio.PyAudio()

        stream = p.open(format=format,
                        channels=channels,
                        rate=rate,
                        input=True,
                        frames_per_buffer=chunk)

        while self.btn_speech_input_is_clicked:
            data = stream.read(chunk)
            self.frames.append(data)
        stream.stop_stream()
        stream.close()
        p.terminate()
        self.thread_record_running = False
        threading.Thread(target=self.thread_recognize).start()
        self.thread_recognize_running = False

    def thread_recognize(self):
        self.thread_recognize_running = True
        recognizer = sr.Recognizer()
        # 将音频数据转换为 AudioData 对象
        audio_data = sr.AudioData(b''.join(self.frames), 16000, 2)
        try:
            # 使用Google Web Speech API进行中文语音识别
            text = recognizer.recognize_google(audio_data, language='zh-CN')
            print("识别结果: " + text)
            self.signal_update_lineEdit_navigation_destination.emit(text)
        except sr.UnknownValueError:
            self.signal_update_textEdit_terminal.emit("Google Web Speech API 无法识别音频")
            self.signal_back_lineEdit_navigation_destination.emit()
        except sr.RequestError as e:
            self.signal_update_textEdit_terminal.emit(f"无法连接到 Google Web Speech API 服务; {e}")
            self.signal_back_lineEdit_navigation_destination.emit()
        self.frames.clear()


    def closeEvent(self, event):
        self.threads_running = False  # 关闭窗口时将 running 设置为 False
        reply = QMessageBox.question(self, '确认', '你确定要退出程序吗？',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()  # 接受关闭事件，窗口将被关闭
            logging.info("System stopped...")
        else:
            event.ignore()  # 忽略关闭事件，窗口将不会关闭
            self.threads_running = True  # 保持线程运行


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())
