import sys
import os
import time

from pocketsphinx import AudioFile
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtWebEngineWidgets import *
from utils.OAK_Info import oak_device
from ultralytics import YOLO
import speech_recognition as sr
from scipy.spatial.transform import Rotation as R
import pyaudio

import threading
import numpy as np
import depthai as dai
import requests
import cv2
import math
import sys
import time

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtWebEngineWidgets import *
from utils.OAK_Info import oak_device
from utils.models import RoadBoundGetter
import sounddevice as sd
from utils.tools import *
import numpy as np
from ultralytics import YOLO
import librosa
import threading
import cv2
import logging
import random
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

objs = [labels_dict[i] for i in
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 56, 57, 58, 59, 60, 61, 62, 63, 67, 68, 72, 73]]


class Window(QMainWindow):
    signal_update_label = pyqtSignal(str)
    signal_update_map = pyqtSignal(str)
    signal_running = pyqtSignal()
    signal_pausing = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.threads_running = False
        self.current_address = [117.204424,31.769715]
        self.frames = []
        self.thread_record_running = False
        self.thread_recognize_running = False
        self.btn_clicked = False
        self.btn_run_is_clicked = False
        self.initUI()
        self.initdata()
        self.initDevice()

    def initUI(self):
        self.setWindowTitle("[用户界面] 基于视觉环境感知的视障人士出行导航系统 V1.0")
        self.setFixedSize(400, 800)  # 设置窗口大小

        self.lable_sign = QLabel(self)
        self.lable_sign.setGeometry(10, 10, 380, 20)
        self.lable_sign.setStyleSheet("background-color: #FAF5E4;")

        self.browser_amap = QWebEngineView(self)
        self.browser_amap.setGeometry(10, 40, 380, 350)
        self.draw_current_amap()

        self.lineEdit_terminal = QLineEdit(self)
        self.lineEdit_terminal.setGeometry(10, 400, 380, 50)
        self.signal_update_label.connect(lambda txt: self.lineEdit_terminal.setText(txt))
        self.signal_update_map.connect(lambda txt: self.update_map(txt))

        self.speak_btn = QPushButton("Speak", self)
        self.speak_btn.setGeometry(10, 460, 380, 160)
        self.run_btn = QPushButton("Run", self)
        self.run_btn.setGeometry(10, 630, 380, 160)

        self.speak_btn.pressed.connect(self.start_recording)
        self.speak_btn.released.connect(self.stop_recording)

        self.run_btn.clicked.connect(self.clicked_run_btn)
        self.signal_running.connect(
            lambda: self.run_btn.setStyleSheet("QPushButton {background-color: #8DBF8B; font-family: 'Times New Roman'; font-size: 20pt; font-weight: bold;}"))
        self.signal_pausing.connect(
            lambda: self.run_btn.setStyleSheet("QPushButton {font-family: 'Times New Roman'; font-size: 20pt; font-weight: bold;}"))

        for btn in [self.speak_btn, self.run_btn]:
            btn.setStyleSheet("QPushButton {font-family: 'Times New Roman'; font-size: 20pt; font-weight: bold;}")

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
                            zoom:17, // 初始化地图层级
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

    def update_map(self, txt):
        # 将用户输入解析为起点和终点
        navigation_destination = txt
        if navigation_destination is not None:
            print(navigation_destination)
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

    def start_recording(self):
        if not self.btn_clicked:
            self.btn_clicked = True
            self.signal_update_label.emit("正在录音...")
            threading.Thread(target=self.thread_record).start()

    def stop_recording(self):
        self.btn_clicked = False
        while self.thread_record_running:
            print("self.thread_record_running", self.thread_record_running)
            pass
        self.signal_update_label.emit('正在识别...')
        while self.thread_recognize_running:
            print("self.thread_recognize_running", self.thread_recognize_running)
            pass

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

        while self.btn_clicked:
            data = stream.read(chunk)
            self.frames.append(data)
        stream.stop_stream()
        stream.close()
        p.terminate()
        self.thread_recognize_running = True
        self.thread_record_running = False
        threading.Thread(target=self.thread_recognize).start()
        self.thread_recognize_running = False

    def thread_recognize(self):
        recognizer = sr.Recognizer()
        # 将音频数据转换为 AudioData 对象
        audio_data = sr.AudioData(b''.join(self.frames), 16000, 2)
        try:
            # 使用Google Web Speech API进行中文语音识别
            text = recognizer.recognize_google(audio_data, language='zh-CN')
            self.signal_update_label.emit(text)
            self.signal_update_map.emit(text)
        except sr.UnknownValueError:
            self.signal_update_label.emit("Google Web Speech API 无法识别音频")
        except sr.RequestError as e:
            self.signal_update_label.emit(f"无法连接到 Google Web Speech API 服务; {e}")
        self.frames.clear()

    def clicked_run_btn(self):
        self.btn_run_is_clicked = ~self.btn_run_is_clicked
        if self.btn_run_is_clicked:
            self.signal_running.emit()

            threading.Thread(target=self.thread_play_3D_audio).start()

            YOLO_model = YOLO('utils/yolov8s-seg.pt')
            tracks = dict()
            speeds = dict()
            collision_probability = dict()

            while self.btn_run_is_clicked and self.threads_running:
                t0 = time.time()
                img_BGR = self.img_BGR.copy()
                img_D = self.img_D.copy()
                # 创建Rotation对象
                quat = np.array([self.imu.rotationVector.i,
                                 self.imu.rotationVector.j,
                                 self.imu.rotationVector.k,
                                 self.imu.rotationVector.real])
                rotation = R.from_quat(quat)

                object_result = YOLO_model.track(img_BGR, persist=True, verbose=False)[0]
                azimuth_pitch = []
                for i in range(len(object_result)):
                    box = object_result.boxes[i]
                    mask = object_result.masks[i].data[0][12: -12, :].int().cpu()
                    bbox = box.xyxy.cpu().round().int().tolist()[0]
                    name = object_result.names[box.cls.cpu().round().int().tolist()[0]]
                    if name not in objs:
                        continue

                    rows, cols = np.where((mask == 1) & (img_D != 0))
                    if len(rows) > 0 and len(cols) > 0:
                        distances = img_D[rows, cols]
                        # 计算百分位数
                        percentile_low = 10
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
                        # 确保过滤后的坐标有效且不为空
                        if coords.size > 0:
                            ids = object_result.boxes.id
                            if ids is not None:
                                ids = ids.int().cpu().tolist()
                                id = ids[i]
                                if id not in tracks.keys():
                                    tracks[id] = list()
                                tracks[id].append((np.median(coords[:, 0]),  # X
                                                   np.median(coords[:, 1]),  # Y
                                                   t0))  # T
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

                speed_ = speeds.copy()
                for id in speed_:
                    if id not in tracks.keys():
                        del speeds[id], collision_probability[id]
                color = '#FAF5E4'
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
                    self.lable_sign.setText(f"{cp * 100:.1f}%")
                self.lable_sign.setStyleSheet(f"background-color: {color};")
                QApplication.processEvents()

        else:
            self.signal_pausing.emit()

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
        while self.threads_running:
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
            else:
                time.sleep(0.5)

    def closeEvent(self, event):
        reply = QMessageBox.question(self, '确认', '你确定要退出程序吗？',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.threads_running = False  # 关闭窗口时将 running 设置为 False
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
