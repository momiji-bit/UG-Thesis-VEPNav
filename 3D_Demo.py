import time

import numpy as np
import sounddevice as sd
from pysofaconventions import SOFAFile
from scipy.signal import fftconvolve
import librosa
import cv2
import time
import threading

click_pos = None
mouse_pressed = False  # 用于跟踪鼠标按键是否被按下
distance = None
pitch_angle = None
azimuth_angle = None
current_position = 0


def calculate_distance_and_angle(x, y):
    dx = x - 500
    dy = 500 - y  # 注意坐标系的方向，图像的y轴是向下的
    distance = np.sqrt(dx ** 2 + dy ** 2)
    pitch_angle_rad = np.arctan(160 / distance)
    pitch_angle = - np.degrees(pitch_angle_rad)
    azimuth_angle = (np.degrees(np.arctan2(-dx, dy)) + 360) % 360
    return distance, azimuth_angle, pitch_angle


def mouse_callback(event, x, y, flags, param):
    global click_pos, BEV, mouse_pressed, distance, azimuth_angle, pitch_angle
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_pressed = True
    elif event == cv2.EVENT_MOUSEMOVE and mouse_pressed:
        x = max(0, min(x, BEV.shape[1] - 1))
        y = max(0, min(y, BEV.shape[0] - 1))
        click_pos = (x, y)
        distance, azimuth_angle, pitch_angle = calculate_distance_and_angle(x, y)  # 计算距离和角度
        # 每次拖动时重新绘制背景来清除旧的坐标显示
        BEV = background()
        cv2.putText(BEV,
                    f'X: {x}, Y: {y}, Distance: {distance:.2f}, Azimuth_angle: {azimuth_angle:.2f}, Pitch_angle: {pitch_angle:.2f}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.circle(BEV, click_pos, 10, (255, 0, 255), -1)
    elif event == cv2.EVENT_LBUTTONUP:
        mouse_pressed = False


def background():
    BEV = np.ones((1001, 1001, 3), np.uint8) * 255
    cv2.circle(BEV, (500, 500), 5, (0, 0, 255), -1)
    cv2.circle(BEV, (500, 500), 100, (0, 0, 255), 1)
    cv2.circle(BEV, (500, 500), 300, (0, 0, 255), 1)
    cv2.line(BEV, (500, 500), (500, 0), (0, 0, 255), 1)
    return BEV


class HRFT():
    def __init__(self, sofa_path):
        # 加载SOFA文件和HRTF数据
        self.sofa = SOFAFile(sofa_path, 'r')
        # 获取SOFA文件中的所有测量位置
        self.positions = self.sofa.getVariableValue('SourcePosition')

    def get_LR_HRFT(self, pitch_angle, azimuth_angle):
        closest_index = np.argmin(
            np.sqrt((self.positions[:, 0] - azimuth_angle) ** 2 + (self.positions[:, 1] - pitch_angle) ** 2))
        hrtf_left = self.sofa.getDataIR()[closest_index, 0, :]
        hrtf_right = self.sofa.getDataIR()[closest_index, 1, :]

        return hrtf_left, hrtf_right

    def run(self, azimuth=0, elevation=0, c=(0, 0, 0.1)):
        # 找到与目标角度最接近的HRTF
        distances = np.sqrt((self.positions[:, 0] - azimuth) ** 2 + (self.positions[:, 1] - elevation) ** 2)
        closest_index = np.argmin(
            np.sqrt((self.positions[:, 0] - azimuth) ** 2 + (self.positions[:, 1] - elevation) ** 2))
        hrtf_left = self.sofa.getDataIR()[closest_index, 0, :]
        hrtf_right = self.sofa.getDataIR()[closest_index, 1, :]

        # 设置采样率
        samplerate = 44100  # Hz
        # 设置声音的持续时间为0.1秒
        duration = c[-1]  # 秒
        # 重新生成纯正弦波声音，使用不同的基频
        frequencies = [220, 330, 440, 660, 880]  # A3, E4, A4, E5, A5
        # 时间数组
        t = np.linspace(0, duration, int(samplerate * duration), endpoint=False)

        if c[0] == 0:  # 正弦波
            wave = np.sin(2 * np.pi * frequencies[c[1]] * t)
        elif c[0] == 1:  # 锯齿波
            wave = sum(np.sin(2 * np.pi * (2 * i + 1) * frequencies[c[1]] * t) / (2 * i + 1) for i in range(5))
        elif c[0] == 2:  # 锯齿波
            wave = sum(np.sin(2 * np.pi * i * frequencies[c[1]] * t) / i for i in range(1, 10))
        elif c[0] == 3:  # 三角波
            wave = sum((-1) ** (i) * np.sin(2 * np.pi * (2 * i + 1) * frequencies[c[1]] * t) / (2 * i + 1) ** 2 for i in
                       range(5))
        elif c[0] == 4:  # 复合波形
            wave = np.sin(2 * np.pi * frequencies[c[1]] * t) + 0.5 * np.sin(2 * np.pi * 3 * frequencies[c[1]] * t)
        else:
            return -1

        # 对音频信号进行卷积
        convolved_left = fftconvolve(wave, hrtf_left, mode='same')
        convolved_right = fftconvolve(wave, hrtf_right, mode='same')
        # 合并左右声道
        stereo_signal = np.vstack((convolved_left, convolved_right)).T

        return stereo_signal


hrft = HRFT('./utils/hrtf_nh94.sofa')
# wave, sample_rate = librosa.load('./utils/sounds/sci-fi-glitch-sound.wav', sr=None, mono=False)
wave, sample_rate = librosa.load('./utils/sounds/beep2.wav', sr=None, mono=False)

BEV = background()
cv2.namedWindow("BEV")
cv2.setMouseCallback("BEV", mouse_callback)

# # 初始化信号量，max_threads 是你希望同时运行的最大线程数
# max_threads = 5
# thread_semaphore = threading.Semaphore(max_threads)

# 定义一个全局的输出流
stream = sd.OutputStream(samplerate=44100, channels=2)
stream.start()


def apply_hanning_window(wave, window_length=1024):
    """对波形的开始和结束应用汉宁窗以平滑边缘"""
    if window_length <= 0 or window_length * 2 > len(wave):
        return wave  # 窗口长度不合理时返回原波形

    # 创建汉宁窗
    window = np.hanning(window_length * 2)

    # 对于立体声或多声道波形，需要调整窗以匹配波形的形状
    if wave.ndim > 1:
        # 扩展窗以匹配波形的声道数
        window = np.tile(window[:, np.newaxis], (1, wave.shape[1]))

    # 将窗分成两部分并应用到波形的开始和结束
    window_start, window_end = window[:window_length], window[window_length:]

    # 对于多维数组，需要确保操作沿着正确的轴进行
    wave[:window_length] *= window_start
    wave[-window_length:] *= window_end

    return wave


def play_audio(waves, window_length=1024):
    """向已经开启的输出流写入音频数据进行播放，并在拼接波形前平滑边缘"""
    global stream
    if waves:
        # 在拼接前对每个波形应用汉宁窗平滑边缘
        smoothed_waves = [apply_hanning_window(wave.copy(), window_length) for wave in waves]

        # 拼接波形
        concatenated_waves = np.concatenate(smoothed_waves, axis=0)

        # 转换数据类型为 'float32'
        concatenated_waves = concatenated_waves.astype(np.float32)

        # 确保数据是C连续的
        concatenated_waves = np.ascontiguousarray(concatenated_waves)

        # 播放音频
        stream.write(concatenated_waves)


import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def A():
    global stereo_signal
    x = y = click_pos
    plt.ion()  # 开启交互模式
    fig, ax = plt.subplots(figsize=(6, 2))
    canvas = FigureCanvas(fig)

    while True:
        waves = []
        cv2.imshow("BEV", BEV)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break
        if distance is None or pitch_angle is None or azimuth_angle is None:
            continue

        hrtf_left, hrtf_right = hrft.get_LR_HRFT(pitch_angle, azimuth_angle)
        convolved_left = fftconvolve(wave[0, :], hrtf_left, mode='same') * (1 / (((distance - 100) / 200) ** 2))
        convolved_right = fftconvolve(wave[1, :], hrtf_right, mode='same') * (1 / (((distance - 100) / 200) ** 2))

        stereo_signal = np.vstack((convolved_left, convolved_right)).T

        if (x, y) != click_pos:
            waves.append(stereo_signal)

        play_audio(waves)

        # 绘制波形图
        ax.clear()
        ax.plot(stereo_signal[:, 0], color='red', label='Left Channel', linewidth=0.3)
        ax.plot(stereo_signal[:, 1], color='blue', label='Right Channel', linewidth=0.3)
        ax.legend(loc='upper left')
        ax.set_ylim([-0.5, 0.5])  # 设置固定的纵坐标范围
        canvas.draw()

        # 将图像嵌入到BEV中
        buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        buf = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)

        # 将波形图放置在BEV的左下角
        BEV_height, BEV_width, _ = BEV.shape
        buf_height, buf_width, _ = buf.shape
        BEV[BEV_height - buf_height:, :buf_width] = buf

        cv2.imshow("BEV", BEV)

    cv2.destroyAllWindows()
    plt.ioff()  # 关闭交互模式
    plt.show()


A()
