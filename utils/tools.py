from scipy.signal import fftconvolve
from pysofaconventions import SOFAFile
import numpy as np
import math
import cv2

hfov = 68.7938003540039
vfov = 42.12409823672219

labels_dict = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush"
}


def BEV_Background():
    BEV = np.zeros((1000, 1000, 3), np.uint8) * 100

    alpha = (180 - hfov) / 2
    center = int(BEV.shape[1] / 2)
    max_p = BEV.shape[0] - int(math.tan(math.radians(alpha)) * center)
    fov_cnt = np.array([
        (0, BEV.shape[0]),
        (BEV.shape[1], BEV.shape[0]),
        (BEV.shape[1], max_p),
        (center, BEV.shape[0]),
        (0, max_p),
        (0, BEV.shape[0]),
    ])
    cv2.fillPoly(BEV, [fov_cnt], color=(255, 255, 255))

    for d in range(900,0,-100):
        cv2.line(BEV, (0, d), (BEV.shape[0], d), (200, 200, 200), 1)
        cv2.putText(BEV, str(int((1000-d)/100))+'m', (0, d+10), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (150, 150, 150), 1)

    return BEV

def BEV_Background_2():
    R = 800
    BEV = np.ones((2*R, 2*R, 3), np.uint8) * 100  # 半径8m
    cv2.circle(BEV, (R, R), radius=50, color=(0, 0, 233), thickness=1)
    cv2.circle(BEV, (R, R), radius=100, color=(133, 133, 133), thickness=2)
    cv2.circle(BEV, (R, R), radius=300, color=(133, 133, 133), thickness=2)
    cv2.circle(BEV, (R, R), radius=500, color=(133, 133, 133), thickness=2)
    cv2.circle(BEV, (R, R), radius=800, color=(133, 133, 133), thickness=2)

    return BEV


def getSpatialCoordinates(dist, bbox, axis='x'):
    if axis == 'x':
        center_pos = (bbox[:, 0] + bbox[:, 2]) / 2.0
        diff_from_center = 320 - center_pos
        cam_width = np.tan(np.radians(hfov / 2)) * dist
        coord = cam_width * (diff_from_center / 320)
    elif axis == 'y':
        center_pos = (bbox[:, 1] + bbox[:, 3]) / 2.0
        diff_from_center = center_pos - 180
        cam_height = np.tan(np.radians(vfov / 2)) * dist
        coord = cam_height * (diff_from_center / 180)
    else:
        raise ValueError("Axis must be 'x' or 'y'")

    return coord

def calculate_azimuth_pitch(col, row, distance):
    dx = 320 - col
    dy = 160 - row
    if dx >= 0:
        azimuth_angle = np.degrees(np.arctan(dx / distance))
    else:
        azimuth_angle = 360 + np.degrees(np.arctan(dx / distance))
    if dy >= 0:
        pitch_angle = np.degrees(np.arctan(dy / distance))
    else:
        pitch_angle = np.degrees(np.arctan(dy / distance))
    return azimuth_angle, pitch_angle

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