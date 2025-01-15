import random
import time
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from utils.OAK_Info import oak_device
from utils.tools import *
from ultralytics import YOLO

queue = oak_device.getOutputQueue("xoutGrp", 10, False)
YOLO_model = YOLO('../utils/yolov8s-seg.pt')
objs = [labels_dict[i] for i in
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 56, 57, 58, 59, 60, 61, 62, 63, 67, 68, 72, 73]]


def generate_random_color(seed=None):
    """生成并返回一个随机颜色"""
    random.seed(seed)  # 设置随机数种子
    red = random.randint(0, 255)
    green = random.randint(0, 255)
    blue = random.randint(0, 255)
    return red, green, blue


def draw(rotation):
    BEV = BEV_Background_2()

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

    return BEV


while True:
    t0 = time.time()

    msgGrp = queue.get()
    img_BGR = msgGrp['color'].getCvFrame()
    img_D = msgGrp['depth'].getCvFrame()
    imu = msgGrp['imu'].packets[0]

    img_D_255 = (img_D * (255.0 / 10000)).astype(np.uint8)
    img_D_COLORMAP = cv2.applyColorMap((255 - img_D_255), cv2.COLORMAP_MAGMA)

    # IMU 数据处理
    rotationVector = imu.rotationVector
    i, j, k, real = rotationVector.i, rotationVector.j, rotationVector.k, rotationVector.real
    quat = np.array([i, j, k, real])
    rotation = R.from_quat(quat)

    # BEV seg
    BEV = draw(rotation)

    # 假设img_D是一个2D深度图像
    h, w = img_D.shape
    selected_cols, selected_rows = np.meshgrid(np.arange(w), np.arange(h))
    selected_distances = img_D.flatten()
    bboxes = np.column_stack([selected_cols.flatten(), selected_rows.flatten(),
                              selected_cols.flatten(), selected_rows.flatten()])
    Xs = getSpatialCoordinates(selected_distances, bboxes, 'x') / 10
    Ys = getSpatialCoordinates(selected_distances, bboxes, 'y') / 10
    Zs = selected_distances / 10
    points = np.column_stack((Ys, Xs, Zs))
    rotated_points = rotation.apply(points)

    coords = rotated_points[:, :2]  # 获取前两个维度
    coords[:, 0] += 800  # X坐标偏移
    coords[:, 1] = -coords[:, 1] + 800  # Y坐标偏移并反向
    coords = coords.astype(int)
    mask = (coords[:, 0] >= 0) & (coords[:, 0] < BEV.shape[1]) & \
           (coords[:, 1] >= 0) & (coords[:, 1] < BEV.shape[0])
    obj_filtered_coords = coords[mask]
    if obj_filtered_coords.size > 0:
        # 根据 obj_filtered_coords 过滤 selected_rows 和 selected_cols
        filtered_rows = selected_rows.flatten()[mask]
        filtered_cols = selected_cols.flatten()[mask]

        BEV[obj_filtered_coords[:, 1], obj_filtered_coords[:, 0]] = img_BGR[filtered_rows, filtered_cols]

    height, width, _ = BEV.shape
    start_x = width // 2 - 400
    start_y = height // 2 - 400
    end_x = start_x + 800
    end_y = start_y + 800
    BEV = BEV[start_y:end_y, start_x:end_x]
    cv2.imshow('BEV', BEV)
    cv2.imshow('BGR', img_BGR)
    cv2.imshow('D', img_D_COLORMAP)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break