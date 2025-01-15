import random
import time
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from utils.OAK_Info import oak_device
from utils.tools import *
from ultralytics import YOLO

queue = oak_device.getOutputQueue("xoutGrp", 10, False)
YOLO_model = YOLO('../utils/yolov8l-seg.pt')
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

    # IMU 数据处理
    rotationVector = imu.rotationVector
    i, j, k, real = rotationVector.i, rotationVector.j, rotationVector.k, rotationVector.real
    quat = np.array([i, j, k, real])
    rotation = R.from_quat(quat)

    # BEV seg
    BEV = draw(rotation)


    seg = np.ones((360, 640, 3), np.uint8) * 100

    # 深度图伪色彩显示
    img_D_255 = cv2.normalize(img_D, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    img_D_COLORMAP = cv2.applyColorMap((255 - img_D_255), cv2.COLORMAP_MAGMA)
    img_mix = np.hstack((img_BGR, img_D_COLORMAP))

    object_result = YOLO_model.track(img_BGR, persist=True, verbose=False)[0]

    for i in range(len(object_result)):
        box = object_result.boxes[i]
        mask = object_result.masks[i].data[0][12: -12, :].int().cpu()
        bbox = box.xyxy.cpu().round().int().tolist()[0]
        name = object_result.names[box.cls.cpu().round().int().tolist()[0]]
        if name not in objs:
            continue

        color = generate_random_color(int(box.cls.cpu()))
        seg[mask == 1] = color

        color_expanded = np.zeros_like(img_BGR)
        color_expanded[:, :] = color
        alpha = 0.5
        img_BGR[mask == 1] = (1 - alpha) * img_BGR[mask == 1] + alpha * color_expanded[mask == 1]

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
            if obj_filtered_coords.size > 0:
                BEV[obj_filtered_coords[:, 1], obj_filtered_coords[:, 0]] = color

    height, width, _ = BEV.shape
    start_x = width // 2 - 400
    start_y = height // 2 - 400
    end_x = start_x + 800
    end_y = start_y + 800
    BEV = BEV[start_y:end_y, start_x:end_x]
    cv2.imshow('BEV', BEV)

    cv2.imshow('img_BGR', img_BGR)
    cv2.imshow('img_D_COLORMAP', img_D_COLORMAP)
    cv2.imshow('seg', seg)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

