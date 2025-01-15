import threading
import random
import time
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from utils.tools import *
from ultralytics import YOLO
import numpy as np
import time
import cv2

thread_is_running = False
img_BGR = None
img_D = None
imu = None

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


class ids_track:
    def __init__(self):
        self.ids_track = dict()

    def append(self, id, point):
        if id not in self.ids_track.keys():
            self.ids_track[id] = []

        if len(self.ids_track[id]) > 0:
            x_new, y_new, t_new = point
            x_last, y_last, t_last = self.ids_track[id][-1]

            threshold = 100  # cm/s
            distance = np.sqrt((x_last - x_new) ** 2 + (y_last - y_new) ** 2)
            v = distance/(t_new-t_last)
            if v < threshold:
                self.ids_track[id].append(point)
        else:
            self.ids_track[id].append(point)

    def draw(self, BEV, id, xy, name):
        random.seed(id)  # 设置随机数种子
        x_delt = random.randint(20, 60)
        y_delt = random.randint(0, 30)
        x1, y1 = xy
        x2, y2 = x1 + x_delt, y1 - y_delt
        cv2.line(BEV, (x1, y1), (x2, y2), (255, 255, 255), thickness=1)
        cv2.putText(BEV, f"{id} {name}",
                    (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        return BEV

    def update(self, ids):
        # 使用复制的键列表来避免在遍历时修改字典
        ids_to_check = list(self.ids_track.keys())

        if ids is None:
            ids = []

        for id in ids_to_check:
            if id in ids:
                if len(self.ids_track[id]) > 30:
                    self.ids_track[id].pop(0)
            else:
                self.ids_track[id].pop(0)

            if not self.ids_track[id]:
                del self.ids_track[id]

    def tracks(self):
        return self.ids_track

def refresh(path):
    global thread_is_running, img_BGR, img_D, imu
    # 加载数据
    data = np.load(path)

    frames_BGR_loaded = data['frames_BGR']
    frames_D_loaded = data['frames_D']
    IMU_loaded = data['IMU']
    timestamps_loaded = data['timestamps']

    for i in range(1, len(frames_BGR_loaded)):
        thread_is_running =True
        t0 = time.time() * 1000  # s -> ms

        img_BGR = frames_BGR_loaded[i]
        img_D = frames_D_loaded[i]
        imu = IMU_loaded[i]
        time_to_wait = timestamps_loaded[i] - timestamps_loaded[i - 1]  # ms

        # img_D_255 = (img_D * (255.0 / 10000)).astype(np.uint8)
        # img_D_COLORMAP = cv2.applyColorMap((255 - img_D_255), cv2.COLORMAP_MAGMA)
        # cv2.imshow("Demo", np.hstack((img_BGR, img_D_COLORMAP)))

        t = max(1, int(time_to_wait - (time.time() * 1000 - t0)))
        # print(f"帧间时间差：{time_to_wait:.1f}ms, 实际等待时间：{t:.1f}ms, 程序处理时间：{(time.time() * 1000 - t0):.1f}ms")
        if cv2.waitKey(t) & 0xFF == ord("q"):
            break

    thread_is_running = False
    cv2.destroyAllWindows()


if __name__ == '__main__':
    threading.Thread(target=refresh, args=("oakCam_Data_1716394283.3844376.npz",)).start()

    while True:
        if thread_is_running:
            break
        time.sleep(0.01)

    ids_track = ids_track()
    while thread_is_running:
        t0 = time.time()

        i, j, k, real = imu
        quat = np.array([i, j, k, real])
        rotation = R.from_quat(quat)

        # BEV seg
        BEV = draw(rotation)

        # seg = np.ones((360, 640, 3), np.uint8) * 100
        seg = img_BGR.copy()

        # 深度图伪色彩显示
        img_D_255 = (img_D * (255.0 / 10000)).astype(np.uint8)
        img_D_COLORMAP = cv2.applyColorMap((255 - img_D_255), cv2.COLORMAP_MAGMA)
        img_mix = np.hstack((img_BGR, img_D_COLORMAP))

        object_result = YOLO_model.track(img_BGR, persist=True, verbose=False)[0]

        for i in range(len(object_result)):
            box = object_result.boxes[i]
            mask = object_result.masks[i].data[0][12: -12, :].int().cpu()
            bbox = box.xyxy.cpu().round().int().tolist()[0]
            name = object_result.names[box.cls.cpu().round().int().tolist()[0]]
            ids = object_result.boxes.id
            if name not in objs:
                continue

            color = generate_random_color(int(box.cls.cpu()))
            seg[mask == 1] = color

            rows, cols = np.where((mask == 1) & (img_D != 0))

            if len(rows) > 0 and len(cols) > 0:
                distances = img_D[rows, cols]
                # 点云过滤
                percentile_low = 5
                percentile_high = 65
                low_threshold = np.percentile(distances, percentile_low)
                high_threshold = np.percentile(distances, percentile_high)
                selected_rows = rows[(distances >= low_threshold) & (distances <= high_threshold)]
                selected_cols = cols[(distances >= low_threshold) & (distances <= high_threshold)]
                selected_distances = distances[(distances >= low_threshold) & (distances <= high_threshold)]
                bboxes = np.column_stack([selected_cols, selected_rows, selected_cols, selected_rows])
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

                # 应用mask过滤坐标
                obj_filtered_coords = coords[mask]
                if obj_filtered_coords.size > 0:
                    BEV[obj_filtered_coords[:, 1], obj_filtered_coords[:, 0]] = color
                    cv2.putText(seg, f"{id} {name}",
                                (int(np.median(selected_cols)), int(np.median(selected_rows))),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    x = int(np.median(obj_filtered_coords[:, 0]))  # X
                    y = int(np.median(obj_filtered_coords[:, 1]))  # Y
                    t = t0

                    if ids is not None:
                        ids = ids.int().cpu().tolist()
                        id = ids[i]
                        ids_track.append(id, [x, y, t])  # 压入当前坐标和时间戳
                        BEV = ids_track.draw(BEV, id, [x, y], name)

                    cv2.circle(BEV, (x, y), 5, (0, 0, 255), -1)

            ids_track.update(ids)  # 更新所有轨迹
            for id in ids_track.tracks():
                id_track = ids_track.tracks()[id]
                if len(id_track) == 1:
                    x, y, t = id_track[0]
                    cv2.circle(BEV, (x, y), 2, (0, 0, 200), -1)
                else:
                    for i in range(1, len(id_track)):
                        x1, y1, _ = id_track[i - 1]
                        x2, y2, _ = id_track[i]
                        cv2.line(BEV, (x1, y1), (x2, y2), (0, 0, 200), thickness=2)

        height, width, _ = BEV.shape
        start_x = width // 2 - 400
        start_y = height // 2 - 400
        end_x = start_x + 800
        end_y = start_y + 800
        BEV = BEV[start_y:end_y, start_x:end_x]
        cv2.imshow('BEV', BEV)

        img_mix = np.vstack((img_D_COLORMAP, seg))
        cv2.imshow('Demo', img_mix)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()
