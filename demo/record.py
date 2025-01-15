import time

from utils.OAK_Info import oak_device
from utils.tools import *
from ultralytics import YOLO
import numpy as np


objs = [labels_dict[i] for i in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,56,57,58,59,60,61,62,63,67,68,72,73]]

frames_BGR = []
frames_D = []
IMU = []
timestamps = []
queue = oak_device.getOutputQueue("xoutGrp", 10, False)
while True:
    ts = time.time() * 1000  # 单位ms

    msgGrp = queue.get()
    img_BGR = msgGrp['color'].getCvFrame()
    img_D = msgGrp['depth'].getCvFrame()
    imu = msgGrp['imu'].packets[0]

    img_D_255 = (img_D * (255.0 / 10000)).astype(np.uint8)
    img_D_COLORMAP = cv2.applyColorMap((255 - img_D_255), cv2.COLORMAP_MAGMA)

    img_mix = np.hstack((img_BGR, img_D_COLORMAP))

    rotationVector = imu.rotationVector
    i, j, k, real = rotationVector.i, rotationVector.j, rotationVector.k, rotationVector.real


    frames_BGR.append(img_BGR)
    frames_D.append(img_D)
    IMU.append((i, j, k, real))
    timestamps.append(ts)
    cv2.putText(img_mix, f"i:{i:0.4f} j:{j:0.4f} k:{k:0.4f} real:{real:0.4f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('Demo', img_mix)
    print(time.time() * 1000 - ts, 'ms')  # 单位ms
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cv2.destroyAllWindows()
np.savez(f'./oakCam_Data_{time.time()}.npz',
         frames_BGR=np.array(frames_BGR),
         frames_D=np.array(frames_D),
         IMU=np.array(IMU),
         timestamps=np.array(timestamps))
