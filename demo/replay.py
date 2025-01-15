import numpy as np
import time
import cv2

# 加载数据
data = np.load('oakCam_Data_1716388772.0965369.npz')

frames_BGR_loaded = data['frames_BGR']
frames_D_loaded = data['frames_D']
IMU_loaded = data['IMU']
timestamps_loaded = data['timestamps']

for i in range(1, len(frames_BGR_loaded)):
    t0 = time.time() * 1000  # s -> ms

    img_BGR = frames_BGR_loaded[i]
    img_D = frames_D_loaded[i]
    imu = IMU_loaded[i]
    time_to_wait = timestamps_loaded[i] - timestamps_loaded[i - 1]  # ms

    img_D_255 = (img_D * (255.0 / 10000)).astype(np.uint8)
    img_D_COLORMAP = cv2.applyColorMap((255 - img_D_255), cv2.COLORMAP_MAGMA)
    cv2.imshow("Demo", np.hstack((img_BGR, img_D_COLORMAP)))

    t = max(1, int(time_to_wait - (time.time() * 1000 - t0)))
    print(f"帧间时间差：{time_to_wait:.1f}ms, 实际等待时间：{t:.1f}ms, 程序处理时间：{(time.time() * 1000 - t0):.1f}ms")
    if cv2.waitKey(t) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
