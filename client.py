import socket
import pickle
import time
import cv2
import numpy as np
import random
import threading
import asyncio

# 设置客户端参数
HOST = '127.0.0.1'
PORT = 65432

# 创建一个套接字
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((HOST, PORT))


def generate_random_color(seed=None):
    """生成并返回一个随机颜色"""
    random.seed(seed)  # 设置随机数种子
    red = random.randint(0, 255)
    green = random.randint(0, 255)
    blue = random.randint(0, 255)
    return (red, green, blue)


async def send_images(rgb_image, depth_image):
    # 压缩RGB图像
    _, rgb_image_compressed = cv2.imencode('.jpg', rgb_image, [cv2.IMWRITE_JPEG_QUALITY, 80])
    rgb_image_compressed = rgb_image_compressed.tobytes()

    # 构建数据包
    # data = {'rgb': rgb_image_compressed, 'depth': depth_image}
    data = {'rgb': rgb_image_compressed}
    serialized_data = pickle.dumps(data)
    serialized_data += b'$END#'  # 添加结束标记

    loop = asyncio.get_event_loop()
    await loop.sock_sendall(client_socket, serialized_data)

    # 接收结果
    received_data = b''
    while True:
        packet = await loop.sock_recv(client_socket, 4096)
        if packet.endswith(b'#END$'):
            received_data += packet[:-5]
            break
        received_data += packet

    # 反序列化结果
    result = pickle.loads(received_data)
    return result


async def capture_and_send():
    try:
        # 打开摄像头（0表示默认摄像头）
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Cannot open camera")
            return

        while True:
            t0 = time.time()
            # 从摄像头读取一帧
            ret, frame = cap.read()

            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            start_row = (480 - 360) // 2
            frame = frame[start_row:start_row + 360, 0:640]

            # 假设深度图像是一个全零数组（需要实际深度摄像头来获取真实数据）
            depth_image = np.zeros((frame.shape[0], frame.shape[1]))

            # 发送图像并接收结果
            received_data = await send_images(frame, depth_image)
            yolo_result = received_data['yolo']
            road_result = received_data['road']
            frame[road_result==1] = (233,233,233)
            for i in range(len(yolo_result)):
                box = yolo_result.boxes[i]
                mask = yolo_result.masks[i].data[0][12: -12, :].int().cpu()
                bbox = box.xyxy.cpu().round().int().tolist()[0]
                name = yolo_result.names[box.cls.cpu().round().int().tolist()[0]]
                color = generate_random_color(int(box.cls.cpu()))
                cv2.rectangle(frame, bbox[:2], bbox[2:], color, 2)
            cv2.putText(frame, f"{int((time.time() - t0) * 1000)}ms", (0, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.8,
                        (0, 0, 255))
            # 显示捕获的图像
            cv2.imshow('frame', frame)

            if cv2.waitKey(1) == ord('q'):
                break

    except KeyboardInterrupt:
        print("Client stopped.")

    finally:
        # 释放摄像头并关闭窗口
        cap.release()
        cv2.destroyAllWindows()
        client_socket.close()


# 启动捕获和发送任务
asyncio.run(capture_and_send())
