import socket
import threading
import pickle
import numpy as np
from ultralytics import YOLO
from utils.models import RoadBoundGetter
import cv2
import asyncio

# 设置服务器参数
HOST = '127.0.0.1'
PORT = 65432

# 创建一个套接字
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen()

async def handle_client(client_socket, addr):
    print(f"Connected by {addr}")
    road_model = RoadBoundGetter(scale=0.3, density=10, pretrain="utils/road_model_maxmIOU75.pth")
    YOLO_model = YOLO('utils/yolov8s-seg.pt')
    loop = asyncio.get_event_loop()
    while True:
        try:
            # 接收数据
            data = b''
            while True:
                packet = await loop.sock_recv(client_socket, 4096)
                if packet.endswith(b'$END#'):
                    data += packet[:-5]
                    break
                data += packet

            # 反序列化数据
            if data:
                received_data = pickle.loads(data)
                rgb_image_compressed = received_data['rgb']

                # 解压缩RGB图像
                rgb_image = cv2.imdecode(np.frombuffer(rgb_image_compressed, np.uint8), cv2.IMREAD_COLOR)

                # 进行神经网络推断
                result = YOLO_model.track(rgb_image, persist=True, verbose=False)[0]
                road_masked = road_model(rgb_image)[0].cpu()
                data = {'yolo': result, 'road': road_masked}

                # 序列化结果并发送回客户端
                await loop.sock_sendall(client_socket, pickle.dumps(data) + b'#END$')
        except Exception as e:
            print(f"Error with {addr}: {e}")
            break
    print(f"Connection closed by {addr}")
    client_socket.close()

async def accept_clients():
    loop = asyncio.get_event_loop()
    while True:
        client_socket, addr = await loop.sock_accept(server_socket)
        asyncio.create_task(handle_client(client_socket, addr))

print("Server is listening...")

loop = asyncio.get_event_loop()
loop.run_until_complete(accept_clients())
