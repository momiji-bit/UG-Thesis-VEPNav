import cv2
from ultralytics import YOLO

# Load a pretrained YOLOv10n model
model = YOLO("yolov10n.pt")

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()

    # results = model(frame)
    #
    # results[0].show()