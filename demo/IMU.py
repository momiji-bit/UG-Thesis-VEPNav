#!/usr/bin/env python3

import cv2
import depthai as dai
import time
import math
import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
imu = pipeline.create(dai.node.IMU)
xlinkOut = pipeline.create(dai.node.XLinkOut)

xlinkOut.setStreamName("imu")

# enable ACCELEROMETER_RAW at 500 hz rate
imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, 1000)
# enable GYROSCOPE_RAW at 400 hz rate
imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_RAW, 1000)
imu.setBatchReportThreshold(1)
imu.setMaxBatchReports(10)

# Link plugins IMU -> XLINK
imu.out.link(xlinkOut.input)

# Initialize the Tkinter window
root = tk.Tk()
root.title("IMU Data")

# Create variables to hold sensor data
acc_x_var = tk.StringVar()
acc_y_var = tk.StringVar()
acc_z_var = tk.StringVar()
acc_o_var = tk.StringVar()
acc_ts_var = tk.StringVar()

gyro_x_var = tk.StringVar()
gyro_y_var = tk.StringVar()
gyro_z_var = tk.StringVar()
gyro_o_var = tk.StringVar()
gyro_ts_var = tk.StringVar()

# Create and place labels for accelerometer data
ttk.Label(root, text="Accelerometer:").grid(column=0, row=0, padx=10, pady=10)
ttk.Label(root, textvariable=acc_x_var).grid(column=0, row=1, padx=10, pady=5)
ttk.Label(root, textvariable=acc_y_var).grid(column=0, row=2, padx=10, pady=5)
ttk.Label(root, textvariable=acc_z_var).grid(column=0, row=3, padx=10, pady=5)
ttk.Label(root, textvariable=acc_o_var).grid(column=0, row=4, padx=10, pady=5)
ttk.Label(root, textvariable=acc_ts_var).grid(column=0, row=5, padx=10, pady=5)

# Create and place labels for gyroscope data
ttk.Label(root, text="Gyroscope:").grid(column=1, row=0, padx=10, pady=10)
ttk.Label(root, textvariable=gyro_x_var).grid(column=1, row=1, padx=10, pady=5)
ttk.Label(root, textvariable=gyro_y_var).grid(column=1, row=2, padx=10, pady=5)
ttk.Label(root, textvariable=gyro_z_var).grid(column=1, row=3, padx=10, pady=5)
ttk.Label(root, textvariable=gyro_o_var).grid(column=1, row=4, padx=10, pady=5)
ttk.Label(root, textvariable=gyro_ts_var).grid(column=1, row=5, padx=10, pady=5)

# Create a Matplotlib figure and axes
fig = Figure(figsize=(10, 5), dpi=100)
ax_acc = fig.add_subplot(121, title="Accelerometer Data")
ax_gyro = fig.add_subplot(122, title="Gyroscope Data")

# Create a canvas to embed the Matplotlib figure
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().grid(column=0, row=6, columnspan=2)

# Lists to store data for plotting
acc_x_data, acc_y_data, acc_z_data, acc_o_data, acc_ts_data = [], [], [], [], []
gyro_x_data, gyro_y_data, gyro_z_data, gyro_o_data, gyro_ts_data = [], [], [], [], []

# Function to update the GUI with new data
def update_gui(acceleroValues, gyroValues, acceleroTs, gyroTs):
    acc_x_var.set(f"x: {acceleroValues.x:.06f} m/s^2")
    acc_y_var.set(f"y: {acceleroValues.y:.06f} m/s^2")
    acc_z_var.set(f"z: {acceleroValues.z:.06f} m/s^2")
    acc_o_var.set(f"o: {math.sqrt(acceleroValues.x**2 + acceleroValues.y**2 + acceleroValues.z**2):.06f} m/s^2")
    acc_ts_var.set(f"Timestamp: {acceleroTs:.03f} ms")

    gyro_x_var.set(f"x: {gyroValues.x:.06f} rad/s")
    gyro_y_var.set(f"y: {gyroValues.y:.06f} rad/s")
    gyro_z_var.set(f"z: {gyroValues.z:.06f} rad/s")
    gyro_o_var.set(f"o: {math.sqrt(gyroValues.x**2 + gyroValues.y**2 + gyroValues.z**2):.06f} rad/s")
    gyro_ts_var.set(f"Timestamp: {gyroTs:.03f} ms")

    # Update data lists
    acc_x_data.append(acceleroValues.x)
    acc_y_data.append(acceleroValues.y)
    acc_z_data.append(acceleroValues.z)
    acc_o_data.append(math.sqrt(acceleroValues.x**2 + acceleroValues.y**2 + acceleroValues.z**2))
    acc_ts_data.append(acceleroTs)

    gyro_x_data.append(gyroValues.x)
    gyro_y_data.append(gyroValues.y)
    gyro_z_data.append(gyroValues.z)
    gyro_o_data.append(math.sqrt(gyroValues.x**2 + gyroValues.y**2 + gyroValues.z**2))
    gyro_ts_data.append(gyroTs)

    # Keep only the last 100 data points for plotting
    if len(acc_ts_data) > 100:
        acc_x_data.pop(0)
        acc_y_data.pop(0)
        acc_z_data.pop(0)
        acc_o_data.pop(0)
        acc_ts_data.pop(0)
        gyro_x_data.pop(0)
        gyro_y_data.pop(0)
        gyro_z_data.pop(0)
        gyro_o_data.pop(0)
        gyro_ts_data.pop(0)

    # Clear and redraw the plots
    ax_acc.clear()
    ax_gyro.clear()
    ax_acc.plot(acc_ts_data, acc_x_data, label='x')
    ax_acc.plot(acc_ts_data, acc_y_data, label='y')
    ax_acc.plot(acc_ts_data, acc_z_data, label='z')
    ax_acc.plot(acc_ts_data, acc_o_data, label='o')
    ax_acc.legend()
    ax_gyro.plot(gyro_ts_data, gyro_x_data, label='x')
    ax_gyro.plot(gyro_ts_data, gyro_y_data, label='y')
    ax_gyro.plot(gyro_ts_data, gyro_z_data, label='z')
    ax_gyro.plot(gyro_ts_data, gyro_o_data, label='o')
    ax_gyro.legend()
    canvas.draw()

# Pipeline is defined, now we can connect to the device
with dai.Device(pipeline) as device:
    def timeDeltaToMilliS(delta) -> float:
        return delta.total_seconds() * 1000

    # Output queue for imu bulk packets
    imuQueue = device.getOutputQueue(name="imu", maxSize=50, blocking=False)
    baseTs = None

    def process_data():
        global baseTs
        imuData = imuQueue.get()  # blocking call, will wait until new data arrives

        imuPackets = imuData.packets
        for imuPacket in imuPackets:
            acceleroValues = imuPacket.acceleroMeter
            gyroValues = imuPacket.gyroscope

            acceleroTs = acceleroValues.getTimestampDevice()
            gyroTs = gyroValues.getTimestampDevice()
            if baseTs is None:
                baseTs = acceleroTs if acceleroTs < gyroTs else gyroTs
            acceleroTs = timeDeltaToMilliS(acceleroTs - baseTs)
            gyroTs = timeDeltaToMilliS(gyroTs - baseTs)

            update_gui(acceleroValues, gyroValues, acceleroTs, gyroTs)

        root.after(10, process_data)  # Schedule next data fetch

    root.after(10, process_data)  # Start data fetch loop
    root.mainloop()
