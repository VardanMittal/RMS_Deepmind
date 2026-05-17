import asyncio
import struct
import sys
import threading
import time
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from bleak import BleakClient, BleakScanner
from math import atan2, degrees
from AI_Coach import AICoachWorker

# Constants (Matches original sensor configuration)
PACKET_FORMAT = "<HII9f10HBI"
PACKET_SIZE = struct.calcsize(PACKET_FORMAT)
UART_TX_CHAR_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"
UART_SERVICE_UUID = "6e400001-b5a3-f393-e0a9-e50e24dcca9e"

class RealTimePlotter:
    def __init__(self):
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        
        pg.setConfigOption('background', 'k') 
        pg.setConfigOption('foreground', 'w')

        self.win = pg.GraphicsLayoutWidget(show=True, title="NeuroSync: Edge AI Knee Rehab")
        self.win.resize(1200, 900)
        
        # Plot Setup
        self.p_emg = self.win.addPlot(title="EMG Signal (Neuraline Wearable)")
        self.win.nextRow()
        self.p_angles = self.win.addPlot(title="ENU Orientation (Stryde IMU)")
        self.p_angles.addLegend()
        self.p_angles.setLabel('left', 'Degrees')
        self.win.nextRow()
        self.p_accel = self.win.addPlot(title="Accelerometer (g)")
        
        # Curves
        self.curve_emg = self.p_emg.plot(pen='y')
        self.curve_roll = self.p_angles.plot(pen='r', name='Roll (X)')
        self.curve_pitch = self.p_angles.plot(pen='g', name='Pitch (Y)')
        self.curve_yaw = self.p_angles.plot(pen='b', name='Yaw (Z)')
        self.curve_ax = self.p_accel.plot(pen='r')

        # Data Buffers
        self.max_samples = 200
        self.data_emg = np.zeros(2000)
        self.data_roll = np.zeros(self.max_samples)
        self.data_pitch = np.zeros(self.max_samples)
        self.data_yaw = np.zeros(self.max_samples)
        self.data_ax = np.zeros(self.max_samples)

        self.buffer = bytearray()
        self.is_running = True
        
        # Thread Safety Lock for accessing data across graph and AI loops
        self.data_lock = threading.Lock()

        # Timer for UI Redraws
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(30)

    def compute_angles(self, ax, ay, az, mx, my, mz):
        roll = atan2(ay, az)
        pitch = atan2(-ax, np.sqrt(ay*ay + az*az))
        
        mag_x = mx * np.cos(pitch) + mz * np.sin(pitch)
        mag_y = mx * np.sin(roll) * np.sin(pitch) + my * np.cos(roll) - mz * np.sin(roll) * np.cos(pitch)
        yaw = atan2(-mag_y, mag_x)

        return degrees(roll), degrees(pitch), degrees(yaw)

    def update_plots(self):
        with self.data_lock:
            self.curve_emg.setData(self.data_emg)
            self.curve_roll.setData(self.data_roll)
            self.curve_pitch.setData(self.data_pitch)
            self.curve_yaw.setData(self.data_yaw)
            self.curve_ax.setData(self.data_ax)

    def handle_ble_data(self, sender, data):
        self.buffer.extend(data)
        
        while len(self.buffer) >= PACKET_SIZE:
            if struct.unpack("<H", self.buffer[:2])[0] != 0xbeef:
                self.buffer.pop(0)
                continue
            
            packet = self.buffer[:PACKET_SIZE]
            del self.buffer[:PACKET_SIZE]
            unpacked = struct.unpack(PACKET_FORMAT, packet)
            
            ax, ay, az = unpacked[3:6]
            gx, gy, gz = unpacked[6:9]
            mx, my, mz = unpacked[9:12]
            
            r, p, y = self.compute_angles(ax, ay, az, mx, my, mz)

            with self.data_lock:
                self.data_roll = np.roll(self.data_roll, -1); self.data_roll[-1] = r
                self.data_pitch = np.roll(self.data_pitch, -1); self.data_pitch[-1] = p
                self.data_yaw = np.roll(self.data_yaw, -1); self.data_yaw[-1] = y
                self.data_ax = np.roll(self.data_ax, -1); self.data_ax[-1] = ax
                
                emg_samples = unpacked[12:22]
                self.data_emg = np.roll(self.data_emg, -10)
                self.data_emg[-10:] = emg_samples

async def run_ble(plotter):
    print("Searching for Biokinesis Wearable Sensor via BLE...")
    device = await BleakScanner.find_device_by_filter(
        lambda d, ad: "XIAO" in (d.name or "") or "Sensor" in (d.name or "")
    )
    
    if not device:
        print("Hardware tracking error: Device could not be located via Bluetooth scan.")
        return

    async with BleakClient(device) as client:
        print(f"Hardware Connected Successfully! Ingesting data stream from device: {device.name}")
        await client.start_notify(UART_TX_CHAR_UUID, plotter.handle_ble_data)
        while plotter.is_running:
            await asyncio.sleep(0.1)

if __name__ == "__main__":
    plotter = RealTimePlotter()
    
    # Core 1: Bluetooth Stream Ingestion Worker
    ble_thread = threading.Thread(target=lambda: asyncio.run(run_ble(plotter)), daemon=True)
    ble_thread.start()
    
    # Core 2: Asynchronous Offline Gemma 4 Pipeline 
    coach_worker = AICoachWorker(plotter)
    coach_thread = threading.Thread(target=coach_worker.analyze_and_coach, daemon=True)
    coach_thread.start()
    
    try:
        sys.exit(plotter.app.exec())
    finally:
        plotter.is_running = False