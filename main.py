"""
main.py — BioSense NeuroSync Dashboard
Real-time BLE data ingestion + pyqtgraph visualisation + edge AI coaching.
"""

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

# ── BLE / Packet constants ────────────────────────────────────────────────────
PACKET_FORMAT    = "<HII9f10HBI"
PACKET_SIZE      = struct.calcsize(PACKET_FORMAT)
UART_TX_CHAR_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"
UART_SERVICE_UUID = "6e400001-b5a3-f393-e0a9-e50e24dcca9e"

# Device filter — matches XIAO nRF52840 or any "Sensor" device
DEVICE_FILTER = lambda d, ad: "XIAO" in (d.name or "") or "Sensor" in (d.name or "")


class RealTimePlotter:
    def __init__(self):
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

        pg.setConfigOption('background', 'k')
        pg.setConfigOption('foreground', 'w')

        self.win = pg.GraphicsLayoutWidget(show=True, title="BioSense — Edge AI Knee Rehab")
        self.win.resize(1280, 960)

        # ── Plots ──────────────────────────────────────────────────────────────
        self.p_emg = self.win.addPlot(title="EMG Signal  [µV]")
        self.p_emg.setLabel('left', 'Amplitude (µV)')
        self.p_emg.setLabel('bottom', 'Samples')

        self.win.nextRow()

        self.p_angles = self.win.addPlot(title="Joint Orientation  [°]")
        self.p_angles.addLegend()
        self.p_angles.setLabel('left', 'Degrees')

        self.win.nextRow()

        self.p_accel = self.win.addPlot(title="Accelerometer  [g]")
        self.p_accel.setLabel('left', 'g')

        self.win.nextRow()

        # ── Metrics overlay (rep count + last alert) ───────────────────────────
        self.p_info = self.win.addPlot(title="Session Metrics")
        self.p_info.hideAxis('left')
        self.p_info.hideAxis('bottom')
        self._rep_label = pg.TextItem(text="Reps: 0", color='g', anchor=(0, 0))
        self._alert_label = pg.TextItem(text="Last alert: —", color='y', anchor=(0, 0))
        self.p_info.addItem(self._rep_label)
        self.p_info.addItem(self._alert_label)
        self._rep_label.setPos(0, 1)
        self._alert_label.setPos(0, 0)

        # ── Curves ────────────────────────────────────────────────────────────
        self.curve_emg   = self.p_emg.plot(pen=pg.mkPen('y', width=1))
        self.curve_roll  = self.p_angles.plot(pen=pg.mkPen('r', width=1.5), name='Roll')
        self.curve_pitch = self.p_angles.plot(pen=pg.mkPen('g', width=1.5), name='Pitch')
        self.curve_yaw   = self.p_angles.plot(pen=pg.mkPen('b', width=1.5), name='Yaw')
        self.curve_ax    = self.p_accel.plot(pen=pg.mkPen('r', width=1))

        # ── Data buffers ───────────────────────────────────────────────────────
        self.max_samples = 200
        self.data_emg    = np.zeros(2000)
        self.data_roll   = np.zeros(self.max_samples)
        self.data_pitch  = np.zeros(self.max_samples)
        self.data_yaw    = np.zeros(self.max_samples)
        self.data_ax     = np.zeros(self.max_samples)

        self.buffer      = bytearray()
        self.is_running  = True
        self.data_lock   = threading.Lock()

        # Reference to coach (set after construction)
        self._coach: AICoachWorker | None = None

        # ── UI update timer ────────────────────────────────────────────────────
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._update_plots)
        self.timer.start(30)  # ~33 fps

    def set_coach(self, coach: AICoachWorker):
        self._coach = coach

    # ── Signal processing ─────────────────────────────────────────────────────

    def compute_angles(self, ax, ay, az, mx, my, mz):
        roll  = atan2(ay, az)
        pitch = atan2(-ax, np.sqrt(ay * ay + az * az))
        mag_x = mx * np.cos(pitch) + mz * np.sin(pitch)
        mag_y = (mx * np.sin(roll) * np.sin(pitch)
                 + my * np.cos(roll)
                 - mz * np.sin(roll) * np.cos(pitch))
        yaw = atan2(-mag_y, mag_x)
        return degrees(roll), degrees(pitch), degrees(yaw)

    # ── BLE data handler ──────────────────────────────────────────────────────

    def handle_ble_data(self, sender, data):
        self.buffer.extend(data)

        while len(self.buffer) >= PACKET_SIZE:
            # Sync on magic header 0xBEEF
            if struct.unpack("<H", self.buffer[:2])[0] != 0xBEEF:
                self.buffer.pop(0)
                continue

            packet   = self.buffer[:PACKET_SIZE]
            del self.buffer[:PACKET_SIZE]
            unpacked = struct.unpack(PACKET_FORMAT, packet)

            ax, ay, az = unpacked[3:6]
            mx, my, mz = unpacked[9:12]

            r, p, y = self.compute_angles(ax, ay, az, mx, my, mz)

            with self.data_lock:
                self.data_roll  = np.roll(self.data_roll, -1);  self.data_roll[-1]  = r
                self.data_pitch = np.roll(self.data_pitch, -1); self.data_pitch[-1] = p
                self.data_yaw   = np.roll(self.data_yaw, -1);   self.data_yaw[-1]   = y
                self.data_ax    = np.roll(self.data_ax, -1);    self.data_ax[-1]    = ax

                emg_samples      = unpacked[12:22]
                self.data_emg    = np.roll(self.data_emg, -10)
                self.data_emg[-10:] = emg_samples

    # ── Plot refresh ──────────────────────────────────────────────────────────

    def _update_plots(self):
        with self.data_lock:
            self.curve_emg.setData(self.data_emg)
            self.curve_roll.setData(self.data_roll)
            self.curve_pitch.setData(self.data_pitch)
            self.curve_yaw.setData(self.data_yaw)
            self.curve_ax.setData(self.data_ax)

        # Update metrics overlay from coach state
        if self._coach:
            self._rep_label.setText(f"Reps: {self._coach.session.rep_count}")
            if self._coach.session.alert_count > 0:
                last = self._coach.session.events[-1]
                self._alert_label.setText(
                    f"Alerts: {self._coach.session.alert_count}  |  "
                    f"Last: {last.get('urgency', '')} @ rep {self._coach.session.rep_count}"
                )


# ── BLE async runner ──────────────────────────────────────────────────────────

async def run_ble(plotter: RealTimePlotter):
    print("[BioSense BLE] Scanning for Biokinesis wearable sensor...")
    device = await BleakScanner.find_device_by_filter(DEVICE_FILTER)

    if not device:
        print("[BioSense BLE] Device not found. Check Bluetooth and sensor power.")
        return

    print(f"[BioSense BLE] Connected → {device.name}  ({device.address})")

    async with BleakClient(device) as client:
        await client.start_notify(UART_TX_CHAR_UUID, plotter.handle_ble_data)
        while plotter.is_running:
            await asyncio.sleep(0.1)

    print("[BioSense BLE] Disconnected.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    plotter = RealTimePlotter()

    # Thread 1 — BLE ingestion
    ble_thread = threading.Thread(
        target=lambda: asyncio.run(run_ble(plotter)), daemon=True
    )
    ble_thread.start()

    # Thread 2 — AI coaching loop
    coach = AICoachWorker(plotter)
    plotter.set_coach(coach)
    coach_thread = threading.Thread(target=coach.analyze_and_coach, daemon=True)
    coach_thread.start()

    print("[BioSense] Dashboard running. Close the window to end session and generate report.")

    try:
        exit_code = plotter.app.exec()
    finally:
        plotter.is_running = False
        print("\n[BioSense] Session ended. Generating report...")
        # Give threads a moment to finish their current cycle
        time.sleep(1.5)
        coach.generate_session_report()
        sys.exit(exit_code)