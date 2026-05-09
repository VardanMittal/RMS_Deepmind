import asyncio
import time
import threading
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from biokinesis_sdk.receiver.ble_receiver import SensorRTReceiver
from biokinesis_sdk.export import ExportManager
from biokinesis_sdk.emg.features import rms_envelope
from biokinesis_sdk.imu.kinematics import quaternions_to_euler_batch

is_running = True

def start_ble_receiver(receiver):
    """Runs the asyncio event loop for BLE receiving in a separate thread."""
    async def run_receiver():
        receive_task = asyncio.create_task(receiver.connect_and_run())
        
        while is_running:
            await asyncio.sleep(0.1)
            if receive_task.done():
                break
        
        # Cancel the task and wait for it to clean up gracefully
        receive_task.cancel()
        try:
            await receive_task
        except asyncio.CancelledError:
            pass

    asyncio.run(run_receiver())

def main():
    global is_running
    
    print("=" * 60)
    print(" Biokinesis-SDK Live Visualizer")
    print(" Close the plot window to stop recording and export data.")
    print("=" * 60)
    
    receiver = SensorRTReceiver()

    ble_thread = threading.Thread(target=start_ble_receiver, args=(receiver,), daemon=True)
    ble_thread.start()
    
    print("\nWaiting for BLE connection... Make sure the sensor is ON.")
    
    fig, (ax_emg, ax_imu) = plt.subplots(2, 1, figsize=(10, 8))
    fig.canvas.manager.set_window_title('Live Sensor Data')

    line_emg, = ax_emg.plot([], [], lw=1.5, color='#1f77b4', label='EMG (Filtered)')
    line_rms, = ax_emg.plot([], [], lw=2, color='#ff7f0e', label='EMG (RMS)')
    ax_emg.set_title('Live Muscle Activity (EMG)')
    ax_emg.set_ylabel('Amplitude (µV)')
    ax_emg.legend(loc='upper right')
    
    # Setup IMU Plot (Euler Angles)
    lines_imu = []
    colors = ['#d62728', '#2ca02c', '#9467bd']
    labels = ['Roll (X)', 'Pitch (Y)', 'Yaw (Z)']
    for i in range(3):
        line, = ax_imu.plot([], [], lw=1.5, color=colors[i], label=labels[i])
        lines_imu.append(line)
        
    ax_imu.set_title('Live Motion (Euler Angles)')
    ax_imu.set_ylabel('Angle (degrees)')
    ax_imu.set_xlabel('Recent Samples')
    ax_imu.legend(loc='upper right')
    
    def init():
        ax_emg.set_xlim(0, 1000)
        ax_emg.set_ylim(-500, 500)
        ax_imu.set_xlim(0, 100)
        ax_imu.set_ylim(-180, 180)
        return [line_emg, line_rms] + lines_imu

    def update(frame):
        """Update function called repeatedly by FuncAnimation."""
        if not is_running:
            return [line_emg, line_rms] + lines_imu
            
        # Safely collect data from the orchestrator
        data = receiver.orch.collect_all_data()
        
        if "1" in data and "emg_filtered" in data["1"]:
            emg_data = data["1"]["emg_filtered"]
            imu_quats = data["1"].get("imu_quaternions", [])
            
            # --- Update EMG ---
            window_size_emg = 1000  # Show last N samples
            if len(emg_data) > 0:
                y = emg_data[-window_size_emg:]
                x = np.arange(len(y))
                
                line_emg.set_data(x, y)
                
                # Dynamically scale the Y axis for EMG
                max_val = max(abs(np.max(y)), abs(np.min(y))) + 50
                ax_emg.set_ylim(-max_val, max_val)
                ax_emg.set_xlim(0, len(y))
                
                # Calculate sliding RMS on the visible window
                if len(y) >= 50:
                    # Using a 50-sample window for the RMS envelope visual
                    rms_val = rms_envelope(y, window_samples=50)
                    step = 25
                    x_rms = np.arange(len(rms_val)) * step + (50 // 2)
                    line_rms.set_data(x_rms, rms_val)
                else:
                    line_rms.set_data([], [])
            
            # --- Update IMU ---
            window_size_imu = 100  # Show last N samples
            if len(imu_quats) > 0:
                q_array = np.array(imu_quats[-window_size_imu:])
                if q_array.ndim == 2 and q_array.shape[1] == 4:
                    euler_rads = quaternions_to_euler_batch(q_array)
                    y_imu_all = np.degrees(euler_rads)
                    x_imu = np.arange(len(y_imu_all))
                    
                    ax_imu.set_xlim(0, len(x_imu))
                    
                    for i in range(3): 
                        if y_imu_all.shape[1] > i:
                            lines_imu[i].set_data(x_imu, y_imu_all[:, i])
                    
        return [line_emg, line_rms] + lines_imu
    
    def on_close(event):
        """Triggered when the user closes the plot window."""
        global is_running
        print("\nPlot window closed. Stopping receiver...")
        is_running = False

    # Register the close event
    fig.canvas.mpl_connect('close_event', on_close)
    
    # Start the matplotlib animation loop
    ani = FuncAnimation(fig, update, frames=None, init_func=init, blit=True, interval=50, cache_frame_data=False)
    
    plt.tight_layout()
    # This call blocks the main thread until the window is closed
    plt.show()

    # --- Clean up and Export ---
    # Wait maximum 2 seconds for the BLE thread to clean up
    if ble_thread.is_alive():
        ble_thread.join(timeout=2.0)
    
    print("\n=== Exporting Recorded Session ===")
    final_data = receiver.orch.collect_all_data()
    
    if "1" not in final_data or "emg_filtered" not in final_data["1"] or len(final_data["1"]["emg_filtered"]) == 0:
         print("No data was collected during the session. Exiting.")
         return

    output_dir = "./Data/"
    formats = ["csv"]
    
    print(f"Exporting files to '{output_dir}/' in formats: {formats}")
    result = ExportManager.export_all(
        final_data, 
        output_dir, 
        prefix="live_session",
        formats=formats,
        model_file="gait2392_simbody.osim",
        imu_to_body_map={"Xiao_Arm": "radius_r"},  
        mvc_values={"Xiao_Arm": 1000.0},         
        emg_to_muscle_map={"Xiao_Arm": ["biceps_brachii"]}
    )
    
    print("\nExport successful!")
    print(f"CSV Check: {result.get('csv', [])}")
    print("Done. Goodbye!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nScript interrupted by user. Stopping...")
        is_running = False
