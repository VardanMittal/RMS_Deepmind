import time
import os
import numpy as np
from scipy.signal import welch
import requests

LLAMA_CPP_URL = "http://localhost:8080/v1/chat/completions"

class AICoachWorker:
    def __init__(self, plotter_ref):
        self.plotter = plotter_ref
        self.fs = 200  
        self.last_alert_time = 0
        self.cooldown_period = 6  # Optimized pacing window for active exercise repetitions

    def analyze_and_coach(self):
        print("[System Core] Advanced Edge-ML JSON Schema Intelligence Layer Engaged.")
        while self.plotter.is_running:
            time.sleep(1)  # High-frequency analysis frame evaluation
            
            current_time = time.time()
            if current_time - self.last_alert_time < self.cooldown_period:
                continue

            with self.plotter.data_lock:
                emg_window = np.copy(self.plotter.data_emg[-1000:]) 
                ax_window = np.copy(self.plotter.data_ax[-150:])
                pitch_window = np.copy(self.plotter.data_pitch[-150:])
                
            if len(emg_window) < 256 or np.all(emg_window == 0):
                continue

            # --- ADVANCED BIOMECHANICAL FEATURE EXTRACTION ---
            rms = np.sqrt(np.mean(np.square(emg_window)))
            freqs, psd = welch(emg_window, fs=self.fs, nperseg=256)
            cumulative_psd = np.cumsum(psd)
            median_freq = freqs[np.where(cumulative_psd >= cumulative_psd[-1] / 2.0)[0][0]] if cumulative_psd[-1] > 0 else 0
            
            pitch_delta = np.max(pitch_window) - np.min(pitch_window)
            acceleration_jitter = np.std(ax_window)  # Tracks raw micro-tremor and spasticity

            is_anomaly = False
            metrics_summary = []

            # Clinical Decision Matrix
            if 0 < median_freq < 44.0 and rms > 1200:
                is_anomaly = True
                metrics_summary.append(f"Severe Neuromuscular Exhaustion (Spectral Shift to {median_freq:.1f}Hz)")
            
            if acceleration_jitter > 0.55:
                is_anomaly = True
                metrics_summary.append(f"High Volitional Jitter / Tremor Detected ({acceleration_jitter:.2f}g)")
                
            if pitch_delta < 15.0 and rms > 500:
                is_anomaly = True
                metrics_summary.append(f"Kinematic Range-of-Motion Compression ({pitch_delta:.1f} degrees change)")

            if is_anomaly:
                # Optimized prompt structure specifically for smaller edge models
                payload = {
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are an advanced clinical edge intelligence system integrated into an active knee orthosis.\n"
                                "Analyze the provided biomechanical anomalies and respond ONLY with a single JSON object. "
                                "Do not include any markdown formatting, backticks, or text before/after the JSON.\n\n"
                                "Expected format:\n"
                                "{\n"
                                '  "urgency": "CRITICAL" or "MODERATE",\n'
                                '  "rationale": "Brief scientific summary of the issue.",\n'
                                '  "voice_cue": "One direct, short, actionable sentence coaching their movement form."\n'
                                "}"
                            )
                        },
                        {
                            "role": "user",
                            "content": f"Biomechanical Telemetry Anomalies Logged: {', '.join(metrics_summary)}. Velocity Jitter: {acceleration_jitter:.2f}g."
                        }
                    ],
                    "temperature": 0.3,
                    "max_tokens": 80
                }
                
                try:
                    res = requests.post(LLAMA_CPP_URL, json=payload, timeout=3)
                    if res.status_code == 200:
                        import json
                        res_data = res.json()
                        raw_content = res_data['choices'][0]['message']['content'].strip()
                        
                        # Strip away any markdown formatting backticks if the model accidentally included them
                        if raw_content.startswith("```"):
                            raw_content = raw_content.strip("```").strip("json").strip()
                        
                        voice_cue = ""
                        urgency = "MODERATE"
                        rationale = "Biomechanical boundary threshold alert."
                        
                        # --- MULTI-MODE ROBUST PARSER ---
                        try:
                            parsed_analysis = json.loads(raw_content)
                            urgency = parsed_analysis.get("urgency", "MODERATE")
                            rationale = parsed_analysis.get("rationale", "Threshold breach detected.")
                            voice_cue = parsed_analysis.get("voice_cue", "")
                        except json.JSONDecodeError:
                            # Fallback: If it completely failed JSON formatting and gave raw text, use it directly!
                            voice_cue = raw_content.replace('"', '').replace("'", "")
                            rationale = "Direct text stream bypass recovery engaged."

                        if voice_cue:
                            print(f"\n🧬 [Edge Matrix] Level: {urgency} | Diagnosis: {rationale}")
                            print(f"🎙️ Voice Cue Pipeline -> {voice_cue}")
                            
                            # Pipe cleanly into your local human neural TTS pipeline
                            project_root = os.path.expanduser("~/Desktop/Workspace/RMS_DEEPMIND")
                            piper_bin = os.path.join(project_root, "piper", "piper")
                            model_path = os.path.join(project_root, "voices", "en_US-ryan-medium.onnx")
                            
                            shell_cmd = (
                                f'echo "{voice_cue}" | '
                                f'"{piper_bin}" --model "{model_path}" --output-raw | '
                                f'aplay -r 22050 -f S16_LE -t raw 2>/dev/null &'
                            )
                            os.system(shell_cmd)
                            self.last_alert_time = time.time()
                        else:
                            # Final absolute safety fallback so your voice engine NEVER stays silent during a live demo
                            print("⚠️ Fallback triggered: Model output empty text. Deploying static cue...")
                            static_cue = "Form degradation detected. Adjust your posture and maintain control."
                            print(f"🎙️ Voice Cue Pipeline (Static Fallback) -> {static_cue}")
                            
                            # Dynamically locate the project workspace folder relative to this file
                            project_root = os.path.dirname(os.path.abspath(__file__))
                            piper_bin = os.path.join(project_root, "piper", "piper")
                            model_path = os.path.join(project_root, "voices", "en_US-ryan-medium.onnx")
                            
                            shell_cmd = (
                                f'echo "{static_cue}" | '
                                f'"{piper_bin}" --model "{model_path}" --output-raw | '
                                f'aplay -r 22050 -f S16_LE -t raw 2>/dev/null &'
                            )
                            os.system(shell_cmd)
                            self.last_alert_time = time.time()
                            
                except Exception as e:
                    print(f"[Core Error] Pipeline processing failure: {e}")