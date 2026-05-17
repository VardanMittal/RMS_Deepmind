# BioSense — Edge AI Knee Rehabilitation Coach

> **Gemma 4 Good Hackathon Submission** · Health & Sciences Track  
> Real-time biomechanical coaching for post-surgical knee rehab patients — running fully offline on a custom wearable sensor.

---

## The Problem

After knee surgery (ACL reconstruction, total knee replacement), patients do physiotherapy exercises at home — alone, unsupervised, for weeks or months. Without a clinician watching:

- **Bad form goes uncorrected** — compensatory movement patterns develop and slow recovery
- **Fatigue goes unnoticed** — patients push through neuromuscular exhaustion and risk re-injury
- **Progress is invisible** — no objective data reaches the physiotherapist between appointments

Supervised physiotherapy is expensive, clinic-dependent, and inaccessible to millions of patients — especially in low-resource settings, rural areas, or regions with poor connectivity.

**BioSense puts a clinical-grade AI coach on the patient's body, running entirely offline.**

---

## What It Does

BioSense is a wearable biosensor system paired with an on-device Gemma 4 AI coach. During a rehabilitation exercise session:

1. **The sensor streams live data** — EMG (muscle activation), accelerometer, gyroscope, and magnetometer — over BLE at 200 Hz
2. **The signal processing pipeline** extracts biomechanical features in real time: muscle fatigue indicators, range-of-motion, tremor/jitter, and rep count
3. **Gemma 4 (Edge 2B, running locally)** analyses detected anomalies via native function calling and decides whether to issue a voice coaching cue, flag the moment for clinician review, or both
4. **Piper TTS** speaks the coaching cue aloud to the patient in real time
5. **At session end**, Gemma 4 generates a structured rehabilitation report — performance score, fatigue progression, ROM trend, and recommendations — which the patient can share with their physiotherapist

No internet. No cloud. No data leaves the device.

---

## Demo

[![BioSense Demo Video](https://img.shields.io/badge/Watch%20Demo-YouTube-red)](YOUR_VIDEO_LINK_HERE)

**Recommended exercise:** Seated knee extension (anterior thigh placement)  
**What to watch for:**
- Rep counter incrementing live from pitch zero-crossings
- EMG median frequency shift triggering a fatigue alert around rep 7–8
- Gemma 4 speaking a coaching cue aloud via on-device TTS
- End-of-session report generated automatically on window close

---

## Hardware

| Component | Role |
|---|---|
| **nRF52840** (Seeed XIAO) | BLE microcontroller — streams packets at 200 Hz |
| **AD8237** | Instrumentation amplifier — EMG front-end signal acquisition |
| **LSM6DSV16X** | 6-axis IMU — accelerometer + gyroscope |
| **Magnetometer** | Yaw/heading fusion for full ENU orientation |

Sensor placement: **anterior thigh (mid-vastus lateralis)** for knee extension monitoring.

---

## Software Architecture

```
Wearable Sensor (BLE 200Hz)
        │
        ▼
  handle_ble_data()          — packet sync, struct unpack, angle computation
        │
        ▼
  Shared Data Buffers        — thread-safe numpy ring buffers (EMG, pitch, accel)
   ┌────┴──────────────────────────────────┐
   │                                       │
   ▼                                       ▼
pyqtgraph UI (30fps)           AICoachWorker thread (1Hz analysis)
   - EMG waveform                  │
   - Orientation (R/P/Y)           ├── _extract_features()
   - Accelerometer                 │     RMS, median frequency, HF/LF ratio,
   - Rep count overlay             │     pitch delta, acceleration jitter
                                   │
                                   ├── _detect_rep()
                                   │     pitch zero-crossing counter
                                   │
                                   ├── _detect_anomalies()
                                   │     clinical decision matrix
                                   │
                                   └── _call_coaching_llm()
                                         Gemma 4 E2B (llama.cpp, local)
                                         Native function calling
                                              │
                                         ┌───┴──────────────────────┐
                                         ▼                          ▼
                                  trigger_coaching_alert   flag_for_clinician_review
                                         │
                                         ▼
                                    Piper TTS → aplay (spoken aloud)
                                         │
                                         ▼
                                    SessionLogger (accumulates events)
                                         │
                                         ▼ (on session end)
                                  generate_session_report()
                                    Gemma 4 structured report
```

---

## Gemma 4 Integration

This project uses **Gemma 4 Edge 2B** (`ggml-org/gemma-4-E2B-it-GGUF`) served locally via llama.cpp. Gemma 4's **native function calling** is used throughout — not prompt engineering or JSON hacks.

### Function 1 — `trigger_coaching_alert`
Called during active exercise when a biomechanical anomaly is detected. Returns `urgency` (CRITICAL / MODERATE), a `voice_cue` spoken aloud to the patient, and a `rationale` logged for the session record.

### Function 2 — `flag_for_clinician_review`
Called when an anomaly is clinically significant but not immediately dangerous. Logs a structured clinical note for the physiotherapist to review asynchronously.

### Function 3 — `generate_session_report`
Called at session end. Receives the full session event log and returns a structured report: performance score, fatigue progression narrative, ROM trend, 3 next-session recommendations, and clinical flags.

**Why function calling matters here:** The model doesn't just generate text — it makes a clinical decision (alert now vs. flag for later vs. both) and returns structured data that directly drives downstream actions (TTS pipeline, session log, report). This is Gemma 4 doing real agentic work, not just answering a question.

---

## Biomechanical Features Extracted

| Feature | Method | Clinical Meaning |
|---|---|---|
| **RMS** | √(mean(EMG²)) | Overall muscle activation intensity |
| **Median Frequency** | Welch PSD, 50% cumulative power | Spectral shift downward = neuromuscular fatigue |
| **HF/LF Power Ratio** | PSD energy >100Hz vs 20–100Hz | Early fatigue: HF components collapse first |
| **Pitch Delta** | max − min of pitch window | Range of motion across the rep |
| **Acceleration Jitter** | std(ax) | Tremor, spasticity, or instability |
| **Rep Count** | Pitch zero-crossing detector | Exercise volume tracking |

---

## Why Edge / Offline

- Patient data never leaves the device — clinically relevant for privacy regulations
- Works in rural clinics, homes, and regions with no reliable internet
- Gemma 4 E2B fits comfortably on a laptop GPU — no server infrastructure required
- Latency from anomaly detection to spoken cue: **< 3 seconds** on a mid-range laptop GPU

---

## Setup & Run

### Prerequisites
```bash
pip install numpy scipy pyqtgraph bleak requests
# Piper TTS binary + en_US-ryan-medium.onnx in ./piper/ and ./voices/
```

### Start Gemma 4 (llama.cpp)
```bash
sudo systemctl restart bluetooth 

./Brain/llama.cpp/build/bin/llama-server \
  -hf ggml-org/gemma-4-E2B-it-GGUF \
  -c 2048 \
  --n-gpu-layers 99 \
  --port 8080
```

### Run BioSense
```bash
python main.py
```

Power on the wearable sensor. The dashboard connects automatically via BLE, begins streaming, and the AI coach activates. Close the window to end the session and generate the report.

---

## Project Structure

```
biosense/
├── main.py              # BLE ingestion, pyqtgraph dashboard, entry point
├── AI_Coach.py          # Feature extraction, anomaly detection, Gemma 4 coaching
├── piper/
│   └── piper            # Piper TTS binary
├── voices/
│   └── en_US-ryan-medium.onnx
└── Brain/
    └── llama.cpp/       # llama.cpp build
```

---

## Impact

BioSense addresses a real gap in post-surgical care:

- **~100 million** knee surgeries performed globally each year
- Physiotherapy non-adherence rates are 30–65% — patients skip or do exercises incorrectly at home
- A voice coaching system that works offline can reach patients in any setting — not just those with smartphones, internet, or clinic access

The same architecture generalises to shoulder rehab, gait retraining, and sports injury prevention with sensor repositioning and prompt adjustment — no hardware changes required.

---

## Built With

- [Gemma 4 E2B](https://huggingface.co/ggml-org/gemma-4-E2B-it-GGUF) — Google DeepMind
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — local inference server
- [Piper TTS](https://github.com/rhasspy/piper) — on-device neural text-to-speech
- [Bleak](https://github.com/hbldh/bleak) — BLE communication
- [pyqtgraph](https://www.pyqtgraph.org/) — real-time signal visualisation
- Custom nRF52840 + AD8237 + LSM6DSV16X wearable PCB

--