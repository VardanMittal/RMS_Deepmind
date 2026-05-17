"""
AI_Coach.py — BioSense Edge Rehab Intelligence
Real-time biomechanical coaching using llama.cpp (OpenAI-compatible server).
Uses native function calling — no JSON prompt hacks.
"""

import time
import os
import json
import threading
import numpy as np
from datetime import datetime
from scipy.signal import welch
import requests

# ── Configuration ────────────────────────────────────────────────────────────
LLAMA_CPP_URL   = "http://localhost:8080/v1/chat/completions"
COACHING_MODEL  = "ggml-org/gemma-4-E2B-it-GGUF"          
COOLDOWN_SEC    = 6                  
MIN_EMG_SAMPLES = 256

# ── Function Calling Tool Definitions ────────────────────────────────────────
# llama.cpp supports OpenAI-compatible tool calling — no JSON hacks needed.

COACHING_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "trigger_coaching_alert",
            "description": (
                "Issue a real-time audio coaching cue to the patient based on "
                "detected biomechanical anomalies during active exercise."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "urgency": {
                        "type": "string",
                        "enum": ["CRITICAL", "MODERATE"],
                        "description": "CRITICAL = stop/reduce load immediately. MODERATE = form correction."
                    },
                    "voice_cue": {
                        "type": "string",
                        "description": "One short, direct coaching sentence spoken aloud to the patient."
                    },
                    "rationale": {
                        "type": "string",
                        "description": "Brief clinical reason for this alert (logged, not spoken)."
                    }
                },
                "required": ["urgency", "voice_cue", "rationale"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "flag_for_clinician_review",
            "description": (
                "Flag a session moment for async physiotherapist review "
                "when an anomaly is clinically significant but not immediately dangerous."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "severity": {
                        "type": "string",
                        "enum": ["HIGH", "MEDIUM"]
                    },
                    "clinical_note": {
                        "type": "string",
                        "description": "Objective clinical observation for the physiotherapist."
                    }
                },
                "required": ["severity", "clinical_note"]
            }
        }
    }
]

REPORT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "generate_session_report",
            "description": "Generate a structured end-of-session physiotherapy report.",
            "parameters": {
                "type": "object",
                "properties": {
                    "performance_score": {
                        "type": "integer",
                        "description": "Overall session quality score from 0–100."
                    },
                    "fatigue_progression": {
                        "type": "string",
                        "description": "Summary of how fatigue evolved across the session."
                    },
                    "rom_trend": {
                        "type": "string",
                        "description": "Range-of-motion trend observation across reps."
                    },
                    "recommendations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "3 specific recommendations for the next session."
                    },
                    "clinical_flags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Any moments that require clinician attention."
                    },
                    "summary": {
                        "type": "string",
                        "description": "2–3 sentence plain-language summary for the patient."
                    }
                },
                "required": [
                    "performance_score", "fatigue_progression", "rom_trend",
                    "recommendations", "clinical_flags", "summary"
                ]
            }
        }
    }
]


class SessionLogger:
    """Accumulates per-rep biomechanical events for end-of-session analysis."""

    def __init__(self):
        self.start_time = datetime.now().isoformat()
        self.events: list[dict] = []
        self.rep_count = 0
        self.alert_count = 0

    def log_event(self, event_type: str, metrics: dict, urgency: str = ""):
        self.events.append({
            "t":        round(time.time(), 2),
            "type":     event_type,
            "metrics":  metrics,
            "urgency":  urgency
        })
        if event_type == "alert":
            self.alert_count += 1

    def log_rep(self):
        self.rep_count += 1

    def to_summary_dict(self) -> dict:
        return {
            "start_time":   self.start_time,
            "duration_sec": round(time.time() - datetime.fromisoformat(self.start_time).timestamp(), 1)
                            if self.events else 0,
            "total_reps":   self.rep_count,
            "total_alerts": self.alert_count,
            "events":       self.events[-40:]  # send last 40 events max to stay within context
        }


class AICoachWorker:
    def __init__(self, plotter_ref):
        self.plotter  = plotter_ref
        self.fs       = 200
        self.last_alert_time = 0
        self.cooldown = COOLDOWN_SEC

        self.session  = SessionLogger()
        self._lock    = threading.Lock()

        # Rep detection state
        self._prev_pitch_sign = 0
        self._rep_threshold   = 15.0  # degrees

        # Locate project root relative to this file
        self._project_root = os.path.dirname(os.path.abspath(__file__))
        self._piper_bin    = os.path.join(self._project_root, "piper", "piper")
        self._voice_model  = os.path.join(self._project_root, "voices", "en_US-ryan-medium.onnx")

    # ── Public interface ──────────────────────────────────────────────────────

    def analyze_and_coach(self):
        """Main loop — runs on its own thread."""
        print("[BioSense Coach] Edge intelligence layer active.")
        while self.plotter.is_running:
            time.sleep(1)

            now = time.time()
            if now - self.last_alert_time < self.cooldown:
                continue

            features = self._extract_features()
            if features is None:
                continue

            self._detect_rep(features["pitch_window"])
            anomalies = self._detect_anomalies(features)

            if anomalies:
                self._call_coaching_llm(anomalies, features)

    def generate_session_report(self) -> dict | None:
        """
        Call at session end. Sends full session log to the LLM and returns
        a structured report dict (or None on failure).
        """
        summary = self.session.to_summary_dict()
        if not summary["events"]:
            print("[BioSense Coach] No session data to report.")
            return None

        print("\n[BioSense Coach] Generating end-of-session report...")

        payload = {
            "model":    COACHING_MODEL,
            "tools":    REPORT_TOOLS,
            "tool_choice": {"type": "function", "function": {"name": "generate_session_report"}},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a physiotherapy AI assistant. "
                        "Analyse the provided knee rehabilitation session log "
                        "and call the generate_session_report function with your findings."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Session log:\n{json.dumps(summary, indent=2)}\n\n"
                        "Generate the session report."
                    )
                }
            ],
            "temperature": 0.2,
            "max_tokens":  500
        }

        try:
            res = requests.post(LLAMA_CPP_URL, json=payload, timeout=15)
            res.raise_for_status()
            report = self._parse_tool_call(res.json(), "generate_session_report")
            if report:
                self._print_report(report)
                return report
        except Exception as e:
            print(f"[BioSense Coach] Report generation failed: {e}")

        return None

    # ── Feature extraction ────────────────────────────────────────────────────

    def _extract_features(self) -> dict | None:
        with self.plotter.data_lock:
            emg_window   = np.copy(self.plotter.data_emg[-1000:])
            ax_window    = np.copy(self.plotter.data_ax[-150:])
            pitch_window = np.copy(self.plotter.data_pitch[-150:])

        if len(emg_window) < MIN_EMG_SAMPLES or np.all(emg_window == 0):
            return None

        # Spectral features
        rms = float(np.sqrt(np.mean(np.square(emg_window))))
        freqs, psd = welch(emg_window, fs=self.fs, nperseg=256)
        cum_psd = np.cumsum(psd)
        median_freq = float(
            freqs[np.where(cum_psd >= cum_psd[-1] / 2.0)[0][0]]
        ) if cum_psd[-1] > 0 else 0.0

        # High-frequency power ratio (fatigue indicator — HF drops as MU fire rate slows)
        hf_mask  = freqs > 100
        lf_mask  = (freqs > 20) & (freqs <= 100)
        hf_lf_ratio = (
            float(np.sum(psd[hf_mask]) / np.sum(psd[lf_mask]))
            if np.sum(psd[lf_mask]) > 0 else 1.0
        )

        # Kinematic features
        pitch_delta         = float(np.max(pitch_window) - np.min(pitch_window))
        acceleration_jitter = float(np.std(ax_window))

        return {
            "rms":               rms,
            "median_freq":       median_freq,
            "hf_lf_ratio":       hf_lf_ratio,
            "pitch_delta":       pitch_delta,
            "acceleration_jitter": acceleration_jitter,
            "pitch_window":      pitch_window,
        }

    # ── Rep counter ───────────────────────────────────────────────────────────

    def _detect_rep(self, pitch_window: np.ndarray):
        """Zero-crossing rep counter on pitch signal."""
        if len(pitch_window) < 2:
            return
        mid = float(np.mean(pitch_window))
        current_sign = 1 if pitch_window[-1] > mid + self._rep_threshold else (
                       -1 if pitch_window[-1] < mid - self._rep_threshold else 0)
        if current_sign != 0 and current_sign != self._prev_pitch_sign:
            if self._prev_pitch_sign != 0:
                self.session.log_rep()
                print(f"[BioSense Coach] Rep #{self.session.rep_count} detected")
            self._prev_pitch_sign = current_sign

    # ── Clinical decision matrix ──────────────────────────────────────────────

    def _detect_anomalies(self, f: dict) -> list[str]:
        anomalies = []

        # 1. Neuromuscular fatigue — spectral shift + elevated RMS
        if 0 < f["median_freq"] < 44.0 and f["rms"] > 1200:
            anomalies.append(
                f"Severe neuromuscular fatigue: median frequency shifted to "
                f"{f['median_freq']:.1f} Hz with RMS {f['rms']:.0f} µV"
            )

        # 2. HF power collapse — more sensitive early fatigue indicator
        if f["hf_lf_ratio"] < 0.3 and f["rms"] > 600:
            anomalies.append(
                f"High-frequency power collapse (HF/LF={f['hf_lf_ratio']:.2f}): "
                f"early motor unit synchronisation — fatigue accumulating"
            )

        # 3. Volitional tremor / spasticity
        if f["acceleration_jitter"] > 0.55:
            anomalies.append(
                f"Elevated volitional jitter / tremor: {f['acceleration_jitter']:.2f} g std"
            )

        # 4. Range-of-motion compression
        if f["pitch_delta"] < 15.0 and f["rms"] > 500:
            anomalies.append(
                f"Kinematic ROM compression: only {f['pitch_delta']:.1f}° pitch delta "
                f"under active load — possible guarding or joint stiffness"
            )

        return anomalies

    # ── LLM call (function calling) ───────────────────────────────────────────

    def _call_coaching_llm(self, anomalies: list[str], features: dict):
        payload = {
            "model":       COACHING_MODEL,
            "tools":       COACHING_TOOLS,
            "tool_choice": "auto",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a clinical edge AI integrated into an active knee orthosis. "
                        "Analyse the biomechanical anomalies and call the appropriate function. "
                        "If the patient needs immediate coaching, call trigger_coaching_alert. "
                        "If a clinician should review later, call flag_for_clinician_review. "
                        "Keep voice_cue short, direct, and actionable — it will be spoken aloud."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Anomalies detected:\n"
                        + "\n".join(f"- {a}" for a in anomalies)
                        + f"\n\nAdditional metrics: "
                        f"RMS={features['rms']:.0f} µV, "
                        f"median_freq={features['median_freq']:.1f} Hz, "
                        f"jitter={features['acceleration_jitter']:.2f} g, "
                        f"rep_count={self.session.rep_count}"
                    )
                }
            ],
            "temperature": 0.3,
            "max_tokens":  120
        }

        try:
            res = requests.post(LLAMA_CPP_URL, json=payload, timeout=3)
            res.raise_for_status()
            data = res.json()

            # Try trigger_coaching_alert first
            alert = self._parse_tool_call(data, "trigger_coaching_alert")
            if alert:
                urgency   = alert.get("urgency", "MODERATE")
                voice_cue = alert.get("voice_cue", "")
                rationale = alert.get("rationale", "")

                print(f"\n🧬 [Edge Coach] {urgency} | {rationale}")
                self.session.log_event("alert", {
                    "rms":        features["rms"],
                    "median_freq": features["median_freq"],
                    "jitter":     features["acceleration_jitter"],
                    "anomalies":  anomalies
                }, urgency)

                if voice_cue:
                    print(f"🎙️  Voice → {voice_cue}")
                    self._speak(voice_cue)
                    self.last_alert_time = time.time()
                return

            # Flag for clinician if model chose that instead
            flag = self._parse_tool_call(data, "flag_for_clinician_review")
            if flag:
                print(f"\n📋 [Clinician Flag] {flag.get('severity')} — {flag.get('clinical_note')}")
                self.session.log_event("clinician_flag", flag)
                self.last_alert_time = time.time()
                return

            # Graceful fallback — model returned text instead of tool call
            text_content = self._extract_text_content(data)
            if text_content:
                print(f"\n⚠️  [Fallback text] {text_content}")
                self._speak(text_content[:120])
                self.last_alert_time = time.time()

        except requests.exceptions.Timeout:
            print("[BioSense Coach] LLM timeout — deploying static cue")
            self._static_fallback(anomalies)
        except Exception as e:
            print(f"[BioSense Coach] Pipeline error: {e}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _parse_tool_call(self, response_data: dict, function_name: str) -> dict | None:
        """Extract arguments from an OpenAI-compatible tool_calls response."""
        try:
            message = response_data["choices"][0]["message"]
            tool_calls = message.get("tool_calls", [])
            for tc in tool_calls:
                if tc.get("function", {}).get("name") == function_name:
                    return json.loads(tc["function"]["arguments"])
        except (KeyError, IndexError, json.JSONDecodeError):
            pass
        return None

    def _extract_text_content(self, response_data: dict) -> str:
        """Extract plain text content from a response (non-tool-call fallback)."""
        try:
            return response_data["choices"][0]["message"].get("content", "").strip()
        except (KeyError, IndexError):
            return ""

    def _static_fallback(self, anomalies: list[str]):
        cue = "Form degradation detected. Slow down and maintain controlled movement."
        print(f"🎙️  Voice (static) → {cue}")
        self._speak(cue)
        self.session.log_event("static_fallback", {"anomalies": anomalies})
        self.last_alert_time = time.time()

    def _speak(self, text: str):
        """Pipe text to Piper TTS → aplay."""
        safe_text = text.replace('"', "'").replace('`', "'").replace('\n', ' ')
        cmd = (
            f'echo "{safe_text}" | '
            f'"{self._piper_bin}" --model "{self._voice_model}" --output-raw | '
            f'aplay -r 22050 -f S16_LE -t raw 2>/dev/null &'
        )
        os.system(cmd)

    def _print_report(self, report: dict):
        print("\n" + "═" * 60)
        print("  BIOSENSE SESSION REPORT")
        print("═" * 60)
        print(f"  Performance Score : {report.get('performance_score', '—')}/100")
        print(f"  Total Reps        : {self.session.rep_count}")
        print(f"  Alerts Triggered  : {self.session.alert_count}")
        print(f"\n  Fatigue           : {report.get('fatigue_progression', '—')}")
        print(f"  ROM Trend         : {report.get('rom_trend', '—')}")
        print(f"\n  Patient Summary   : {report.get('summary', '—')}")

        recs = report.get("recommendations", [])
        if recs:
            print("\n  Next Session Recommendations:")
            for i, r in enumerate(recs, 1):
                print(f"    {i}. {r}")

        flags = report.get("clinical_flags", [])
        if flags:
            print("\n  ⚠️  Clinical Flags:")
            for f in flags:
                print(f"    • {f}")
        print("═" * 60 + "\n")