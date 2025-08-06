# -*- coding: utf-8 -*-
import io
import queue
import time
from typing import Optional, Tuple, List

import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
import soundfile as sf

st.set_page_config(page_title="Acoustic Sandbox", layout="wide")

# Optional: librosa for spectrograms/resampling (fallbacks if missing)
try:
    import librosa
    import librosa.display
    HAVE_LIBROSA = True
    LIBROSA_ERR = None
except Exception as e:
    HAVE_LIBROSA = False
    LIBROSA_ERR = e

# Mic capture (WebRTC)
WEBRTC_AVAILABLE = False
try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode
    WEBRTC_AVAILABLE = True
except Exception:
    WEBRTC_AVAILABLE = False

plt.style.use("dark_background")

# =============================================================================
# CACHED/UTILITY FUNCTIONS
# =============================================================================
@st.cache_data(show_spinner=False)
def load_audio(uploaded_file) -> Tuple[np.ndarray, int]:
    """Read file-like to mono float32 via soundfile."""
    data = uploaded_file.read()
    buf = io.BytesIO(data)
    y, sr = sf.read(buf, dtype="float32", always_2d=True)
    uploaded_file.seek(0)
    y = y.mean(axis=1).astype(np.float32)
    return y, sr

def to_wav_buffer(y: np.ndarray, sr: int) -> io.BytesIO:
    b = io.BytesIO()
    sf.write(b, y.astype(np.float32), sr, format="WAV")
    b.seek(0)
    return b

def bytes_to_mono(buf_bytes: bytes) -> Tuple[np.ndarray, int]:
    with io.BytesIO(buf_bytes) as b:
        y, sr = sf.read(b, dtype="float32", always_2d=True)
    y = y.mean(axis=1).astype(np.float32)
    return y, sr

@st.cache_data(show_spinner=False)
def eq_centers_20() -> Tuple[np.ndarray, List[str]]:
    centers = np.geomspace(31.5, 16000, 20)
    labels = [f"{int(round(c))} Hz" if c < 1000 else f"{c/1000:.2g} kHz" for c in centers]
    return centers.astype(float), labels

def apply_graphic_eq_fft(y: np.ndarray, sr: int, centers: np.ndarray, gains_db: np.ndarray) -> np.ndarray:
    n = len(y)
    nfft = 1 << (n - 1).bit_length()
    Y = np.fft.rfft(y, nfft)
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    # Interpolate gains in log-frequency domain
    logf = np.log10(np.maximum(freqs, 1e-1))
    logc = np.log10(centers)
    g_db = np.interp(logf, logc, gains_db, left=gains_db[0], right=gains_db[-1])
    g_lin = 10 ** (g_db / 20.0)
    Y *= g_lin
    y_eq = np.fft.irfft(Y, nfft)[:n]
    return y_eq.astype(np.float32)

def resample_linear(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return y.astype(np.float32)
    duration = len(y) / float(orig_sr)
    t_old = np.linspace(0, duration, num=len(y), endpoint=False)
    n_new = int(round(duration * target_sr))
    t_new = np.linspace(0, duration, num=n_new, endpoint=False)
    y_new = np.interp(t_new, t_old, y).astype(np.float32)
    return y_new

# =============================================================================
# SYNTHETIC IRs AND CONVOLUTION
# =============================================================================
def _synth_ir(sr: int,
              rt60: float,
              predelay_ms: float = 0.0,
              density: float = 1200.0,
              hf_damp: float = 0.0,
              bass_boost_db: float = 0.0,
              early_taps: Optional[List[Tuple[float, float]]] = None,
              seed: Optional[int] = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = max(int(sr * rt60), int(0.2 * sr))
    ir = np.zeros(n + int(predelay_ms * 1e-3 * sr) + 1, dtype=np.float32)
    offset = int(predelay_ms * 1e-3 * sr)

    # Early reflections
    if early_taps:
        for t_s, g in early_taps:
            idx = offset + int(t_s * sr)
            if 0 <= idx < len(ir):
                ir[idx] += g

    # Dense tail
    num_taps = int(density * max(rt60, 0.05))
    tap_times = rng.uniform(0.0, rt60, size=num_taps)
    tap_idx = offset + (tap_times * sr).astype(int)
    tap_idx = tap_idx[tap_idx < len(ir)]
    amp_env = np.exp(-tap_times * 6.91 / max(rt60, 1e-6))[: len(tap_idx)]
    taps = (rng.normal(0, 1, size=len(tap_idx)).astype(np.float32) * amp_env).astype(np.float32)

    if hf_damp > 0:
        t = tap_times[: len(tap_idx)]
        taps *= np.exp(-hf_damp * t).astype(np.float32)

    for idx, a in zip(tap_idx, taps):
        ir[idx] += a

    # Bass shaping
    if bass_boost_db != 0.0:
        boost = 10 ** (bass_boost_db / 20.0)
        klen = int(sr * 0.004) + 3
        k = np.hanning(klen).astype(np.float32)
        k = k / (k.sum() + 1e-12)
        low = fftconvolve(ir, k, mode="same")
        ir = (0.7 * ir + 0.3 * boost * low).astype(np.float32)

    peak = float(np.max(np.abs(ir))) or 1.0
    return (ir / peak).astype(np.float32)

@st.cache_data(show_spinner=False)
def make_ir(sr: int, preset: str, custom_ir_bytes: Optional[bytes] = None) -> np.ndarray:
    if custom_ir_bytes is not None:
        y_ir, ir_sr = bytes_to_mono(custom_ir_bytes)
        if ir_sr != sr:
            if HAVE_LIBROSA:
                y_ir = librosa.resample(y_ir, orig_sr=ir_sr, target_sr=sr)
            else:
                st.warning(f"Custom IR sr={ir_sr} ≠ audio sr={sr}. Using simple linear resampler.")
                y_ir = resample_linear(y_ir, ir_sr, sr)
        peak = float(np.max(np.abs(y_ir))) or 1.0
        return (y_ir / peak).astype(np.float32)

    presets = {
        "Anechoic (dry)": dict(rt60=0.001, predelay_ms=0, density=0, hf_damp=0.0, bass_boost_db=0.0, early_taps=[]),
        "Vocal Booth":    dict(rt60=0.20,  predelay_ms=1, density=700, hf_damp=2.0, bass_boost_db=0.0, early_taps=[(0.004, 0.6), (0.007, 0.4)]),
        "Small Room":     dict(rt60=0.35,  predelay_ms=2, density=1000, hf_damp=1.0, bass_boost_db=0.0, early_taps=[(0.003, 0.8), (0.006, 0.5), (0.012, 0.35)]),
        "Tiled Bathroom": dict(rt60=0.80,  predelay_ms=3, density=1300, hf_damp=0.2, bass_boost_db=0.0, early_taps=[(0.002, 0.9), (0.004, 0.7), (0.007, 0.6), (0.011, 0.4)]),
        "Plate Reverb":   dict(rt60=1.20,  predelay_ms=8, density=2500, hf_damp=0.3, bass_boost_db=0.0, early_taps=[(0.010, 0.7)]),
        "Spring Reverb":  dict(rt60=0.90,  predelay_ms=6, density=400, hf_damp=0.1, bass_boost_db=0.0, early_taps=[(0.012, 0.8), (0.024, 0.6), (0.036, 0.4)]),
        "Large Hall":     dict(rt60=1.80,  predelay_ms=12, density=1800, hf_damp=0.8, bass_boost_db=0.0, early_taps=[(0.015, 0.8), (0.030, 0.5)]),
        "Cathedral":      dict(rt60=4.50,  predelay_ms=20, density=2200, hf_damp=1.5, bass_boost_db=2.0, early_taps=[(0.020, 0.9), (0.040, 0.6), (0.065, 0.4)]),
        "Car Cabin":      dict(rt60=0.40,  predelay_ms=1, density=900, hf_damp=1.2, bass_boost_db=4.0, early_taps=[(0.0018, 0.8), (0.0045, 0.6), (0.007, 0.45)]),
        "Underground Tunnel": dict(rt60=2.50, predelay_ms=18, density=700, hf_damp=0.6, bass_boost_db=1.0, early_taps=[(0.025, 0.8), (0.050, 0.6), (0.080, 0.4)]),
        "Stadium":        dict(rt60=3.00,  predelay_ms=30, density=1200, hf_damp=0.9, bass_boost_db=1.0, early_taps=[(0.040, 0.9), (0.075, 0.6), (0.120, 0.4)]),
        "Club (dense, bright)": dict(rt60=1.00, predelay_ms=8, density=3000, hf_damp=0.2, bass_boost_db=6.0, early_taps=[(0.010, 0.7), (0.016, 0.5)]),
        "Damped Studio":  dict(rt60=0.30,  predelay_ms=2, density=800, hf_damp=2.5, bass_boost_db=0.0, early_taps=[(0.003, 0.7)]),
        "Forest / Outdoor": dict(rt60=0.40, predelay_ms=0, density=150, hf_damp=1.8, bass_boost_db=0.0, early_taps=[(0.040, 0.5)]),
        "Stairwell":      dict(rt60=1.50,  predelay_ms=10, density=1000, hf_damp=0.7, bass_boost_db=0.0, early_taps=[(0.012, 0.8), (0.021, 0.6), (0.034, 0.45)]),
    }
    if preset not in presets:
        preset = "Small Room"
    return _synth_ir(sr, **presets[preset])

def convolve_ir(y: np.ndarray, ir: np.ndarray) -> np.ndarray:
    out = fftconvolve(y, ir, mode="full")[: len(y)]
    peak = np.max(np.abs(out)) or 1.0
    return (out / peak * 0.95).astype(np.float32)

def plot_spectrogram(y: np.ndarray, sr: int, title: str, n_fft: int = 1024, hop: int = 256):
    fig, ax = plt.subplots(figsize=(9, 3))
    if HAVE_LIBROSA:
        S = librosa.stft(y, n_fft=n_fft, hop_length=hop)
        S_db = librosa.amplitude_to_db(np.abs(S) + 1e-12, ref=np.max)
        img = librosa.display.specshow(S_db, sr=sr, hop_length=hop, x_axis="time", y_axis="log", ax=ax, cmap="magma")
        ax.set(title=title)
        cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB")
        cbar.ax.set_ylabel("dB")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("white")
    else:
        t = np.arange(len(y)) / sr
        ax.plot(t, y)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"{title} (waveform)")
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

# =============================================================================
# SWEEP GENERATION & IR DECONVOLUTION
# =============================================================================
def generate_log_sweep(sr: int, T: float, f_start: float = 20.0, f_end: float = 20000.0, fade_ms: float = 10.0) -> np.ndarray:
    """Log sine sweep with Hann in/out fades."""
    t = np.linspace(0, T, int(sr * T), endpoint=False)
    K = T / np.log(f_end / f_start)
    phase = 2.0 * np.pi * f_start * K * (np.exp(t / K) - 1.0)
    sweep = np.sin(phase).astype(np.float32)

    n_fade = int(sr * fade_ms * 1e-3)
    if n_fade > 0 and n_fade * 2 < len(sweep):
        w = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n_fade)))
        sweep[:n_fade] *= w
        sweep[-n_fade:] *= w[::-1]
    return sweep

def deconvolve_ir(recorded: np.ndarray, sweep: np.ndarray, sr: int, eps: float = 1e-6) -> np.ndarray:
    """Frequency-domain deconvolution with Tikhonov regularization."""
    L = 1 << ((len(recorded) + len(sweep) - 1) - 1).bit_length()
    R = np.fft.rfft(recorded, L)
    S = np.fft.rfft(sweep, L)
    H = R * np.conj(S) / (np.abs(S) ** 2 + eps)
    h = np.fft.irfft(H, L).astype(np.float32)
    # Align to main peak and trim
    peak = int(np.argmax(np.abs(h)))
    h = h[peak : peak + int(sr * 6)]  # keep up to 6 s tail by default
    pk = np.max(np.abs(h)) or 1.0
    return (h / pk).astype(np.float32)

# =============================================================================
# IR RESOLUTION (preset/custom/measured)
# =============================================================================
def get_effective_ir(sr: int, preset: str, custom_ir_file) -> np.ndarray:
    """Resolve IR priority: measured override -> custom upload -> preset synth."""
    if "override_ir_bytes" in st.session_state and st.session_state["override_ir_bytes"] is not None:
        return make_ir(sr, preset, custom_ir_bytes=st.session_state["override_ir_bytes"])
    else:
        custom_bytes = custom_ir_file.read() if custom_ir_file is not None else None
        if custom_ir_file is not None:
            custom_ir_file.seek(0)
        return make_ir(sr, preset, custom_ir_bytes=custom_bytes)

# =============================================================================
# UI
# =============================================================================
st.set_page_config(page_title="Acoustic Sandbox (Rewritten)", layout="wide")
st.title("🔊 Acoustic Sandbox — Equalisation & Impulse Resoponses")

if not HAVE_LIBROSA:
    with st.expander("Librosa not available (spectrogram falls back to waveform)"):
        st.exception(LIBROSA_ERR)

# ---------------- Sidebar controls ----------------
with st.sidebar:
    st.header("🎚️ 20-Band Graphic EQ")
    centers, labels = eq_centers_20()
    gains = []
    for i, label in enumerate(labels):
        g = st.slider(label=label, min_value=-20.0, max_value=20.0, value=0.0, step=0.5, key=f"eq_{i}")
        gains.append(g)
    gains_db = np.array(gains, dtype=np.float32)

    st.divider()
    st.header("🌍 Reverb / IR")
    ir_preset = st.selectbox(
        "Room preset",
        [
            "Anechoic (dry)", "Vocal Booth", "Small Room", "Tiled Bathroom", "Plate Reverb",
            "Spring Reverb", "Large Hall", "Cathedral", "Car Cabin", "Underground Tunnel",
            "Stadium", "Club (dense, bright)", "Damped Studio", "Forest / Outdoor", "Stairwell"
        ],
        index=2,
    )
    custom_ir_file = st.file_uploader("Or upload custom IR (WAV mono/any length)", type=["wav"])
    wet = st.slider("Wet mix", 0.0, 1.0, 0.35, step=0.05)

    st.divider()
    st.header("🎚 Gain Staging")
    pregain = st.slider("Pre-gain (dB)", -24, 24, 0)
    postgain = st.slider("Post-gain (dB)", -24, 24, 0)

    st.divider()
    st.header("⚡ Preview")
    preview_sec = st.slider("Preview length (s)", 1, 15, 5)

# ---------------- Upload ----------------
uploaded = st.file_uploader("Upload audio (WAV/MP3/OGG)", type=["wav", "mp3", "ogg"])
if not uploaded:
    st.info("Upload an audio file to begin.")
    st.stop()

with st.status("Loading audio…", expanded=False):
    y_full, sr = load_audio(uploaded)

# Build preview snippet (middle)
n_total = len(y_full)
n_preview = min(int(preview_sec * sr), n_total)
start = max(0, (n_total - n_preview) // 2)
end = start + n_preview
y_preview = y_full[start:end]
orig_preview_buf = to_wav_buffer(y_preview, sr)

# Preview columns
col1, col2 = st.columns(2)
with col1:
    st.subheader("Original (preview)")
    st.audio(orig_preview_buf, format="audio/wav")
    plot_spectrogram(y_preview, sr, "Original (Preview)")

# Process preview
pregain_lin = 10 ** (pregain / 20.0)
postgain_lin = 10 ** (postgain / 20.0)
y_pre = (y_preview * pregain_lin).astype(np.float32)

with st.status("Applying 20-band EQ (preview)…", expanded=False):
    y_eq = apply_graphic_eq_fft(y_pre, sr, centers, gains_db)

with st.status("Building IR & convolving (preview)…", expanded=False):
    ir = get_effective_ir(sr, ir_preset, custom_ir_file)
    y_ir = convolve_ir(y_eq, ir)

y_proc_prev = ((1.0 - wet) * y_eq + wet * y_ir) * postgain_lin
peak_prev = np.max(np.abs(y_proc_prev)) or 1.0
y_proc_prev = (y_proc_prev * min(1.0, 0.98 / peak_prev)).astype(np.float32)
proc_preview_buf = to_wav_buffer(y_proc_prev, sr)

with col2:
    st.subheader("Processed (preview)")
    st.audio(proc_preview_buf, format="audio/wav")
    plot_spectrogram(y_proc_prev, sr, "Processed (Preview)")

# ---------------- Full processing ----------------
st.divider()
if st.button("🚀 Process FULL file with current settings"):
    with st.spinner("Processing full file…"):
        y_full_pre = (y_full * pregain_lin).astype(np.float32)
        y_full_eq = apply_graphic_eq_fft(y_full_pre, sr, centers, gains_db)
        ir_full = get_effective_ir(sr, ir_preset, custom_ir_file)
        y_full_ir = convolve_ir(y_full_eq, ir_full)
        y_full_proc = ((1.0 - wet) * y_full_eq + wet * y_full_ir) * postgain_lin
        peak_full = np.max(np.abs(y_full_proc)) or 1.0
        y_full_proc = (y_full_proc * min(1.0, 0.98 / peak_full)).astype(np.float32)

        st.session_state["full_original_buf"] = to_wav_buffer(y_full, sr).getvalue()
        st.session_state["full_processed_buf"] = to_wav_buffer(y_full_proc, sr).getvalue()
        st.success("Full file processed.")

if "full_original_buf" in st.session_state and "full_processed_buf" in st.session_state:
    st.subheader("🎧 Full Length (Original vs Processed)")
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Original (full)**")
        full_orig_buf = io.BytesIO(st.session_state["full_original_buf"])
        st.audio(full_orig_buf, format="audio/wav")
        st.download_button("⬇️ Download Original (full)", data=io.BytesIO(st.session_state["full_original_buf"]),
                           file_name="original_full.wav", mime="audio/wav")
    with c2:
        st.write("**Processed (full)**")
        full_proc_buf = io.BytesIO(st.session_state["full_processed_buf"])
        st.audio(full_proc_buf, format="audio/wav")
        st.download_button("⬇️ Download Processed (full)", data=io.BytesIO(st.session_state["full_processed_buf"]),
                           file_name="processed_full.wav", mime="audio/wav")

    st.subheader("📊 Full Spectrograms")
    y_full_orig, sr_o = bytes_to_mono(st.session_state["full_original_buf"])
    y_full_proc, sr_p = bytes_to_mono(st.session_state["full_processed_buf"])
    sr_plot = sr_o if sr_o == sr_p else sr
    a1, a2 = st.columns(2)
    with a1:
        plot_spectrogram(y_full_orig, sr_plot, "Original (Full)")
    with a2:
        plot_spectrogram(y_full_proc, sr_plot, "Processed (Full)")

st.divider()
st.download_button("⬇️ Download preview (processed)", data=proc_preview_buf,
                   file_name="processed_preview.wav", mime="audio/wav")

# =============================================================================
# MEASURE YOUR OWN ROOM IR (MIC) — BETA
# =============================================================================
st.header("🧪 Measure Room IR (Mic) — Beta")
st.caption("Play a log sweep through your speakers and record with your mic. Disable echo cancellation / AGC / noise suppression if possible in your OS/browser.")

# Prepare ICE servers (STUN default, optional TURN from secrets)
ice_servers = [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]
try:
    if "iceServers" in st.secrets:
        # Expect a list like: [{"urls": "...", "username": "...", "credential": "..."}]
        ice_servers.extend(st.secrets["iceServers"])
except Exception:
    pass

if not WEBRTC_AVAILABLE:
    st.warning("Install `streamlit-webrtc` and `av` to enable mic capture:\n`pip install streamlit-webrtc av`")
else:
    with st.expander("Measure & use a custom room IR"):
        msr_sr = st.selectbox("Sample rate for measurement", [48000, 44100], index=0)
        msr_T = st.slider("Sweep length (s)", 4, 20, 10)
        msr_fstart = st.number_input("Start freq (Hz)", 10.0, 200.0, 20.0, step=1.0)
        msr_fend = st.number_input("End freq (Hz)", 2000.0, 24000.0, 20000.0, step=100.0)
        msr_tail = st.slider("Extra tail to record after sweep (s)", 1, 8, 3)
        ir_keep_s = st.slider("IR length to keep (s)", 1, 8, 6)

        # Generate sweep & player
        sweep = generate_log_sweep(msr_sr, msr_T, msr_fstart, msr_fend)
        sweep_buf = to_wav_buffer(sweep, msr_sr)
        st.write("1) Set your volume. 2) Toggle **Start capture**. 3) Press ▶️ to play the sweep while capturing. 4) Click **Compute IR**.")
        st.audio(sweep_buf, format="audio/wav")

        # Start/stop mic capture
        capturing = st.toggle("🎙️ Start capture", value=False, help="Starts the browser microphone stream.")

        # WebRTC mic: SENDONLY means browser sends mic to the app
        ctx = webrtc_streamer(
            key="ir-measure",
            mode=WebRtcMode.SENDONLY,
            audio_receiver_size=1024,  # increase queue for buffering
            rtc_configuration={"iceServers": ice_servers},
            media_stream_constraints={
                "audio": {
                    "echoCancellation": False,
                    "noiseSuppression": False,
                    "autoGainControl": False,
                },
                "video": False,
            },
            desired_playing_state=capturing,  # bound to toggle
        )

        # Initialize buffers
        if "rec_chunks" not in st.session_state:
            st.session_state["rec_chunks"] = []
            st.session_state["rec_sr"] = msr_sr
            st.session_state["rec_samples"] = 0

        status_ph = st.empty()

        # Pull frames while playing
        if ctx and ctx.state.playing and ctx.audio_receiver:
            try:
                frames = ctx.audio_receiver.get_frames(timeout=0.2)
            except queue.Empty:
                frames = []
            for f in frames:
                arr = f.to_ndarray()  # (channels, samples) or (samples,)
                if arr.ndim == 2:
                    arr = arr.mean(axis=0)
                arr = arr.astype(np.float32)
                st.session_state["rec_chunks"].append(arr)
                st.session_state["rec_samples"] += arr.shape[-1]
                # Some environments provide f.sample_rate
                sr_frame = getattr(f, "sample_rate", None)
                if sr_frame:
                    st.session_state["rec_sr"] = sr_frame

            dur = st.session_state["rec_samples"] / float(st.session_state["rec_sr"] or msr_sr)
            status_ph.info(f"Capturing… {dur:.1f}s of audio buffered.")

        colm1, colm2, colm3 = st.columns(3)
        with colm1:
            if st.button("🧹 Reset recording buffer"):
                st.session_state["rec_chunks"] = []
                st.session_state["rec_samples"] = 0
                st.session_state.pop("measured_ir", None)
                st.toast("Recording buffer cleared.", icon="🧹")

        with colm2:
            st.caption(f"Tip: aim for ≥ {msr_T + msr_tail:.1f}s capture (sweep + tail).")

        with colm3:
            if st.button("🧮 Compute IR from recording"):
                if len(st.session_state["rec_chunks"]) == 0:
                    st.error("No audio captured yet. Start capture, play the sweep, then try again.")
                else:
                    recorded = np.concatenate(st.session_state["rec_chunks"])
                    rec_sr = st.session_state.get("rec_sr", msr_sr)

                    # Resample recorded to target measurement rate if needed
                    if rec_sr != msr_sr:
                        if HAVE_LIBROSA:
                            recorded = librosa.resample(recorded, orig_sr=rec_sr, target_sr=msr_sr)
                        else:
                            st.warning(f"Mic sr={rec_sr} ≠ measurement sr={msr_sr}. Using linear resampler.")
                            recorded = resample_linear(recorded, rec_sr, msr_sr)

                    # Keep at least sweep + tail samples if the user under-recorded
                    min_len = int((msr_T + msr_tail) * msr_sr)
                    if len(recorded) < min_len:
                        st.warning("Captured duration is shorter than sweep+tail; results may be noisy.")

                    # Deconvolve and trim
                    ir_est = deconvolve_ir(recorded.astype(np.float32), sweep.astype(np.float32), msr_sr)
                    ir_est = ir_est[: int(ir_keep_s * msr_sr)]
                    st.session_state["measured_ir"] = ir_est.astype(np.float32)

        # If we have a measured IR, preview & allow using it
        if "measured_ir" in st.session_state:
            st.success("Measured IR ready.")
            ir_est = st.session_state["measured_ir"]
            buf_ir = to_wav_buffer(ir_est, msr_sr)
            st.audio(buf_ir, format="audio/wav")
            st.download_button("⬇️ Download measured IR (WAV)", data=buf_ir, file_name="measured_ir.wav", mime="audio/wav")

            if st.button("🎧 Use this IR for processing (overrides preset/custom)"):
                st.session_state["override_ir_bytes"] = buf_ir.getvalue()
                # Clear old full-length renders safely (avoid None BytesIO errors)
                st.session_state.pop("full_original_buf", None)
                st.session_state.pop("full_processed_buf", None)
                st.toast("Measured IR will be used for new processing runs.", icon="✅")
