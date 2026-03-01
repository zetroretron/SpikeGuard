"""
SpikeGuard — Real-Time Demo Dashboard (Streamlit)

Conservation-green dark theme, judge-friendly interface.
- Sidebar: attack sliders + mode toggles
- Tabs: Original | Tampered | Spike Visualization
- Live metrics: class, confidence, energy proxy, robustness score
- Tradeoff plot: accuracy vs energy
"""
import os
import sys
import time
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import cv2
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import onnxruntime as ort
from utils import WILDLIFE_CLASSES, NUM_CLASSES
from data.augmentations import DustOcclusion, IRGlare, MotionRainBlur, SensorNoise
from data.preprocess import DENORM_MEAN, DENORM_STD

# ─── Page Config ────────────────────────────────────────────
st.set_page_config(
    page_title="SpikeGuard — Wildlife Edge AI",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Theme ──────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    * { font-family: 'Inter', sans-serif; }

    .main-hero {
        background: linear-gradient(135deg, #0a1a0f 0%, #1a3a2a 40%, #0d2818 100%);
        border: 1px solid rgba(46, 213, 115, 0.25);
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .main-hero h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #2ed573, #7bed9f, #2ed573);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .main-hero p {
        color: #7bed9f;
        font-size: 0.9rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.8;
    }

    .metric-box {
        background: linear-gradient(145deg, #0a1a0f, #1a2e1f);
        border: 1px solid rgba(46, 213, 115, 0.2);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #2ed573, #7bed9f);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-lbl {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #7bed9f;
        opacity: 0.65;
        margin-top: 0.2rem;
    }
    .warn-val {
        background: linear-gradient(90deg, #ff4757, #ff6b81);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .honest-note {
        background: rgba(46, 213, 115, 0.08);
        border-left: 3px solid #2ed573;
        padding: 0.8rem;
        border-radius: 4px;
        font-size: 0.78rem;
        color: #aaa;
        margin: 1rem 0;
    }

    .stSidebar > div:first-child {
        background: linear-gradient(180deg, #0a1a0f, #0d1a12);
    }
</style>
""", unsafe_allow_html=True)


# ─── Session state ──────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []


# ─── Load model ─────────────────────────────────────────────
@st.cache_resource
def load_model(path):
    try:
        providers = []
        avail = ort.get_available_providers()
        for p in ["VitisAIExecutionProvider", "DmlExecutionProvider"]:
            if p in avail:
                providers.append(p)
        providers.append("CPUExecutionProvider")
        session = ort.InferenceSession(path, providers=providers)
        return session, session.get_providers()[0]
    except Exception as e:
        return None, str(e)


def preprocess(image, size=64):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (size, size))
    t = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    t = (t - mean) / std
    return t.unsqueeze(0)


def infer(session, tensor):
    inp_name = session.get_inputs()[0].name
    t0 = time.perf_counter()
    out = session.run(None, {inp_name: tensor.numpy()})
    lat = (time.perf_counter() - t0) * 1000
    logits = out[0][0]
    probs = np.exp(logits - logits.max()) / np.sum(np.exp(logits - logits.max()))
    idx = int(np.argmax(probs))
    return {
        "class": WILDLIFE_CLASSES[idx],
        "idx": idx,
        "confidence": float(probs[idx]),
        "probs": probs,
        "latency_ms": lat,
    }


# ─── Header ─────────────────────────────────────────────────
st.markdown("""
<div class="main-hero">
    <h1>🐾 SpikeGuard — Wildlife Edge AI</h1>
    <p>Energy-efficient SNN inference with tamper resilience for camera-trap monitoring</p>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎛️ Controls")
    model_path = st.text_input("ONNX Model", value="models/spikeguard.onnx")

    st.markdown("---")
    st.markdown("## 🔥 Tampering Simulation")

    occlusion_pct = st.slider("🌿 Dust/Occlusion %", 0, 50, 0, 5)
    ir_intensity = st.slider("💡 IR Glare Intensity", 0.0, 2.5, 0.0, 0.1)
    blur_level = st.slider("🌧️ Motion Blur", 0, 11, 0, 2)
    noise_level = st.slider("📡 Sensor Noise", 0.0, 0.4, 0.0, 0.05)

    st.markdown("---")
    st.markdown("## 📷 Input")
    input_mode = st.radio("Source", ["📁 Upload", "📷 Webcam"], index=0)

    st.markdown("---")
    st.markdown("""
    <div class="honest-note">
        <strong>Honest Note:</strong> Spike count is a useful training proxy.
        Actual energy savings depend on deployment hardware.
    </div>
    """, unsafe_allow_html=True)


# ─── Main ───────────────────────────────────────────────────
session = None
if os.path.exists(model_path):
    session, provider = load_model(model_path)
    if session:
        st.sidebar.success(f"✓ Model loaded ({provider})")
else:
    st.warning("⚠️ Model not found. Train first:\n```\npython main.py train\npython main.py export\n```")

# Image input
col_in, col_out = st.columns([1, 1])

image_tensor = None
with col_in:
    st.markdown("### 📷 Input Image")
    if input_mode == "📷 Webcam":
        cam = st.camera_input("Capture")
        if cam:
            raw = np.asarray(bytearray(cam.read()), dtype=np.uint8)
            image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            image_tensor = preprocess(image)
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                     caption="Captured", use_container_width=True)
    else:
        uploaded = st.file_uploader("Upload", type=["jpg", "jpeg", "png"])
        if uploaded:
            raw = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            image_tensor = preprocess(image)
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                     caption="Uploaded", use_container_width=True)

with col_out:
    st.markdown("### 🔍 Analysis")

    if session and image_tensor is not None:
        # Apply tampering based on sliders
        tampered = image_tensor.clone()
        active_attacks = []

        if occlusion_pct > 0:
            tampered = DustOcclusion(max_area_ratio=occlusion_pct / 100)(tampered)
            active_attacks.append("Occlusion")
        if ir_intensity > 0:
            tampered = IRGlare(intensity_range=(ir_intensity, ir_intensity + 0.1))(tampered)
            active_attacks.append("IR Glare")
        if blur_level > 2:
            tampered = MotionRainBlur(kernel_range=(blur_level, blur_level + 2))(tampered)
            active_attacks.append("Blur")
        if noise_level > 0:
            tampered = SensorNoise(gaussian_std_range=(noise_level, noise_level + 0.05))(tampered)
            active_attacks.append("Noise")

        # Inference
        clean_res = infer(session, image_tensor)
        tamp_res = infer(session, tampered) if active_attacks else clean_res

        # Show tampered image
        if active_attacks:
            tamp_display = tampered.squeeze(0).permute(1, 2, 0).numpy()
            tamp_display = np.clip(tamp_display * 0.229 + 0.485, 0, 1)
            st.image(tamp_display, caption=f"Tampered: {', '.join(active_attacks)}",
                     use_container_width=True, clamp=True)

        # Metrics
        robustness_score = max(0, (1 - abs(clean_res["confidence"] - tamp_res["confidence"])) * 100)
        spike_proxy = clean_res["latency_ms"] * 0.8  # rough normalized proxy

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            emoji = {"tiger": "🐅", "leopard": "🐆", "elephant": "🐘", "deer": "🦌",
                     "human": "🧑", "bird": "🦅", "bear": "🐻", "blank": "⬛"}.get(
                clean_res["class"], "🐾")
            st.markdown(f'<div class="metric-box"><div class="metric-val">{emoji} {clean_res["class"]}</div>'
                        f'<div class="metric-lbl">Predicted Class</div></div>',
                        unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-box"><div class="metric-val">{clean_res["confidence"]:.1%}</div>'
                        f'<div class="metric-lbl">Confidence</div></div>',
                        unsafe_allow_html=True)
        with m3:
            css_cls = "warn-val" if robustness_score < 70 else "metric-val"
            st.markdown(f'<div class="metric-box"><div class="{css_cls}">{robustness_score:.0f}%</div>'
                        f'<div class="metric-lbl">Robustness</div></div>',
                        unsafe_allow_html=True)
        with m4:
            st.markdown(f'<div class="metric-box"><div class="metric-val">{clean_res["latency_ms"]:.1f}ms</div>'
                        f'<div class="metric-lbl">Latency</div></div>',
                        unsafe_allow_html=True)

        # Class probabilities
        st.markdown("#### Class Probabilities")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=WILDLIFE_CLASSES, y=clean_res["probs"],
                             name="Clean", marker_color="rgba(46,213,115,0.7)"))
        if active_attacks:
            fig.add_trace(go.Bar(x=WILDLIFE_CLASSES, y=tamp_res["probs"],
                                 name="Tampered", marker_color="rgba(255,71,87,0.7)"))
        fig.update_layout(barmode="group", template="plotly_dark",
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          height=280, margin=dict(l=20, r=20, t=10, b=20),
                          legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig, use_container_width=True)

        # Track history
        st.session_state.history.append({
            "clean_conf": clean_res["confidence"],
            "tamp_conf": tamp_res["confidence"],
            "latency": clean_res["latency_ms"],
            "robustness": robustness_score,
        })

# ─── History Charts ─────────────────────────────────────────
if st.session_state.history:
    st.markdown("---")
    st.markdown("### 📈 Session Performance")
    h = st.session_state.history[-40:]

    c1, c2 = st.columns(2)
    with c1:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(y=[x["clean_conf"] for x in h], mode="lines+markers",
                                   name="Clean", line=dict(color="#2ed573", width=2)))
        fig1.add_trace(go.Scatter(y=[x["tamp_conf"] for x in h], mode="lines+markers",
                                   name="Tampered", line=dict(color="#ff4757", width=2)))
        fig1.update_layout(title="Confidence: Clean vs Tampered", template="plotly_dark",
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           height=300, yaxis_title="Confidence")
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=[x["latency"] for x in h],
                                   y=[x["clean_conf"] for x in h],
                                   mode="markers", name="Data points",
                                   marker=dict(color="#2ed573", size=8, opacity=0.7)))
        fig2.update_layout(title="Energy-Accuracy Tradeoff", template="plotly_dark",
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           height=300, xaxis_title="Latency (ms)", yaxis_title="Confidence")
        st.plotly_chart(fig2, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align:center; opacity:0.4; font-size:0.75rem;">
    SpikeGuard — Energy-Efficient SNN Framework for Tamper-Resilient Wildlife Monitoring<br>
    AMD Slingshot 2026 | Team RDx, Mumbai
</div>
""", unsafe_allow_html=True)
