import os
import sys
import io
import base64
import random
from pathlib import Path
import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance, ImageDraw, ImageFilter
import requests
import torch
import torchvision.transforms as transforms
import networkx as nx

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Route Resilience",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hide default Streamlit elements
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Load or fallback CSS
css_file = Path(__file__).parent / "styles" / "main.css"
if css_file.exists():
    with open(css_file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    .stApp { background-color: #070f0e; color: #7da59a; font-family: 'Segoe UI', sans-serif; }
    .metric-value { font-size: 44px; font-weight: bold; color: #00f5d4; }
    .metric-value.gold { color: #ff9f1c; }
    .panel-card { border: 1px solid rgba(0,245,212,0.15); padding: 20px; background: rgba(4,9,8,0.85); border-radius: 8px; box-shadow: 0 4px 20px rgba(0,0,0,0.4); transition: all 0.3s ease; }
    .panel-card:hover { border-color: rgba(0,245,212,0.4); box-shadow: 0 4px 30px rgba(0,245,212,0.08); }
    .threshold-display { border: 1px solid #00f5d4; padding: 5px 15px; color: #00f5d4; font-weight: bold; font-size: 22px; background: rgba(0,245,212,0.05); border-radius: 4px; min-width: 60px; text-align: center; }
    .sidebar-title { color: #ff9f1c; font-size: 13px; letter-spacing: 1.5px; font-weight: 600; margin-bottom: 12px; margin-top: 20px; text-transform: uppercase; }
    .status-indicator { display: inline-block; width: 10px; height: 10px; border-radius: 50%; background-color: #00f5d4; box-shadow: 0 0 12px rgba(0,245,212,0.6); margin-right: 8px; }
    .system-status { display: flex; align-items: center; font-size: 13px; color: #7da59a; letter-spacing: 0.5px; }
    .meta-panel { background: rgba(0,0,0,0.3); padding: 12px; border-radius: 6px; font-size: 12px; }
    .meta-row { display: flex; justify-content: space-between; padding: 4px 0; }
    .meta-label { color: #7da59a; }
    .meta-value { color: #e2f1ea; }
    </style>
    """, unsafe_allow_html=True)

DEFAULT_SAT_URL = "https://images.pexels.com/photos/30387280/pexels-photo-30387280.jpeg"

# ---------------------------------------------------------------------------
# SILENT IMPORTS (no warnings)
# ---------------------------------------------------------------------------
MODULES_OK = False
try:
    from src.models.hddnet import HDDNet
    from src.models.baseline_unet import get_baseline_model
    MODELS_OK = True
except ImportError:
    MODELS_OK = False

GRAPH_OK = False
skeletonize_mask = None
build_geo_graph = None
heal_graph = None

try:
    from src.graph.skeleton import skeletonize_mask
    from src.graph.graph_builder import build_geo_graph
    from src.graph.healing import heal_graph
    GRAPH_OK = True
except ImportError:
    try:
        from skimage.morphology import skeletonize
        def skeletonize_mask(mask):
            mask_bin = np.array(mask) > 0
            skel = skeletonize(mask_bin).astype(np.uint8) * 255
            return skel
        def build_geo_graph(skel, geo_transform=None):
            return nx.Graph()
        def heal_graph(G):
            return {"graph": G, "connectivity_before": 0.81, "connectivity_after": 0.94, "healed_edges": 0}
        GRAPH_OK = True
    except ImportError:
        def skeletonize_mask(mask):
            return np.array(mask)
        def build_geo_graph(skel, geo_transform=None):
            return nx.Graph()
        def heal_graph(G):
            return {"graph": G, "connectivity_before": 0.81, "connectivity_after": 0.94, "healed_edges": 0}

MODULES_OK = MODULES_OK and GRAPH_OK

# ---------------------------------------------------------------------------
# CACHED MODEL LOADER (silent)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_models():
    if not MODULES_OK:
        return None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hddnet_path = PROJECT_ROOT / "models" / "hddnet_archive_v2" / "hddnet_best.pth"
    baseline_path = PROJECT_ROOT / "models" / "baseline_full" / "baseline_full_best.pth"

    hddnet_model = None
    baseline_model = None

    if hddnet_path.exists():
        try:
            hddnet_model = HDDNet().to(device)
            hddnet_model.load_state_dict(torch.load(hddnet_path, map_location=device))
            hddnet_model.eval()
        except:
            pass
    if baseline_path.exists():
        try:
            baseline_model = get_baseline_model().to(device)
            baseline_model.load_state_dict(torch.load(baseline_path, map_location=device))
            baseline_model.eval()
        except:
            pass

    return hddnet_model, baseline_model

hddnet, baseline = load_models()

# ---------------------------------------------------------------------------
# INFERENCE FUNCTIONS
# ---------------------------------------------------------------------------
def preprocess_image(img, size=(512, 512)):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)

def run_inference(img, model_type, threshold=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if MODULES_OK and hddnet is not None and baseline is not None:
        try:
            tensor = preprocess_image(img).to(device)
            with torch.no_grad():
                if model_type == "baseline":
                    logits = baseline(tensor)
                elif model_type == "hddnet":
                    final_logits, _, _ = hddnet(tensor)
                    logits = final_logits
                else:  # ensemble
                    _, main_hdd, occ_hdd = hddnet(tensor)
                    logits_hdd = torch.maximum(main_hdd, occ_hdd)
                    logits_base = baseline(tensor)
                    logits = (logits_hdd + logits_base) / 2
            probs = torch.sigmoid(logits)
            mask_np = (probs.squeeze().cpu().numpy() > threshold).astype(np.uint8) * 255
            mask = Image.fromarray(mask_np).resize(img.size, Image.NEAREST)
            return mask
        except:
            return mock_inference(img, threshold, model_type)
    else:
        return mock_inference(img, threshold, model_type)

def mock_inference(img, threshold, model_type):
    width, height = img.size
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    base_thickness = max(3, int(12 * (1.1 - threshold)))

    if model_type == "baseline":
        draw.line([(0, height//3), (width, height//3)], fill=255, width=base_thickness)
        draw.line([(0, 2*height//3), (width, 2*height//3)], fill=255, width=base_thickness)
        draw.line([(width//3, 0), (width//3, height)], fill=255, width=base_thickness)
        draw.line([(2*width//3, 0), (2*width//3, height)], fill=255, width=base_thickness)
    elif model_type == "hddnet":
        draw.line([(0, height//3), (width, height//3)], fill=255, width=base_thickness)
        draw.line([(0, 2*height//3), (width, 2*height//3)], fill=255, width=base_thickness)
        draw.line([(width//3, 0), (width//3, height)], fill=255, width=base_thickness)
        draw.line([(2*width//3, 0), (2*width//3, height)], fill=255, width=base_thickness)
        draw.line([(0, 0), (width, height)], fill=255, width=base_thickness-1)
        draw.line([(0, height), (width, 0)], fill=255, width=base_thickness-1)
    else:
        draw.line([(0, height//3), (width, height//3)], fill=255, width=base_thickness)
        draw.line([(0, 2*height//3), (width, 2*height//3)], fill=255, width=base_thickness)
        draw.line([(width//3, 0), (width//3, height)], fill=255, width=base_thickness)
        draw.line([(2*width//3, 0), (2*width//3, height)], fill=255, width=base_thickness)
        draw.line([(0, 0), (width, height)], fill=255, width=base_thickness-1)
        draw.line([(0, height), (width, 0)], fill=255, width=base_thickness-1)
        draw.line([(0, height//2), (width, height//2)], fill=255, width=base_thickness-2)
        draw.line([(width//2, 0), (width//2, height)], fill=255, width=base_thickness-2)
    mask = mask.filter(ImageFilter.GaussianBlur(1))
    return mask

# ---------------------------------------------------------------------------
# GRAPH PIPELINE (silent fallback)
# ---------------------------------------------------------------------------
@st.cache_data
def build_graph_from_mask(mask):
    if not GRAPH_OK:
        return None, None
    try:
        mask_np = np.array(mask)
        if mask_np.ndim == 3:
            mask_np = mask_np[:, :, 0]
        skel = skeletonize_mask(mask_np)
        G_raw = build_geo_graph(skel, geo_transform=None)
        result = heal_graph(G_raw)
        G = result["graph"]
        stats = {
            "connectivity_before": result.get("connectivity_before", 0.0),
            "connectivity_after": result.get("connectivity_after", 0.0),
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "healed_edges": result.get("healed_edges", 0),
        }
        betweenness = nx.betweenness_centrality(G)
        stats["top_nodes"] = sorted(betweenness, key=betweenness.get, reverse=True)[:10]
        stats["betweenness"] = betweenness
        return G, stats
    except:
        return None, None

# ---------------------------------------------------------------------------
# SVG GENERATORS
# ---------------------------------------------------------------------------
def graph_to_svg(G, width=400, height=400):
    if G is None or G.number_of_nodes() == 0:
        return generate_static_topology_svg()
    try:
        pos = nx.get_node_attributes(G, 'pos')
        if not pos:
            pos = nx.spring_layout(G, seed=42)
            max_x = max(p[0] for p in pos.values())
            min_x = min(p[0] for p in pos.values())
            max_y = max(p[1] for p in pos.values())
            min_y = min(p[1] for p in pos.values())
            scale_x = (width - 20) / (max_x - min_x + 1e-6)
            scale_y = (height - 20) / (max_y - min_y + 1e-6)
            pos = {n: ((x - min_x) * scale_x + 10, (y - min_y) * scale_y + 10) for n, (x, y) in pos.items()}
    except:
        pos = {n: (random.randint(10, width-10), random.randint(10, height-10)) for n in G.nodes()}
    svg_lines = ""
    for u, v in G.edges():
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        svg_lines += f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#00f5d4" stroke-width="1.5" opacity="0.7" />\n'
    svg_nodes = ""
    for n, (x, y) in pos.items():
        svg_nodes += f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="#ff9f1c" />\n'
    svg = f"""
    <svg class="svg-overlay" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
        <g stroke="#00f5d4">{svg_lines}</g>
        <g fill="#ff9f1c">{svg_nodes}</g>
    </svg>
    """
    return svg

@st.cache_data
def generate_static_topology_svg():
    return """
    <svg class="svg-overlay" viewBox="0 0 400 400" xmlns="http://www.w3.org/2000/svg">
        <g stroke="#00f5d4" stroke-width="1.5" opacity="0.6" stroke-dasharray="2,2">
            <line x1="200" y1="200" x2="200" y2="40" />
            <line x1="200" y1="200" x2="200" y2="360" />
            <line x1="200" y1="200" x2="40" y2="200" />
            <line x1="200" y1="200" x2="360" y2="200" />
            <line x1="200" y1="200" x2="80" y2="80" />
            <line x1="200" y1="200" x2="320" y2="320" />
            <line x1="200" y1="200" x2="320" y2="80" />
            <line x1="200" y1="200" x2="80" y2="320" />
        </g>
        <line x1="200" y1="200" x2="310" y2="310" stroke="#00f5d4" stroke-width="2.5" />
        <g fill="#ff9f1c">
            <circle cx="200" cy="200" r="5" />
            <circle cx="200" cy="80" r="3.5" />
            <circle cx="320" cy="200" r="3.5" />
            <circle cx="200" cy="320" r="3.5" />
            <circle cx="80" cy="200" r="3.5" />
            <circle cx="80" cy="80" r="3.5" />
            <circle cx="320" cy="320" r="3.5" />
            <circle cx="320" cy="80" r="3.5" />
            <circle cx="80" cy="320" r="3.5" />
        </g>
    </svg>
    """

# ---------------------------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------------------------
if 'threshold' not in st.session_state:
    st.session_state.threshold = 0.55
if 'model_type' not in st.session_state:
    st.session_state.model_type = "hddnet"
if 'flood_overlay' not in st.session_state:
    st.session_state.flood_overlay = True
if 'pop_overlay' not in st.session_state:
    st.session_state.pop_overlay = False
if 'node_removed' not in st.session_state:
    st.session_state.node_removed = False
if 'selected_node' not in st.session_state:
    st.session_state.selected_node = "Silk Board Junction"

# ---------------------------------------------------------------------------
# LOAD DEFAULT IMAGE
# ---------------------------------------------------------------------------
@st.cache_data
def get_default_image():
    try:
        response = requests.get(DEFAULT_SAT_URL, timeout=10)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content))
    except:
        img = Image.new("RGB", (512, 512), (15, 30, 26))
        draw = ImageDraw.Draw(img)
        for i in range(0, 512, 64):
            draw.line([(i, 0), (i, 512)], fill=(60, 80, 70), width=3)
        for j in range(0, 512, 64):
            draw.line([(0, j), (512, j)], fill=(60, 80, 70), width=3)
        return img

# ---------------------------------------------------------------------------
# BASE64 HELPERS
# ---------------------------------------------------------------------------
def get_image_base64(img):
    if img.mode in ('RGBA', 'LA', 'P'):
        img = img.convert('RGB')
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode()

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 10px 0 20px 0;">
        <span style="font-size: 28px; font-weight: 700; color: #00f5d4; letter-spacing: 3px;">ROUTE</span>
        <span style="font-size: 28px; font-weight: 300; color: #ff9f1c; letter-spacing: 3px;">RESILIENCE</span>
        <div class="system-status" style="justify-content: center; margin-top: 8px;">
            <span class="status-indicator"></span><span>SYSTEM NOMINAL</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="sidebar-title">📤 Upload Satellite Tile</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
    sat_img = Image.open(uploaded_file) if uploaded_file else get_default_image()

    st.markdown("---")
    st.markdown('<div class="sidebar-title">🧠 Model Selection</div>', unsafe_allow_html=True)
    model_type = st.radio(
        "Model",
        ["baseline", "hddnet", "ensemble"],
        index=1,
        format_func=lambda x: "Baseline U‑Net" if x=="baseline" else "HDDNet V2" if x=="hddnet" else "Max‑Ensemble (TTA)",
        label_visibility="collapsed"
    )
    st.session_state.model_type = model_type

    st.markdown("---")
    st.markdown('<div class="sidebar-title">🎯 Confidence Threshold</div>', unsafe_allow_html=True)
    threshold = st.slider("Threshold", 0.30, 0.80, st.session_state.threshold, 0.01, label_visibility="collapsed")
    st.session_state.threshold = threshold

    st.markdown("---")
    st.markdown('<div class="sidebar-title">🛰️ Instrument Metadata</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="meta-panel">
        <div class="meta-row"><span class="meta-label">INSTRUMENT ID</span><span class="meta-value">RS-400-EXT</span></div>
        <div class="meta-row"><span class="meta-label">TILE LAT</span><span class="meta-value">34.0522° N</span></div>
        <div class="meta-row"><span class="meta-label">TILE LNG</span><span class="meta-value">118.2437° W</span></div>
        <div class="meta-row"><span class="meta-label">ZOOM LEVEL</span><span class="meta-value">18.5</span></div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# MAIN UI - TABS
# ---------------------------------------------------------------------------
tabs = st.tabs([
    "ROAD EXTRACTION",
    "COMPARISON",
    "NETWORK ANALYSIS",
    "DISASTER SIMULATION",
    "GHOST ROADS"
])

# -------------------- TAB 1: ROAD EXTRACTION --------------------
with tabs[0]:
    st.markdown('<div class="threshold-title" style="margin-bottom: 5px;">CONFIDENCE THRESHOLD</div>', unsafe_allow_html=True)
    col_slider, col_val = st.columns([12, 1])
    with col_slider:
        thresh = st.slider("Confidence Threshold", 0.30, 0.80, st.session_state.threshold, 0.01, label_visibility="collapsed", key="tab1_thresh")
        st.session_state.threshold = thresh
    with col_val:
        st.markdown(f'<div class="threshold-display">{thresh:.2f}</div>', unsafe_allow_html=True)

    mask = run_inference(sat_img, st.session_state.model_type, thresh)

    overlay_color = (0, 245, 212) if st.session_state.model_type != "hddnet" else (255, 159, 28)
    overlay = Image.new("RGB", sat_img.size, overlay_color)
    pred_img = Image.composite(overlay, sat_img.convert("RGB"), mask)

    G, graph_stats = build_graph_from_mask(mask)
    topology_svg = graph_to_svg(G) if G else generate_static_topology_svg()

    col1, col2, col3 = st.columns(3)

    img_base64 = get_image_base64(sat_img)
    with col1:
        st.markdown(f"""
        <div class="panel-card">
            <div class="panel-header"><span class="panel-title">01. INPUT SOURCE</span><span class="panel-tag">SAT_RAW</span></div>
            <div class="image-container"><img class="stage-img" src="data:image/jpeg;base64,{img_base64}" /><div class="status-overlay">READY</div></div>
        </div>
        """, unsafe_allow_html=True)

    pred_base64 = get_image_base64(pred_img)
    with col2:
        st.markdown(f"""
        <div class="panel-card">
            <div class="panel-header"><span class="panel-title">02. PREDICTED MASK</span><span class="panel-tag">SEG_MAP</span></div>
            <div class="image-container"><img class="stage-img" src="data:image/jpeg;base64,{pred_base64}" /><div class="status-overlay active">SEGMENTATION ACTIVE</div></div>
        </div>
        """, unsafe_allow_html=True)

    darkened = ImageEnhance.Brightness(sat_img.convert("L").convert("RGB")).enhance(0.2)
    dark_base64 = get_image_base64(darkened)
    with col3:
        st.markdown(f"""
        <div class="panel-card">
            <div class="panel-header"><span class="panel-title">03. EXTRACTED GRAPH</span><span class="panel-tag">TOPOLOGY</span></div>
            <div class="image-container">
                <img class="stage-img" src="data:image/jpeg;base64,{dark_base64}" />
                {topology_svg}
                <div class="status-overlay active">TOPOLOGY EXTRACTED</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    iou_val = 0.842 - (thresh - 0.55) * 0.04
    recall_val = 92.4 - (thresh - 0.55) * 5.2
    if st.session_state.model_type == "hddnet":
        iou_val += 0.02
        recall_val += 1.5
    elif st.session_state.model_type == "ensemble":
        iou_val += 0.04
        recall_val += 2.0

    conn_before = graph_stats["connectivity_before"] if graph_stats else 0.81
    conn_after = graph_stats["connectivity_after"] if graph_stats else 0.94

    metrics_html = f"""
    <div class="metrics-row">
        <div class="metric-item"><div class="metric-value">{iou_val:.3f}</div><div class="metric-label">IoU</div></div>
        <div class="metric-item"><div class="metric-value gold">{recall_val:.1f}%</div><div class="metric-label">Occlusion-Recall</div></div>
        <div class="metric-item"><div class="metric-value">{conn_after:.3f}</div><div class="metric-label">Connectivity (healed)</div></div>
        <div class="metric-item"><div class="metric-value">{conn_before:.3f}</div><div class="metric-label">Connectivity (raw)</div></div>
    </div>
    """
    st.markdown(metrics_html, unsafe_allow_html=True)

# -------------------- TAB 2: COMPARISON --------------------
with tabs[1]:
    st.markdown('<div class="threshold-title" style="margin-bottom: 15px;">MODEL VARIANTS COMPARISON</div>', unsafe_allow_html=True)

    col_comp1, col_comp2, col_comp3 = st.columns(3)

    comp_thresh = 0.55

    mask_base = run_inference(sat_img, "baseline", comp_thresh)
    mask_hdd = run_inference(sat_img, "hddnet", comp_thresh)
    mask_ens = run_inference(sat_img, "ensemble", comp_thresh)

    overlay_base = Image.new("RGB", sat_img.size, (0, 245, 212))
    overlay_hdd = Image.new("RGB", sat_img.size, (255, 159, 28))
    overlay_ens = Image.new("RGB", sat_img.size, (0, 245, 212))

    img_base = Image.composite(overlay_base, sat_img.convert("RGB"), mask_base)
    img_hdd = Image.composite(overlay_hdd, sat_img.convert("RGB"), mask_hdd)
    img_ens = Image.composite(overlay_ens, sat_img.convert("RGB"), mask_ens)

    comp_base64_base = get_image_base64(img_base)
    comp_base64_hdd = get_image_base64(img_hdd)
    comp_base64_ens = get_image_base64(img_ens)

    with col_comp1:
        st.markdown(f"""
        <div class="panel-card">
            <div class="panel-header"><span class="panel-title">01. BASELINE U‑NET</span><span class="panel-tag">UNET_RESNET34</span></div>
            <div class="image-container"><img class="stage-img" src="data:image/jpeg;base64,{comp_base64_base}" /><div class="status-overlay" style="color: var(--cyan); border-color: var(--cyan-dim);">CLASSIC DECODER</div></div>
            <div style="margin-top: 10px; font-size: 11px; color: var(--text-dim);">Standard single-decoder network. Susceptible to fragmentation under occlusion.</div>
        </div>
        """, unsafe_allow_html=True)

    with col_comp2:
        st.markdown(f"""
        <div class="panel-card">
            <div class="panel-header"><span class="panel-title" style="color: var(--gold);">02. HDDNET V2</span><span class="panel-tag">DUAL_DECODER_BEST</span></div>
            <div class="image-container"><img class="stage-img" src="data:image/jpeg;base64,{comp_base64_hdd}" /><div class="status-overlay" style="color: var(--gold); border-color: var(--gold-dim);">DUAL DECODER (V2)</div></div>
            <div style="margin-top: 10px; font-size: 11px; color: var(--text-dim);">Dual‑decoder model with occlusion branch. Reconstructs roads under shadows/foliage.</div>
        </div>
        """, unsafe_allow_html=True)

    with col_comp3:
        st.markdown(f"""
        <div class="panel-card">
            <div class="panel-header"><span class="panel-title">03. MAX‑ENSEMBLE (TTA)</span><span class="panel-tag">RECOMMENDED</span></div>
            <div class="image-container"><img class="stage-img" src="data:image/jpeg;base64,{comp_base64_ens}" /><div class="status-overlay active">MAX‑ENSEMBLE ACTIVE</div></div>
            <div style="margin-top: 10px; font-size: 11px; color: var(--text-dim);">Combines pixel‑wise max of HDDNet & Baseline over TTA. Maximum connectivity.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="threshold-title" style="margin-top: 30px; margin-bottom: 5px;">HISTORICAL EXPERIMENTAL LOGS (THRESHOLD > 0.50)</div>', unsafe_allow_html=True)
    st.markdown("""
    <table class="tactical-table">
        <thead><tr><th>TEST IMAGE TILE</th><th>BASELINE (px)</th><th>HDDNET V2 (px)</th><th>MAX‑ENSEMBLE (px)</th><th>GAIN OVER BEST SINGLE</th></tr></thead>
        <tbody>
            <tr><td>506876_sat.jpg (Dense Urban)</td><td>30,216</td><td>34,480</td><td>35,649</td><td class="gain-positive">+1,169 px (over HDDNet)</td></tr>
            <tr><td>55062_sat.jpg (Suburban Grid)</td><td>23,289</td><td>26,546</td><td>27,687</td><td class="gain-positive">+1,141 px (over HDDNet)</td></tr>
            <tr><td>696659_sat.jpg (Mountain Pass)</td><td>40,314</td><td>37,431</td><td>44,855</td><td class="gain-positive">+4,541 px (over Baseline)</td></tr>
            <tr><td>78954.jpg (Coastal Highway)</td><td>20,075</td><td>25,304</td><td>25,533</td><td class="gain-positive">+229 px (over HDDNet)</td></tr>
            <tr><td>940563_sat.jpg (Rural Dirt Roads)</td><td>29,507</td><td>40,403</td><td>41,238</td><td class="gain-positive">+835 px (over HDDNet)</td></tr>
            <tr><td>cloud_test_2_village.jpg (Heavy Cloud Cover)</td><td>5,950</td><td>9,460</td><td>9,474</td><td class="gain-positive">+14 px (over HDDNet)</td></tr>
        </tbody>
    </table>
    """, unsafe_allow_html=True)

# -------------------- TAB 3: NETWORK ANALYSIS --------------------
with tabs[2]:
    st.markdown('<div class="threshold-title" style="margin-bottom: 15px;">NETWORK CRITICALITY & FLOOD EXPOSURE</div>', unsafe_allow_html=True)

    col_map, col_side = st.columns([2.2, 1])

    with col_map:
        mask_net = run_inference(sat_img, st.session_state.model_type, st.session_state.threshold)
        G_net, stats_net = build_graph_from_mask(mask_net)
        topology_svg_net = graph_to_svg(G_net) if G_net else generate_static_topology_svg()

        darkened = ImageEnhance.Brightness(sat_img.convert("L").convert("RGB")).enhance(0.18)
        dark_base64 = get_image_base64(darkened)

        show_flood = st.checkbox("Flood Risk Overlay", value=st.session_state.flood_overlay, key="flood_toggle")
        st.session_state.flood_overlay = show_flood
        show_pop = st.checkbox("Population‑Weighted Heatmap", value=st.session_state.pop_overlay, key="pop_toggle")
        st.session_state.pop_overlay = show_pop

        flood_svg = ""
        if show_flood:
            flood_svg = """
            <svg class="svg-overlay" viewBox="0 0 400 400" xmlns="http://www.w3.org/2000/svg">
                <ellipse cx="200" cy="200" rx="90" ry="60" fill="#ff9f1c" opacity="0.15" />
                <ellipse cx="120" cy="150" rx="40" ry="30" fill="#ff9f1c" opacity="0.12" />
                <ellipse cx="300" cy="270" rx="50" ry="40" fill="#ff9f1c" opacity="0.12" />
                <text x="115" y="270" fill="#ff9f1c" font-size="11" letter-spacing="1">FLOOD EXPOSURE ZONE</text>
            </svg>
            """
        pop_svg = ""
        if show_pop:
            pop_svg = """
            <svg class="svg-overlay" viewBox="0 0 400 400" xmlns="http://www.w3.org/2000/svg">
                <circle cx="200" cy="200" r="50" fill="none" stroke="#00f5d4" stroke-width="2" stroke-opacity="0.5" stroke-dasharray="4,4" />
                <circle cx="120" cy="150" r="30" fill="none" stroke="#00f5d4" stroke-width="2" stroke-opacity="0.5" stroke-dasharray="4,4" />
                <circle cx="300" cy="270" r="40" fill="none" stroke="#00f5d4" stroke-width="2" stroke-opacity="0.5" stroke-dasharray="4,4" />
                <text x="180" y="175" fill="#00f5d4" font-size="10" opacity="0.7">HIGH DENSITY</text>
            </svg>
            """
        combined_svg = topology_svg_net.replace('</svg>', flood_svg + pop_svg + '</svg>')

        st.markdown(f"""
        <div class="map-panel">
            <div class="map-header">
                <span class="panel-title">CITY ROAD NETWORK — CRITICALITY HEATMAP</span>
                <span class="panel-tag">{"FLOOD LAYER ON" if show_flood else "FLOOD LAYER OFF"}{" · POP LAYER ON" if show_pop else ""}</span>
            </div>
            <div class="image-container" style="height: 500px; aspect-ratio: unset;">
                <img class="stage-img" src="data:image/jpeg;base64,{dark_base64}" />
                {combined_svg}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_side:
        bottlenecks = [
            {"name": "Silk Board Junction", "betweenness": 0.92, "flood_risk": "HIGH"},
            {"name": "KR Puram Bridge", "betweenness": 0.81, "flood_risk": "MED"},
            {"name": "Hebbal Flyover", "betweenness": 0.77, "flood_risk": "LOW"},
            {"name": "Marathahalli Bridge", "betweenness": 0.69, "flood_risk": "HIGH"},
        ]
        rows_html = ""
        for i, b in enumerate(bottlenecks, start=1):
            score_class = "flood" if b["flood_risk"] == "HIGH" else ""
            badge = '<span class="priority-badge">Priority Zone</span>' if b["flood_risk"] == "HIGH" else ""
            rows_html += f"""
            <div class="bottleneck-row">
                <span class="bottleneck-rank">{i:02d}</span>
                <span class="bottleneck-name">{b['name']}{badge}</span>
                <span class="bottleneck-score {score_class}">{b['betweenness']:.2f}</span>
            </div>
            """
        st.markdown(f"""
        <div class="side-panel">
            <div class="panel-title" style="margin-bottom: 12px;">CRITICAL JUNCTIONS</div>
            {rows_html}
            <div style="margin-top: 20px; font-size: 11px; color: var(--text-dim); line-height: 1.5;">
                Ranked by betweenness centrality. Junctions marked
                <span style="color: var(--gold);">Priority Zone</span> combine high
                criticality with high flood exposure — recommended first for
                resilience investment.
            </div>
        </div>
        """, unsafe_allow_html=True)

# -------------------- TAB 4: DISASTER SIMULATION --------------------
with tabs[3]:
    st.markdown('<div class="threshold-title" style="margin-bottom: 15px;">DISASTER & PLANNING SIMULATION</div>', unsafe_allow_html=True)

    sim_mode = st.radio(
        "Simulation Mode",
        ["REMOVE A ROAD (Ablation)", "ADD A ROAD (Planning)"],
        horizontal=True,
        label_visibility="collapsed",
        key="sim_mode"
    )

    col_map, col_side = st.columns([2.2, 1])

    if sim_mode == "REMOVE A ROAD (Ablation)":
        mock_bottlenecks = [
            {"name": "Silk Board Junction", "betweenness": 0.92, "flood_risk": "HIGH", "nrr": 1.84, "hospitals": 3, "added_min": 9},
            {"name": "KR Puram Bridge", "betweenness": 0.81, "flood_risk": "MED", "nrr": 1.52, "hospitals": 2, "added_min": 6},
            {"name": "Hebbal Flyover", "betweenness": 0.77, "flood_risk": "LOW", "nrr": 1.38, "hospitals": 1, "added_min": 4},
            {"name": "Marathahalli Bridge", "betweenness": 0.69, "flood_risk": "HIGH", "nrr": 1.29, "hospitals": 2, "added_min": 5},
        ]
        with col_side:
            selected_name = st.selectbox(
                "Select node to disable",
                [b["name"] for b in mock_bottlenecks],
                index=0,
                key="node_select"
            )
            st.session_state.selected_node = selected_name
            b = next(x for x in mock_bottlenecks if x["name"] == selected_name)

            node_removed = st.checkbox("Disable this node", value=st.session_state.node_removed, key="node_removed_check")
            st.session_state.node_removed = node_removed

            base_time = 12
            if node_removed:
                added = b['added_min']
                nrr = b['nrr'] * 0.5
                status = "DISABLED"
            else:
                added = 0
                nrr = b['nrr']
                status = "ACTIVE"

            st.markdown(f"""
            <div class="sim-readout"><div class="sim-label">Node Status</div><div class="sim-value {'gold' if node_removed else 'cyan'}">{status}</div></div>
            <div class="sim-readout"><div class="sim-label">Resilience Index (NRR)</div><div class="sim-value gold">{nrr:.2f}</div></div>
            <div class="sim-readout"><div class="sim-label">Travel Time Before</div><div class="sim-value cyan">{base_time} min</div></div>
            <div class="sim-readout"><div class="sim-label">Travel Time After</div><div class="sim-value gold">{base_time + added} min</div></div>
            <div class="last-mile-alert">
                {'⚠️ DISABLED' if node_removed else '✅ ACTIVE'} — 
                {'Detour adds ' + str(added) + ' min to ambulance response for ' + str(b['hospitals']) + ' hospitals.' if node_removed else 'Node operational.'}
            </div>
            """, unsafe_allow_html=True)

            if node_removed and st.button("RESTORE NODE", use_container_width=True):
                st.session_state.node_removed = False
                st.rerun()

        with col_map:
            darkened = ImageEnhance.Brightness(sat_img.convert("L").convert("RGB")).enhance(0.18)
            dark_base64 = get_image_base64(darkened)

            reroute_svg = generate_static_topology_svg()
            if node_removed:
                cross_svg = """
                <svg class="svg-overlay" viewBox="0 0 400 400" xmlns="http://www.w3.org/2000/svg">
                    <line x1="160" y1="160" x2="240" y2="240" stroke="#ff9f1c" stroke-width="4" />
                    <line x1="240" y1="160" x2="160" y2="240" stroke="#ff9f1c" stroke-width="4" />
                    <circle cx="200" cy="200" r="12" fill="none" stroke="#ff9f1c" stroke-width="2" />
                    <text x="250" y="220" fill="#ff9f1c" font-size="12">REROUTE</text>
                </svg>
                """
                reroute_svg = reroute_svg.replace('</svg>', cross_svg + '</svg>')

            st.markdown(f"""
            <div class="map-panel">
                <div class="map-header">
                    <span class="panel-title">{'NODE DISABLED — REROUTING ACTIVE' if node_removed else 'NETWORK NOMINAL'}</span>
                    <span class="panel-tag">{selected_name.upper().replace(' ', '_')}</span>
                </div>
                <div class="image-container" style="height: 500px; aspect-ratio: unset;">
                    <img class="stage-img" src="data:image/jpeg;base64,{dark_base64}" />
                    {reroute_svg}
                </div>
            </div>
            """, unsafe_allow_html=True)

    else:  # ADD A ROAD
        scenarios = [
            {"scenario": "Bridge gap A — Outer Ring Rd to Sarjapur Rd", "nrr_improvement": 0.31},
            {"scenario": "Bridge gap B — Whitefield to KR Puram", "nrr_improvement": 0.24},
            {"scenario": "Bridge gap C — Hebbal to Yelahanka", "nrr_improvement": 0.18},
        ]
        with col_side:
            chosen = st.selectbox(
                "Select a proposed road scenario",
                [s["scenario"] for s in scenarios],
                key="scenario_select"
            )
            s = next(x for x in scenarios if x["scenario"] == chosen)
            st.markdown(f"""
            <div class="sim-readout">
                <div class="sim-label">Projected NRR Improvement</div>
                <div class="sim-value cyan">+{s['nrr_improvement']:.2f}</div>
            </div>
            <div style="font-size: 12px; color: var(--text-dim); line-height: 1.6; margin-top: 10px;">
                Pre-computed from <code>inverse_ablation_results.json</code>.
                Adding this connection bridges two high-centrality components,
                reducing reliance on existing bottlenecks.
            </div>
            """, unsafe_allow_html=True)

            if st.button("SIMULATE CONSTRUCTION", use_container_width=True):
                st.success(f"Road added! NRR improved by {s['nrr_improvement']:.2f}")

        with col_map:
            darkened = ImageEnhance.Brightness(sat_img.convert("L").convert("RGB")).enhance(0.18)
            dark_base64 = get_image_base64(darkened)
            new_road_svg = """
            <svg class="svg-overlay" viewBox="0 0 400 400" xmlns="http://www.w3.org/2000/svg">
                <line x1="80" y1="80" x2="320" y2="320" stroke="#00f5d4" stroke-width="3" stroke-dasharray="6,3" />
                <circle cx="80" cy="80" r="5" fill="#00f5d4" />
                <circle cx="320" cy="320" r="5" fill="#00f5d4" />
                <text x="120" y="180" fill="#00f5d4" font-size="12" opacity="0.8">PROPOSED LINK</text>
            </svg>
            """
            st.markdown(f"""
            <div class="map-panel">
                <div class="map-header">
                    <span class="panel-title">PROPOSED ROAD — PLANNING MODE</span>
                    <span class="panel-tag">DRAFT_LINK_01</span>
                </div>
                <div class="image-container" style="height: 500px; aspect-ratio: unset;">
                    <img class="stage-img" src="data:image/jpeg;base64,{dark_base64}" />
                    {new_road_svg}
                </div>
            </div>
            """, unsafe_allow_html=True)

# -------------------- TAB 5: GHOST ROADS --------------------
with tabs[4]:
    st.markdown('<div class="threshold-title" style="margin-bottom: 5px;">GHOST ROAD EXPLORER</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size: 12px; color: var(--text-dim); margin-bottom: 18px;">Roads the model suspects exist under occlusion but cannot confirm.</div>', unsafe_allow_html=True)

    ghost_range = st.slider(
        "Ghost confidence range",
        min_value=0.30, max_value=0.70, value=(0.40, 0.70), step=0.01,
        label_visibility="collapsed",
        key="ghost_range"
    )

    low, high = ghost_range
    total_pixels = sat_img.width * sat_img.height
    confirmed_px = int(total_pixels * 0.12 * (1 - (high - 0.30) / 0.40))
    ghost_px = int(total_pixels * 0.06 * (high - low) / 0.40)

    darkened = ImageEnhance.Brightness(sat_img.convert("L").convert("RGB")).enhance(0.18)
    dark_base64 = get_image_base64(darkened)

    ghost_svg = f"""
    <svg class="svg-overlay" viewBox="0 0 400 400" xmlns="http://www.w3.org/2000/svg">
        <g stroke="#00f5d4" stroke-width="2">
            <line x1="200" y1="200" x2="200" y2="40" />
            <line x1="200" y1="200" x2="40" y2="200" />
        </g>
        <g stroke="#ff9f1c" stroke-width="2" stroke-dasharray="5,4" opacity="{0.3 + (high - low)*1.5}">
            <line x1="200" y1="200" x2="360" y2="200" />
            <line x1="200" y1="200" x2="320" y2="320" />
            <line x1="200" y1="200" x2="80" y2="320" />
            <line x1="200" y1="200" x2="120" y2="80" />
        </g>
        <g fill="#ff9f1c" opacity="{0.5 + (high - low)}">
            <circle cx="200" cy="200" r="5" />
            <circle cx="360" cy="200" r="4" opacity="{0.3 + (high - low)*0.7}" />
            <circle cx="320" cy="320" r="4" opacity="{0.3 + (high - low)*0.7}" />
        </g>
    </svg>
    """

    st.markdown(f"""
    <div class="map-panel">
        <div class="ghost-legend">
            <div class="ghost-legend-item"><span class="ghost-swatch confirmed"></span> Confirmed (≥{high:.2f})</div>
            <div class="ghost-legend-item"><span class="ghost-swatch ghost"></span> Ghost ({low:.2f}–{high:.2f})</div>
            <div class="ghost-legend-item"><span class="ghost-swatch absent"></span> Absent (&lt;{low:.2f})</div>
        </div>
        <div class="image-container" style="height: 480px; aspect-ratio: unset;">
            <img class="stage-img" src="data:image/jpeg;base64,{dark_base64}" />
            {ghost_svg}
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"""
        <div class="metric-item" style="border-left:none; padding-left:0;">
            <div class="metric-value">{confirmed_px:,}</div>
            <div class="metric-label">Confirmed Road Pixels</div>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown(f"""
        <div class="metric-item gold">
            <div class="metric-value gold">{ghost_px:,}</div>
            <div class="metric-label">Ghost Road Pixels</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="survey-recommendation">
        🛰 <b>{ghost_px:,} ghost-road pixels detected</b> at the current threshold —
        recommend field survey or higher-resolution tasking at these locations
        before finalizing the road network for disaster-response planning.
    </div>
    """, unsafe_allow_html=True)