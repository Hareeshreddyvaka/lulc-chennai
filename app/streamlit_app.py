"""
🛰️ Multi-Temporal Land Use / Land Cover (LULC) Analysis Dashboard
Siamese Swin-Tiny + Bottleneck Self-Attention (BSA) + ASPP + Cross-Transformer
Zero-Lag Interactive Explorer for Chennai & Zero-Shot Cross-City Transfer (Hyderabad, Bengaluru, Pune)

Run locally:
    streamlit run 6_streamlit_app.py
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    import streamlit as st
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "-q", "streamlit"])
    import streamlit as st

try:
    import timm
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "-q", "timm"])
    import timm

# ─────────────────────────────────────────────────────────────────────────────
# 1. Page Configuration & Custom CSS
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="LULC Analysis Dashboard",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-val {
        font-size: 1.4rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 4px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Constants & Palette Definitions
# ─────────────────────────────────────────────────────────────────────────────

NUM_CLASSES = 8
CLASS_NAMES = ["Trees", "Shrubland", "Grassland", "Cropland", "Built-up", "Bare/Sparse", "Water", "Wetland"]

LULC_COLORS = np.array([
    [0, 100, 0],      # Trees - Dark Green
    [139, 90, 43],    # Shrubland - Brown
    [124, 252, 0],    # Grassland - Lime Green
    [255, 215, 0],    # Cropland - Gold
    [255, 0, 0],      # Built-up - Red
    [210, 180, 140],  # Bare/Sparse - Tan
    [0, 0, 255],      # Water - Blue
    [0, 128, 128],    # Wetland - Teal
], dtype=np.float32) / 255.0

LULC_HEX = ["#006400", "#8B5A2B", "#7CFC00", "#FFD700", "#FF0000", "#D2B48C", "#0000FF", "#008080"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = {
    "img_size": 128,
    "in_chans": 9,
    "num_classes": NUM_CLASSES,
    "backbone": "swin_tiny_patch4_window7_224",
    "attn_heads": 8,
    "cross_attn_heads": 8,
    "num_layers": 3,
}

# ─────────────────────────────────────────────────────────────────────────────
# 3. Model Architecture
# ─────────────────────────────────────────────────────────────────────────────

class UNetDecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(True)
        )
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))

class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch, rates=[1, 2, 4]):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(True))
        ])
        for r in rates:
            self.stages.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=r, dilation=r, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(True)
            ))
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(True)
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * (len(rates) + 2), out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(True)
        )
    def forward(self, x):
        feats = [s(x) for s in self.stages]
        p = F.interpolate(self.pool(x), size=x.shape[-2:], mode="bilinear", align_corners=False)
        feats.append(p)
        return self.project(torch.cat(feats, dim=1))

def widen_patch_embed(backbone, in_chans):
    proj = backbone.patch_embed.proj
    old_w = proj.weight.data
    new_proj = nn.Conv2d(in_chans, proj.out_channels, proj.kernel_size, proj.stride, proj.padding, bias=proj.bias is not None)
    with torch.no_grad():
        new_proj.weight[:, :3] = old_w
        if in_chans > 3:
            new_proj.weight[:, 3:] = old_w.mean(1, keepdim=True).repeat(1, in_chans - 3, 1, 1)
        if proj.bias is not None:
            new_proj.bias[:] = proj.bias.data
    backbone.patch_embed.proj = new_proj

def to_nchw(feat, chs):
    if feat.dim() == 4 and feat.shape[1] != feat.shape[-1] and feat.shape[-1] in chs:
        feat = feat.permute(0, 3, 1, 2).contiguous()
    return feat

class BSALayer(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
    def forward(self, x):
        h = self.n1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        h = self.n2(x)
        return x + self.mlp(h)

class BottleneckSelfAttention(nn.Module):
    def __init__(self, dim, heads=8, num_layers=1):
        super().__init__()
        self.layers = nn.ModuleList([BSALayer(dim, heads) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        B, C, H, W = x.shape
        t = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        for l in self.layers:
            t = l(t)
        return self.norm(t).reshape(B, H, W, C).permute(0, 3, 1, 2)

class CrossTransformerBlock(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.na = nn.LayerNorm(dim)
        self.nb = nn.LayerNorm(dim)
        self.attn12 = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.attn21 = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.nc = nn.LayerNorm(dim)
        self.nd = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim * 2, dim)
    def forward(self, f1, f2):
        B, C, H, W = f1.shape
        t1 = f1.permute(0, 2, 3, 1).reshape(B, H * W, C)
        t2 = f2.permute(0, 2, 3, 1).reshape(B, H * W, C)
        n1, n2 = self.na(t1), self.nb(t2)
        c1 = t1 + self.attn12(n1, n2, n2, need_weights=False)[0]
        c2 = t2 + self.attn21(n2, n1, n1, need_weights=False)[0]
        out = self.proj(torch.cat([self.nc(c1), self.nd(c2)], dim=-1))
        return out.reshape(B, H, W, C).permute(0, 3, 1, 2)

class SiameseSwinRefined(nn.Module):
    def __init__(self, num_classes=8, in_chans=9, img_size=128,
                 backbone_name="swin_tiny_patch4_window7_224",
                 pretrained=False, attn_heads=8, cross_heads=8, num_layers=3):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, features_only=True, img_size=img_size, in_chans=3)
        widen_patch_embed(self.backbone, in_chans)
        self.chs = self.backbone.feature_info.channels()
        self.self_attn = BottleneckSelfAttention(self.chs[3], attn_heads, num_layers)
        self.aspp = ASPP(self.chs[3], self.chs[3], rates=[1, 2, 4])
        self.cross_attn = CrossTransformerBlock(self.chs[3], cross_heads)
        self.dec3 = UNetDecoderBlock(self.chs[3], self.chs[2], self.chs[2])
        self.dec2 = UNetDecoderBlock(self.chs[2], self.chs[1], self.chs[1])
        self.dec1 = UNetDecoderBlock(self.chs[1], self.chs[0], self.chs[0])
        self.lulc_head = nn.Sequential(
            nn.Conv2d(self.chs[0], self.chs[0] // 2, 3, padding=1),
            nn.BatchNorm2d(self.chs[0] // 2), nn.ReLU(True),
            nn.Conv2d(self.chs[0] // 2, num_classes, 1)
        )
        self.change_head = nn.Sequential(
            nn.Conv2d(self.chs[3], self.chs[3] // 2, 3, padding=1),
            nn.BatchNorm2d(self.chs[3] // 2), nn.ReLU(True),
            nn.Conv2d(self.chs[3] // 2, 1, 1)
        )
        self.img_size = img_size

    def encode(self, x):
        return [to_nchw(f, self.chs) for f in self.backbone(x)]

    def forward(self, x1, x2):
        f1, f2 = self.encode(x1), self.encode(x2)
        f1a, f2a = self.self_attn(f1[3]), self.self_attn(f2[3])
        f1p, f2p = self.aspp(f1a), self.aspp(f2a)
        s1, s2, s3, _ = f2
        d3 = self.dec3(f2p, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        lulc = F.interpolate(self.lulc_head(d1), (self.img_size, self.img_size), mode="bilinear", align_corners=False)
        fused = self.cross_attn(f1p, f2p)
        change = F.interpolate(self.change_head(fused), (self.img_size, self.img_size), mode="bilinear", align_corners=False)
        return lulc, change.squeeze(1)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Fast Spectral Processing & Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

def to_nine_band(patch_5band, eps=1e-6):
    b2, b3, b4, b8, b11 = patch_5band[:5]
    ndvi = (b8 - b4) / (b8 + b4 + eps)
    ndwi = (b3 - b8) / (b3 + b8 + eps)
    ndbi = (b11 - b8) / (b11 + b8 + eps)
    mndwi = (b3 - b11) / (b3 + b11 + eps)
    return np.stack([b2, b3, b4, b8, b11, ndvi, ndwi, ndbi, mndwi], axis=0).astype(np.float32)

def normalize_bands(bands):
    out = bands.copy()
    out[:5] = np.clip(out[:5], 0, 1)
    out[5:] = (np.clip(out[5:], -1, 1) + 1) / 2
    return out

def label_to_rgb(label_map):
    rgb = np.zeros((*label_map.shape, 3), dtype=np.float32)
    for c in range(NUM_CLASSES):
        rgb[label_map == c] = LULC_COLORS[c]
    return rgb

def make_rgb(img_5band):
    rgb = np.stack([img_5band[2], img_5band[1], img_5band[0]], axis=-1)
    lo, hi = np.percentile(rgb, 2), np.percentile(rgb, 98)
    return np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)

def get_semantic_transitions(pred1, pred2, sem_change):
    p1 = pred1[sem_change == 1]
    p2 = pred2[sem_change == 1]
    
    transitions = {
        "Urban Expansion (Veg → Built-up)": 0,
        "Land Clearing (Veg → Bare)": 0,
        "Water/Wetland Loss": 0,
        "Construction (Bare → Built-up)": 0,
        "Regrowth (Bare/Built → Veg)": 0,
        "Other Changes": 0
    }
    
    for c1, c2 in zip(p1, p2):
        is_veg1 = c1 in [0,1,2,3]
        is_veg2 = c2 in [0,1,2,3]
        
        if is_veg1 and c2 == 4:
            transitions["Urban Expansion (Veg → Built-up)"] += 1
        elif is_veg1 and c2 == 5:
            transitions["Land Clearing (Veg → Bare)"] += 1
        elif c1 in [6,7] and c2 in [4,5]:
            transitions["Water/Wetland Loss"] += 1
        elif c1 == 5 and c2 == 4:
            transitions["Construction (Bare → Built-up)"] += 1
        elif c1 in [4,5] and is_veg2:
            transitions["Regrowth (Bare/Built → Veg)"] += 1
        else:
            transitions["Other Changes"] += 1
            
    total_pixels = pred1.size
    results = {}
    for k, v in transitions.items():
        if v > 0:
            pct = (v / total_pixels) * 100
            if pct >= 0.1:
                results[k] = f"{pct:.2f}%"
    return results

# ─────────────────────────────────────────────────────────────────────────────
# 5. Cached Model & Zero-Lag Memory-Mapped Data Loading
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    """Load model once into memory/GPU and keep cached."""
    model = SiameseSwinRefined(
        num_classes=NUM_CLASSES, in_chans=CONFIG["in_chans"], img_size=CONFIG["img_size"],
        backbone_name=CONFIG["backbone"], pretrained=False,
        attn_heads=CONFIG["attn_heads"], cross_heads=CONFIG["cross_attn_heads"],
        num_layers=CONFIG["num_layers"]
    ).to(device)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base_dir, "training_final_5", "results (1)", "best_model_final_5.pth"),
        os.path.join(base_dir, "best_model_final_5.pth"),
        r".\project_folder\training_final_5\results (1)\best_model_final_5.pth",
        "/kaggle/input/datasets/user/lulc-model/best_model_final_5.pth",
        "/kaggle/input/best_model_final_5.pth",
    ]
    loaded_ckpt = None
    for c in candidates:
        if os.path.exists(c):
            try:
                ckpt = torch.load(c, map_location=device, weights_only=False)
                state = ckpt.get("model_state_dict", ckpt)
                model.load_state_dict(state)
                loaded_ckpt = c
                break
            except Exception as e:
                print(f"Error loading {c}: {e}")

    model.eval()
    return model, loaded_ckpt

@st.cache_resource
def get_data_handles():
    """
    Open .npy files via memory-mapping (mmap_mode='r').
    Virtually 0 MB RAM overhead and opens in < 2ms!
    Completely eliminates startup/patch selection lag.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    search_dirs = [
        os.path.join(base_dir, "lulc_chennai", "patches"),
        os.path.join(base_dir, "lulc_chennai"),
        r".\project_folder\lulc_chennai\patches",
        r".\project_folder\lulc_chennai",
        "/kaggle/input/datasets/user/lulc-chennai",
        "/kaggle/input/datasets/user/lulc-chennai/patches",
        base_dir,
    ]

    # Find Chennai patches
    chennai_patches = {}
    chennai_labels = None
    for y in [2017, 2019, 2024, 2025]:
        for d in search_dirs:
            p = os.path.join(d, f"patches_{y}.npy")
            if os.path.exists(p):
                try:
                    chennai_patches[y] = np.load(p, mmap_mode="r")
                    break
                except Exception:
                    pass

    for d in search_dirs:
        lp = os.path.join(d, "label_patches.npy")
        if os.path.exists(lp):
            try:
                chennai_labels = np.load(lp, mmap_mode="r")
                break
            except Exception:
                pass

    # Find Cross-City datasets (Hyderabad, Bengaluru, Pune, etc.)
    city_search_dirs = [
        base_dir,
        r".\project_folder",
        "/kaggle/input/datasets/user/lulc-chennai",
    ]

    cross_cities = {}
    city_meta = {
        "Hyderabad": {
            "p19": "hyderabad_patches_2019.npy",
            "p24": "hyderabad_patches_2024.npy",
            "psingle": "hyderabad_patches.npy",
            "desc": "Telangana • Hussain Sagar & Core Urban Metro (~105 km²)"
        },
        "Bengaluru": {
            "p19": "bengaluru_patches_2019.npy",
            "p24": "bengaluru_patches_2024.npy",
            "psingle": "bengaluru_patches.npy",
            "desc": "Karnataka • Bellandur Lake & Tech Corridor (~105 km²)"
        },
        "Pune": {
            "p19": "pune_patches_2019.npy",
            "p24": "pune_patches_2024.npy",
            "psingle": "pune_patches.npy",
            "desc": "Maharashtra • Mula-Mutha River & Urban Foothills (~105 km²)"
        },
    }

    for city_name, meta in city_meta.items():
        t1_arr, t2_arr = None, None
        for d in city_search_dirs:
            p19_f = os.path.join(d, meta["p19"])
            p24_f = os.path.join(d, meta["p24"])
            ps_f = os.path.join(d, meta["psingle"])

            if os.path.exists(p19_f) and os.path.exists(p24_f):
                try:
                    t1_arr = np.load(p19_f, mmap_mode="r")
                    t2_arr = np.load(p24_f, mmap_mode="r")
                    break
                except Exception:
                    pass
            elif os.path.exists(ps_f):
                try:
                    t1_arr = np.load(ps_f, mmap_mode="r")
                    t2_arr = t1_arr
                    break
                except Exception:
                    pass

        if t1_arr is not None:
            cross_cities[city_name] = {
                "t1": t1_arr,
                "t2": t2_arr,
                "desc": meta["desc"]
            }

    return chennai_patches, chennai_labels, cross_cities

# ─────────────────────────────────────────────────────────────────────────────
# 6. Main App Header & Data Check
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="main-header">🌍 Land Use & Land Cover Change Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated Satellite Monitoring (10m Resolution)</div>', unsafe_allow_html=True)

model, ckpt_path = load_model()
chennai_patches, chennai_labels, cross_cities = get_data_handles()

# Sidebar: Status & Mode Selection
st.sidebar.markdown("### ⚙️ System Status")
if ckpt_path:
    st.sidebar.success(f"**Model:** `{os.path.basename(ckpt_path)}` ({device.type.upper()})")
else:
    st.sidebar.warning("⚠️ No weights found (using initialized model)")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🗺️ Select Study Domain")

available_modes = []
if chennai_patches and len(chennai_patches) > 0:
    available_modes.append("🏙️ Chennai Study Area (~3,400 Patches)")
if cross_cities and len(cross_cities) > 0:
    available_modes.append("🌐 Cross-City Generalization (3 Cities Demo)")

if not available_modes:
    st.error("No patch data found in workspace. Please verify .npy files.")
    st.stop()

selected_mode = st.sidebar.radio("Navigation", available_modes, index=0)

# ─────────────────────────────────────────────────────────────────────────────
# 7. Domain 1: Chennai Multi-Temporal (Primary Benchmark)
# ─────────────────────────────────────────────────────────────────────────────

if selected_mode == "🏙️ Chennai Study Area (~3,400 Patches)":
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎛️ Chennai Configuration")

    years_avail = sorted(list(chennai_patches.keys()))
    col_y1, col_y2 = st.sidebar.columns(2)
    with col_y1:
        idx_default_t1 = years_avail.index(2019) if 2019 in years_avail else 0
        year1 = st.selectbox("T1 (Initial Year)", years_avail, index=idx_default_t1)
    with col_y2:
        valid_t2 = [y for y in years_avail if y != year1]
        idx_default_t2 = valid_t2.index(2024) if 2024 in valid_t2 else len(valid_t2) - 1
        year2 = st.selectbox("T2 (Final Year)", valid_t2 if valid_t2 else years_avail, index=max(0, idx_default_t2))

    n_total_patches = chennai_patches[year1].shape[0]

    # Patch Selector (Synced Slider + Direct Input)
    st.sidebar.markdown(f"**Patch Selection** (Total: {n_total_patches:,d} patches)")
    patch_id = st.sidebar.number_input(f"Enter Patch ID [0 - {n_total_patches - 1}]", min_value=0, max_value=n_total_patches - 1, value=1257, step=1)
    patch_slider = st.sidebar.slider("Quick Navigate", 0, n_total_patches - 1, int(patch_id), key="chennai_slider")
    if patch_slider != patch_id:
        patch_id = patch_slider

    confidence_tau = st.sidebar.slider("Confidence Threshold (τ)", min_value=0.30, max_value=0.95, value=0.60, step=0.05)

    # Fast Memory-Mapped Slice (< 1ms)
    t0 = time.time()
    patch_t1_raw = np.array(chennai_patches[year1][patch_id])
    patch_t2_raw = np.array(chennai_patches[year2][patch_id])

    t1_norm = normalize_bands(to_nine_band(patch_t1_raw))
    t2_norm = normalize_bands(to_nine_band(patch_t2_raw))

    tensor_t1 = torch.from_numpy(t1_norm).unsqueeze(0).to(device)
    tensor_t2 = torch.from_numpy(t2_norm).unsqueeze(0).to(device)

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            lulc1_logits, _ = model(tensor_t1, tensor_t1)
            lulc2_logits, chg_logits = model(tensor_t1, tensor_t2)

        pred1 = lulc1_logits.argmax(1).cpu().numpy()[0]
        pred2 = lulc2_logits.argmax(1).cpu().numpy()[0]
        prob1 = F.softmax(lulc1_logits, dim=1).max(dim=1)[0].cpu().numpy()[0]
        prob2 = F.softmax(lulc2_logits, dim=1).max(dim=1)[0].cpu().numpy()[0]
        chg_prob = torch.sigmoid(chg_logits).cpu().numpy()[0]

    inference_ms = (time.time() - t0) * 1000

    # Semantic change mask: only where class changed and both predictions exceed tau
    sem_change = ((pred1 != pred2) & (prob1 >= confidence_tau) & (prob2 >= confidence_tau)).astype(np.uint8)
    sem_change_pct = sem_change.mean() * 100

    dom_c1 = CLASS_NAMES[np.argmax([(pred1 == c).sum() for c in range(NUM_CLASSES)])]
    dom_c2 = CLASS_NAMES[np.argmax([(pred2 == c).sum() for c in range(NUM_CLASSES)])]

    # Metrics Summary Row
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Selected Patch</div><div class="metric-val">#{patch_id}</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Time Horizon</div><div class="metric-val">{year1} → {year2}</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Semantic Change Area</div><div class="metric-val">{sem_change_pct:.2f}%</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Dominant Shift</div><div class="metric-val" style="font-size:1.1rem;">{dom_c1} → {dom_c2}</div></div>', unsafe_allow_html=True)
    with m5:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Inference Latency</div><div class="metric-val">{inference_ms:.1f} ms</div></div>', unsafe_allow_html=True)

    st.write("")

    # Visual Multi-Panel Layout
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader(f"📅 T1: Baseline ({year1})")
        st.image(make_rgb(patch_t1_raw), caption=f"Sentinel-2 Natural RGB ({year1})", use_container_width=True)
        st.image(label_to_rgb(pred1), caption=f"Predicted LULC Map (Mean Conf: {prob1.mean()*100:.1f}%)", use_container_width=True)

        # Class Distribution Bar Chart
        fig1, ax1 = plt.subplots(figsize=(5, 3))
        pcts1 = [(pred1 == c).mean() * 100 for c in range(NUM_CLASSES)]
        ax1.barh(CLASS_NAMES, pcts1, color=[LULC_COLORS[c] for c in range(NUM_CLASSES)], edgecolor="black", linewidth=0.5)
        ax1.set_xlim(0, 100)
        ax1.set_xlabel("Area Share (%)", fontsize=8)
        ax1.set_title(f"{year1} Class Distribution", fontsize=9, fontweight="bold")
        ax1.grid(axis="x", alpha=0.3)
        fig1.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

    with col2:
        st.subheader(f"📅 T2: Monitoring ({year2})")
        st.image(make_rgb(patch_t2_raw), caption=f"Sentinel-2 Natural RGB ({year2})", use_container_width=True)
        st.image(label_to_rgb(pred2), caption=f"Predicted LULC Map (Mean Conf: {prob2.mean()*100:.1f}%)", use_container_width=True)

        fig2, ax2 = plt.subplots(figsize=(5, 3))
        pcts2 = [(pred2 == c).mean() * 100 for c in range(NUM_CLASSES)]
        ax2.barh(CLASS_NAMES, pcts2, color=[LULC_COLORS[c] for c in range(NUM_CLASSES)], edgecolor="black", linewidth=0.5)
        ax2.set_xlim(0, 100)
        ax2.set_xlabel("Area Share (%)", fontsize=8)
        ax2.set_title(f"{year2} Class Distribution", fontsize=9, fontweight="bold")
        ax2.grid(axis="x", alpha=0.3)
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    with col3:
        st.subheader("🔍 Bi-Temporal Change Detection")
        fig_prob, ax_prob = plt.subplots(figsize=(5, 4))
        im = ax_prob.imshow(chg_prob, cmap="magma", vmin=0, vmax=1)
        ax_prob.set_title("Continuous Change Probability", fontsize=9, fontweight="bold")
        ax_prob.axis("off")
        plt.colorbar(im, ax=ax_prob, fraction=0.046, pad=0.04)
        fig_prob.tight_layout()
        st.pyplot(fig_prob)
        plt.close(fig_prob)

        # Semantic change overlay
        chg_overlay = np.stack([sem_change * 255, np.zeros_like(sem_change), np.zeros_like(sem_change)], axis=-1).astype(np.uint8)
        st.image(chg_overlay, caption=f"High-Confidence Change Mask (τ={confidence_tau:.2f})", use_container_width=True)

        # Semantic Transitions
        st.markdown("**Specific Land Changes:**")
        trans_data = get_semantic_transitions(pred1, pred2, sem_change)
        if trans_data:
            for k, v in trans_data.items():
                st.markdown(f"- **{k}:** {v}")
        else:
            st.info("No significant specific transitions detected in this patch.")

    # Ground Truth Comparison Expander
    if chennai_labels is not None and patch_id < chennai_labels.shape[0]:
        with st.expander("🌍 Ground Truth Validation (ESA WorldCover 2021 Reference)"):
            gt_patch = np.array(chennai_labels[patch_id])
            col_gt1, col_gt2, col_gt3 = st.columns([1, 1, 2])
            with col_gt1:
                st.image(label_to_rgb(gt_patch), caption="ESA WorldCover Ground Truth", use_container_width=True)
            with col_gt2:
                st.image(label_to_rgb(pred2), caption=f"Predicted LULC ({year2})", use_container_width=True)
            with col_gt3:
                pixel_acc = (pred2 == gt_patch).mean() * 100
                st.metric("Patch Pixel Accuracy vs Ground Truth", f"{pixel_acc:.1f}%")
                st.write(f"This patch covers 128×128 pixels (1.64 km²) at 10m Sentinel-2 spatial resolution.")

# ─────────────────────────────────────────────────────────────────────────────
# 8. Domain 2: Cross-City Generalization (3 Cities Demo)
# ─────────────────────────────────────────────────────────────────────────────

else:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🌐 Cross-City Selection")

    city_names = list(cross_cities.keys())
    selected_city = st.sidebar.selectbox("Target City", city_names, index=0)
    city_info = cross_cities[selected_city]

    st.info(f"**Cross-City Evaluation**: Model trained on **Chennai**, evaluated on **{selected_city}** without any retraining.\n\n📍 *Region:* {city_info['desc']}")

    tab_patch, tab_mosaic = st.tabs(["🔍 Patch-by-Patch Explorer", "🗺️ Continuous City-Wide Mosaic (~105 km²)"])

    with tab_patch:
        t1_data = city_info["t1"]
        t2_data = city_info["t2"]
        n_city_patches = t1_data.shape[0]

        col_cp1, col_cp2 = st.columns([2, 1])
        with col_cp1:
            city_patch_idx = st.slider(f"Select {selected_city} Patch Index (0 - {n_city_patches - 1})", 0, n_city_patches - 1, 0)
        with col_cp2:
            city_tau = st.slider("Confidence Threshold (τ)", 0.30, 0.90, 0.60, 0.05, key="city_tau")

        # Slice patch
        p1 = np.array(t1_data[city_patch_idx])
        p2 = np.array(t2_data[city_patch_idx])

        t1_norm = normalize_bands(to_nine_band(p1))
        t2_norm = normalize_bands(to_nine_band(p2))

        ten1 = torch.from_numpy(t1_norm).unsqueeze(0).to(device)
        ten2 = torch.from_numpy(t2_norm).unsqueeze(0).to(device)

        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                lulc1, _ = model(ten1, ten1)
                lulc2, chg = model(ten1, ten2)

            pr1 = lulc1.argmax(1).cpu().numpy()[0]
            pr2 = lulc2.argmax(1).cpu().numpy()[0]
            pb1 = F.softmax(lulc1, dim=1).max(dim=1)[0].cpu().numpy()[0]
            pb2 = F.softmax(lulc2, dim=1).max(dim=1)[0].cpu().numpy()[0]
            cg_map = torch.sigmoid(chg).cpu().numpy()[0]

        sem_chg_city = ((pr1 != pr2) & (pb1 >= city_tau) & (pb2 >= city_tau)).astype(np.uint8)

        # 3 Columns
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            st.subheader(f"📅 Baseline (2019)")
            st.image(make_rgb(p1), caption=f"{selected_city} S2 RGB (2019)", use_container_width=True)
            st.image(label_to_rgb(pr1), caption=f"Predicted LULC (Mean Conf: {pb1.mean()*100:.1f}%)", use_container_width=True)

            fig_c1, ax_c1 = plt.subplots(figsize=(5, 3))
            pct1 = [(pr1 == c).mean() * 100 for c in range(NUM_CLASSES)]
            ax_c1.barh(CLASS_NAMES, pct1, color=[LULC_COLORS[c] for c in range(NUM_CLASSES)], edgecolor="black", linewidth=0.5)
            ax_c1.set_xlim(0, 100)
            ax_c1.set_title("2019 Class Distribution (%)", fontsize=9, fontweight="bold")
            ax_c1.grid(axis="x", alpha=0.3)
            fig_c1.tight_layout()
            st.pyplot(fig_c1)
            plt.close(fig_c1)

        with cc2:
            st.subheader(f"📅 Monitoring (2024)")
            st.image(make_rgb(p2), caption=f"{selected_city} S2 RGB (2024)", use_container_width=True)
            st.image(label_to_rgb(pr2), caption=f"Predicted LULC (Mean Conf: {pb2.mean()*100:.1f}%)", use_container_width=True)

            fig_c2, ax_c2 = plt.subplots(figsize=(5, 3))
            pct2 = [(pr2 == c).mean() * 100 for c in range(NUM_CLASSES)]
            ax_c2.barh(CLASS_NAMES, pct2, color=[LULC_COLORS[c] for c in range(NUM_CLASSES)], edgecolor="black", linewidth=0.5)
            ax_c2.set_xlim(0, 100)
            ax_c2.set_title("2024 Class Distribution (%)", fontsize=9, fontweight="bold")
            ax_c2.grid(axis="x", alpha=0.3)
            fig_c2.tight_layout()
            st.pyplot(fig_c2)
            plt.close(fig_c2)

        with cc3:
            st.subheader("🔍 Automated Change Detection")
            fig_cp, ax_cp = plt.subplots(figsize=(5, 4))
            im_c = ax_cp.imshow(cg_map, cmap="magma", vmin=0, vmax=1)
            ax_cp.set_title("Change Probability Map", fontsize=9, fontweight="bold")
            ax_cp.axis("off")
            plt.colorbar(im_c, ax=ax_cp, fraction=0.046, pad=0.04)
            fig_cp.tight_layout()
            st.pyplot(fig_cp)
            plt.close(fig_cp)

            chg_city_overlay = np.stack([sem_chg_city * 255, np.zeros_like(sem_chg_city), np.zeros_like(sem_chg_city)], axis=-1).astype(np.uint8)
            st.image(chg_city_overlay, caption=f"High-Confidence Change Mask ({sem_chg_city.mean()*100:.1f}% area)", use_container_width=True)

            # Semantic Transitions
            st.markdown("**Specific Land Changes:**")
            city_trans_data = get_semantic_transitions(pr1, pr2, sem_chg_city)
            if city_trans_data:
                for k, v in city_trans_data.items():
                    st.markdown(f"- **{k}:** {v}")
            else:
                st.info("No significant specific transitions detected in this patch.")

    with tab_mosaic:
        st.markdown(f"### 🏙️ Continuous 8×8 Mosaic Reconstitution: {selected_city} Core (~105 km²)")
        st.caption("All 64 Sentinel-2 patches seamlessly assembled into a continuous 1024×1024 pixel landscape.")

        with st.spinner(f"Rendering {selected_city} 8×8 full mosaic..."):
            GRID_N = int(np.sqrt(n_city_patches))
            mosaic_rgb_19 = np.zeros((GRID_N * 128, GRID_N * 128, 3), dtype=np.float32)
            mosaic_lulc_19 = np.zeros((GRID_N * 128, GRID_N * 128), dtype=np.uint8)
            mosaic_rgb_24 = np.zeros((GRID_N * 128, GRID_N * 128, 3), dtype=np.float32)
            mosaic_lulc_24 = np.zeros((GRID_N * 128, GRID_N * 128), dtype=np.uint8)

            # Fast batch inference for mosaic
            for idx in range(n_city_patches):
                r = idx // GRID_N
                c = idx % GRID_N
                y, x = r * 128, c * 128

                raw1 = np.array(t1_data[idx])
                raw2 = np.array(t2_data[idx])

                mosaic_rgb_19[y:y+128, x:x+128] = make_rgb(raw1)
                mosaic_rgb_24[y:y+128, x:x+128] = make_rgb(raw2)

                t1n = normalize_bands(to_nine_band(raw1))
                t2n = normalize_bands(to_nine_band(raw2))

                with torch.no_grad():
                    l1, _ = model(torch.from_numpy(t1n).unsqueeze(0).to(device), torch.from_numpy(t1n).unsqueeze(0).to(device))
                    l2, _ = model(torch.from_numpy(t1n).unsqueeze(0).to(device), torch.from_numpy(t2n).unsqueeze(0).to(device))

                mosaic_lulc_19[y:y+128, x:x+128] = l1.argmax(1).cpu().numpy()[0]
                mosaic_lulc_24[y:y+128, x:x+128] = l2.argmax(1).cpu().numpy()[0]

            fig_mos, ax_mos = plt.subplots(1, 3, figsize=(20, 7))
            ax_mos[0].imshow(mosaic_rgb_24)
            ax_mos[0].set_title(f"{selected_city} 2024 — Sentinel-2 Natural RGB (10.24×10.24 km)", fontsize=11, fontweight="bold")
            ax_mos[0].axis("off")

            ax_mos[1].imshow(label_to_rgb(mosaic_lulc_19))
            ax_mos[1].set_title(f"{selected_city} 2019 — Predicted LULC", fontsize=11, fontweight="bold")
            ax_mos[1].axis("off")

            ax_mos[2].imshow(label_to_rgb(mosaic_lulc_24))
            ax_mos[2].set_title(f"{selected_city} 2024 — Predicted LULC", fontsize=11, fontweight="bold")
            ax_mos[2].axis("off")

            legend_patches = [mpatches.Patch(color=LULC_COLORS[c], label=CLASS_NAMES[c]) for c in range(NUM_CLASSES)]
            fig_mos.legend(handles=legend_patches, loc="lower center", ncol=4, fontsize=10, frameon=True,
                           title="Predicted Land Cover Class", bbox_to_anchor=(0.5, -0.08))
            fig_mos.tight_layout()
            st.pyplot(fig_mos)
            plt.close(fig_mos)

# ─────────────────────────────────────────────────────────────────────────────
# 9. Sidebar Class Legend
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎨 Land Cover Legend")
for name, color in zip(CLASS_NAMES, LULC_HEX):
    st.sidebar.markdown(f"<span style='color:{color}; font-size:1.1rem;'>■</span> **{name}**", unsafe_allow_html=True)
