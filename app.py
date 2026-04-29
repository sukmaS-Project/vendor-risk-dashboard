"""
=============================================================
  VENDOR RISK DASHBOARD — app.py
  Sistem Monitoring & Prediksi Risiko Vendor TAD
  Framework: Streamlit + Plotly
  Model: Random Forest Eks.4 (best_model.pkl)
=============================================================
CARA MENJALANKAN:
  pip install streamlit plotly openpyxl pandas numpy scikit-learn xgboost
  streamlit run app.py

STRUKTUR FOLDER:
  penelitian_vendor/
  ├── app.py                         ← file ini
  ├── Data.xlsx                      ← data historis
  ├── output_7eksperimen/
  │   └── best_model.pkl             ← model terbaik
  └── assets/                        ← dibuat otomatis
=============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io
from pathlib import Path
from datetime import datetime, timezone, timedelta
import plotly.graph_objects as go
import plotly.express as px

# ─────────────────────────────────────────────────────────────
# KONFIGURASI HALAMAN
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vendor Risk Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# KONSTANTA
# ─────────────────────────────────────────────────────────────
DATA_FILE  = "Data.xlsx"
MODEL_PATH = Path("output_7eksperimen/best_model.pkl")

BULAN_ORDER = [
    "JANUARI","FEBRUARI","MARET","APRIL","MEI","JUNI",
    "JULI","AGUSTUS","SEPTEMBER","OKTOBER","NOVEMBER","DESEMBER"
]

FITUR_NON_LAG = [
    "Proportion Delay","Proportion Gap",
    "Mean Delay (X01)","Mean Delay (X02)","Mean Delay (X04)",
    "Mean Gap (X11)","Mean Gap (X14)",
]
FITUR_LAG = [
    "Lag Proportion Delay","Lag Proportion Gap",
    "Lag Mean Delay (X02)","Lag Mean Delay (X04)",
    "Lag  Mean Gap (X11)","Lag Mean Gap (X14)",
]
LAG_SOURCE_MAP = {
    "Lag Proportion Delay" : "Proportion Delay",
    "Lag Proportion Gap"   : "Proportion Gap",
    "Lag Mean Delay (X02)" : "Mean Delay (X02)",
    "Lag Mean Delay (X04)" : "Mean Delay (X04)",
    "Lag  Mean Gap (X11)"  : "Mean Gap (X11)",
    "Lag Mean Gap (X14)"   : "Mean Gap (X14)",
}
ALL_FEATURES = FITUR_NON_LAG + FITUR_LAG

# Akun hardcoded
ACCOUNTS = {
    "adminTESTING"      : {"password": "admin123",  "role": "Admin"},
    "manajemenTESTING"  : {"password": "manajemen123",  "role": "Manajemen"},
}

# Warna risiko
COLOR_MAP = {
    "Patuh"  : "#639922",
    "Waspada": "#BA7517",
    "Kritis" : "#E24B4A",
}
BG_MAP = {
    "Patuh"  : "#EAF3DE",
    "Waspada": "#FAEEDA",
    "Kritis" : "#FCEBEB",
}

# ─────────────────────────────────────────────────────────────
# CSS TABLEAU STYLE
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Base */
[data-testid="stAppViewContainer"] {background: #F4F5F7}
[data-testid="stSidebar"] {background: #1B2A47}
[data-testid="stSidebar"] * {color: #CBD5E1}
[data-testid="stSidebar"] .stRadio label {
    color: #CBD5E1 !important; font-size: 14px;
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {color: #FFFFFF !important}

/* Header top */
.dash-header {
    background: #1B2A47; color: white;
    padding: 12px 20px; border-radius: 8px;
    margin-bottom: 20px; display: flex;
    justify-content: space-between; align-items: center;
}
.dash-header h2 {margin:0; font-size:18px; color:white}
.dash-header span {font-size:12px; opacity:0.7}

/* KPI Cards */
.kpi-card {
    background: white; border-radius: 10px;
    padding: 16px 20px; border: 1px solid #E2E8F0;
    text-align: center;
}
.kpi-label {font-size: 11px; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.5px}
.kpi-value {font-size: 28px; font-weight: 600; margin: 4px 0}
.kpi-badge {display: inline-block; font-size: 11px; padding: 2px 10px; border-radius: 20px; margin-top: 2px}

/* Risk badges inline */
.badge-patuh   {background:#EAF3DE; color:#3B6D11; padding:3px 10px; border-radius:20px; font-size:12px; font-weight:500}
.badge-waspada {background:#FAEEDA; color:#854F0B; padding:3px 10px; border-radius:20px; font-size:12px; font-weight:500}
.badge-kritis  {background:#FCEBEB; color:#A32D2D; padding:3px 10px; border-radius:20px; font-size:12px; font-weight:500}

/* Section title */
.section-label {
    font-size:11px; font-weight:600; color:#64748B;
    text-transform:uppercase; letter-spacing:0.8px;
    margin: 8px 0 12px;
}

/* Alert banner */
.alert-kritis {
    background:#FCEBEB; border-left: 4px solid #E24B4A;
    padding:10px 14px; border-radius:4px; margin-bottom:12px;
    font-size:13px; color:#A32D2D;
}
.alert-waspada {
    background:#FAEEDA; border-left: 4px solid #BA7517;
    padding:10px 14px; border-radius:4px; margin-bottom:12px;
    font-size:13px; color:#854F0B;
}

/* Metric highlight */
.highlight-red {color: #E24B4A; font-weight:600}
.highlight-ok  {color: #639922}

/* Table style */
.styled-table {width:100%; border-collapse:collapse; font-size:13px}
.styled-table th {
    background:#F8FAFC; color:#64748B; font-weight:500;
    padding:10px 12px; text-align:left;
    border-bottom:2px solid #E2E8F0; font-size:11px;
    text-transform:uppercase; letter-spacing:0.4px
}
.styled-table td {padding:10px 12px; border-bottom:1px solid #F1F5F9; color:#1E293B}
.styled-table tr:hover td {background:#F8FAFC}

/* Login card */
.login-card {
    max-width: 380px; margin: 80px auto;
    background: white; border-radius: 12px;
    padding: 36px; box-shadow: 0 4px 24px rgba(0,0,0,0.08);
    border: 1px solid #E2E8F0;
}

/* Prediksi timeline */
.pred-label {
    font-size:11px; text-align:center;
    color:#64748B; margin-top:4px
}

/* Generate page */
.upload-zone {
    border: 2px dashed #CBD5E1; border-radius: 8px;
    padding: 24px; text-align:center; color:#94A3B8;
    background: #F8FAFC; margin: 12px 0;
}

/* Info box */
.info-box {
    background:#EFF6FF; border:1px solid #BFDBFE;
    border-radius:8px; padding:12px 16px;
    font-size:12px; color:#1D4ED8; margin:8px 0;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# FUNGSI — DATA & MODEL
# ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_data(ttl=30)
def load_data():
    if not Path(DATA_FILE).exists():
        return pd.DataFrame()
    df = pd.read_excel(DATA_FILE)
    if "MLag ean Delay (X04)" in df.columns:
        df = df.rename(columns={"MLag ean Delay (X04)": "Lag Mean Delay (X04)"})
    df["Bulan Tagihan"] = df["Bulan Tagihan"].str.upper().str.strip()
    df["Bulan_sort"] = pd.Categorical(
        df["Bulan Tagihan"], categories=BULAN_ORDER, ordered=True
    )
    df = df.sort_values(
        ["Tahun Tagihan","Bulan_sort","Nama Vendor"]
    ).reset_index(drop=True)
    df["periode_num"] = df["Tahun Tagihan"]*12 + df["Bulan_sort"].cat.codes

    # Hitung skor dan label jika belum ada
    if "Skor Risiko (Expert Judgment)" not in df.columns:
        df["Skor Risiko (Expert Judgment)"] = (
            (0.4*df["Proportion Delay"] + 0.6*df["Proportion Gap"])*100
        ).round(2)
    if "Klasifikasi Risiko" not in df.columns:
        df["Klasifikasi Risiko"] = df["Skor Risiko (Expert Judgment)"].apply(
            lambda s: "Patuh" if s==0 else ("Waspada" if s<=25 else "Kritis")
        )
    return df


def get_lag(df, vendor, bulan, tahun):
    bulan_u = bulan.upper().strip()
    periode_ini  = tahun*12 + BULAN_ORDER.index(bulan_u)
    periode_prev = periode_ini - 1
    baris = df[(df["Nama Vendor"]==vendor) & (df["periode_num"]==periode_prev)]
    if len(baris)==0:
        return {c:0.0 for c in FITUR_LAG}, False
    row = baris.iloc[0]
    return {lag_c: float(row.get(src_c,0.0))
            for lag_c, src_c in LAG_SOURCE_MAP.items()}, True


def predict_row(artifact, fitur_row):
    if artifact is None:
        return "N/A", {}
    model, scaler = artifact["model"], artifact["scaler"]
    label_inv = artifact["label_inv"]
    X = np.array([[fitur_row[f] for f in ALL_FEATURES]])
    X_sc = scaler.transform(X)
    code = model.predict(X_sc)[0]
    label = label_inv[int(code)]
    prob = {}
    if hasattr(model,"predict_proba"):
        p = model.predict_proba(X_sc)[0]
        prob = {"Patuh":p[0],"Waspada":p[1],"Kritis":p[2]}
    return label, prob


def hitung_klasifikasi_formula(skor):
    if skor == 0: return "Patuh"
    elif skor <= 25: return "Waspada"
    return "Kritis"


def bulan_berikutnya(bulan, tahun):
    idx = BULAN_ORDER.index(bulan.upper())
    if idx==11: return "JANUARI", tahun+1
    return BULAN_ORDER[idx+1], tahun


def badge_html(label):
    cls = {"Patuh":"badge-patuh","Waspada":"badge-waspada","Kritis":"badge-kritis"}
    dot = {"Patuh":"🟢","Waspada":"🟡","Kritis":"🔴"}
    return f'<span class="{cls.get(label,"")}"> {dot.get(label,"")} {label}</span>'


def get_periode_options(df):
    """Kembalikan opsi Tahun dan mapping Tahun → list Bulan (terbaru dulu)."""
    if df.empty: return [], {}
    years = sorted(df["Tahun Tagihan"].unique(), reverse=True)
    bulan_per_tahun = {}
    for y in years:
        bulan_list = (df[df["Tahun Tagihan"]==y]["Bulan Tagihan"]
                      .unique().tolist())
        bulan_list = sorted(bulan_list,
                            key=lambda b: BULAN_ORDER.index(b.upper()),
                            reverse=True)
        bulan_per_tahun[y] = bulan_list
    return years, bulan_per_tahun


# ─────────────────────────────────────────────────────────────
# FUNGSI — LOGIN
# ─────────────────────────────────────────────────────────────

def show_login():
    st.markdown("""
    <style>
    .login-wrapper {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 50vh;
    }

    .login-card {
        width: 420px;
        background: white;
        border-radius: 14px;
        padding: 40px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.08);
        border: 1px solid #E2E8F0;
        text-align: center;
    }

    .login-title {
        font-size: 26px;
        font-weight: 700;
        color: #1B2A47;
        margin-bottom: 6px;
    }

    .login-subtitle {
        font-size: 13px;
        color: #94A3B8;
        margin-bottom: 25px;
    }

    .login-icon {
        font-size: 40px;
        margin-bottom: 10px;
    }

    .footer-text {
        text-align:center;
        margin-top:20px;
        font-size:11px;
        color:#94A3B8;
    }
    </style>

    <div class="login-wrapper">
        <div class="login-card">
            <div class="login-icon">📊</div>
            <div class="login-title">VendorRisk</div>
            <div class="login-subtitle">
                Sistem Monitoring & Prediksi Risiko Vendor TAD
            </div>
    """, unsafe_allow_html=True)

    with st.form("login_form"):
        username = st.text_input("Username", placeholder="Masukkan username")
        password = st.text_input("Password", type="password", placeholder="Masukkan password")

        submitted = st.form_submit_button("🔐 Masuk", use_container_width=True)

        if submitted:
            if username in ACCOUNTS and ACCOUNTS[username]["password"] == password:
                st.session_state["logged_in"] = True
                st.session_state["username"]  = username
                st.session_state["role"]      = ACCOUNTS[username]["role"]
                st.rerun()
            else:
                st.error("❌ Username atau password salah")

    st.markdown("""
        </div>
    </div>

    <div class="footer-text">
        © 2026 VendorRisk Dashboard<br>
        Developed by Sukma Sufryanto<br>
        Version 1.0.0
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# HALAMAN — DASHBOARD (MONITORING)
# ─────────────────────────────────────────────────────────────

def page_dashboard(df, artifact):
    st.markdown('<div class="section-label">Monitoring Risiko Vendor</div>',
                unsafe_allow_html=True)

    if df.empty:
        st.warning("Data belum tersedia. Silakan upload data melalui menu Generate.")
        return

    years, bulan_per_tahun = get_periode_options(df)

    # ── Filter ─────────────────────────────────────────────
    c1, c2, c3 = st.columns([1,1,2])
    with c1:
        sel_year = st.selectbox("Tahun", years, key="dash_year")
    with c2:
        sel_bulan = st.selectbox("Bulan", bulan_per_tahun.get(sel_year,[]),
                                 key="dash_bulan")
    with c3:
        vendors_avail = sorted(
            df[(df["Tahun Tagihan"]==sel_year) &
               (df["Bulan Tagihan"]==sel_bulan)]["Nama Vendor"].unique()
        )
        sel_vendor = st.selectbox("Vendor",
                                  ["All Vendors"] + vendors_avail,
                                  key="dash_vendor")

    # Filter data
    dff = df[(df["Tahun Tagihan"]==sel_year) & (df["Bulan Tagihan"]==sel_bulan)]
    if sel_vendor != "All Vendors":
        dff = dff[dff["Nama Vendor"]==sel_vendor]

    if dff.empty:
        st.info("Tidak ada data untuk periode yang dipilih.")
        return

    # ── Alert banner ────────────────────────────────────────
    kritis_list  = dff[dff["Klasifikasi Risiko"]=="Kritis"]["Nama Vendor"].tolist()
    waspada_list = dff[dff["Klasifikasi Risiko"]=="Waspada"]["Nama Vendor"].tolist()
    if kritis_list:
        st.markdown(
            f'<div class="alert-kritis">🔴 <b>Perhatian Kritis:</b> '
            f'{", ".join(kritis_list)} memerlukan tindakan segera.</div>',
            unsafe_allow_html=True)
    if waspada_list:
        st.markdown(
            f'<div class="alert-waspada">🟡 <b>Perlu Pemantauan:</b> '
            f'{", ".join(waspada_list)} menunjukkan indikasi risiko.</div>',
            unsafe_allow_html=True)

    # ── KPI Cards ───────────────────────────────────────────
    total   = len(dff)
    n_patuh = (dff["Klasifikasi Risiko"]=="Patuh").sum()
    n_wasp  = (dff["Klasifikasi Risiko"]=="Waspada").sum()
    n_krit  = (dff["Klasifikasi Risiko"]=="Kritis").sum()
    avg_skor = dff["Skor Risiko (Expert Judgment)"].mean()

    k1,k2,k3,k4,k5 = st.columns(5)
    kpi_data = [
        (k1, "Total Vendor",   str(total),   f"{sel_bulan} {sel_year}", "#1B2A47", "#E2E8F0"),
        (k2, "Patuh",          str(n_patuh), f"{n_patuh/total*100:.0f}%", "#3B6D11", "#EAF3DE"),
        (k3, "Waspada",        str(n_wasp),  f"{n_wasp/total*100:.0f}%",  "#854F0B", "#FAEEDA"),
        (k4, "Kritis",         str(n_krit),  f"{n_krit/total*100:.0f}%",  "#A32D2D", "#FCEBEB"),
        (k5, "Avg Skor Risiko",f"{avg_skor:.1f}", "dari 100", "#185FA5","#EFF6FF"),
    ]
    for col, label, val, sub, color, bg in kpi_data:
        col.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value" style="color:{color}">{val}</div>
            <span class="kpi-badge" style="background:{bg};color:{color}">{sub}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts ──────────────────────────────────────────────
    col_l, col_r = st.columns([1,1])

    with col_l:
        # Bar chart skor risiko per vendor
        dff_sorted = dff.sort_values("Skor Risiko (Expert Judgment)", ascending=True)
        colors_bar = [COLOR_MAP.get(k,"#888") for k in dff_sorted["Klasifikasi Risiko"]]
        fig_bar = go.Figure(go.Bar(
            x=dff_sorted["Skor Risiko (Expert Judgment)"],
            y=dff_sorted["Nama Vendor"],
            orientation="h",
            marker_color=colors_bar,
            text=[f"{s:.1f}" for s in dff_sorted["Skor Risiko (Expert Judgment)"]],
            textposition="outside",
        ))
        fig_bar.add_vline(x=25, line_dash="dash", line_color="#E24B4A",
                          annotation_text="Batas Kritis (25)", annotation_position="top")
        fig_bar.update_layout(
            title=f"Skor Risiko Vendor — {sel_bulan} {sel_year}",
            xaxis_title="Skor Risiko", yaxis_title="",
            height=280, margin=dict(l=0,r=40,t=40,b=20),
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(family="Arial", size=12),
            xaxis=dict(range=[0, max(30, dff_sorted["Skor Risiko (Expert Judgment)"].max()+5)]),
            showlegend=False
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_r:
        # Donut chart distribusi kategori
        cat_counts = dff["Klasifikasi Risiko"].value_counts().reindex(
            ["Patuh","Waspada","Kritis"], fill_value=0
        )
        fig_pie = go.Figure(go.Pie(
            labels=cat_counts.index,
            values=cat_counts.values,
            marker_colors=[COLOR_MAP[k] for k in cat_counts.index],
            hole=0.5,
            textinfo="label+percent",
            textfont_size=12,
        ))
        fig_pie.update_layout(
            title=f"Distribusi Kategori — {sel_bulan} {sel_year}",
            height=280, margin=dict(l=0,r=0,t=40,b=20),
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(family="Arial"),
            showlegend=False,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # ── Tren Skor Risiko ────────────────────────────────────
    st.markdown('<div class="section-label">Tren Skor Risiko (12 Bulan Terakhir)</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        Garis menampilkan skor risiko numerik (0–100). Titik berwarna mengindikasikan kategori:
        <b style="color:#3B6D11">Hijau = Patuh</b> |
        <b style="color:#854F0B">Kuning = Waspada</b> |
        <b style="color:#A32D2D">Merah = Kritis</b>.
        Garis putus-putus merah = batas Kritis (skor 25).
    </div>""", unsafe_allow_html=True)

    # Ambil 12 periode terakhir
    all_periods = df[["Tahun Tagihan","Bulan Tagihan","Bulan_sort","periode_num"]]\
        .drop_duplicates().sort_values(["Tahun Tagihan","Bulan_sort"])
    sel_periode_num = (df[(df["Tahun Tagihan"]==sel_year) &
                          (df["Bulan Tagihan"]==sel_bulan)]["periode_num"].iloc[0]
                       if not df[(df["Tahun Tagihan"]==sel_year) &
                                 (df["Bulan Tagihan"]==sel_bulan)].empty else 0)
    recent_periods = all_periods[
        all_periods["periode_num"] <= sel_periode_num
    ].tail(12)

    vendors_trend = vendors_avail if sel_vendor=="All Vendors" else [sel_vendor]
    fig_trend = go.Figure()
    fig_trend.add_hline(y=25, line_dash="dash", line_color="#E24B4A",
                        line_width=1, annotation_text="Kritis",
                        annotation_position="right")
    fig_trend.add_hrect(y0=0, y1=0.5, fillcolor="#EAF3DE", opacity=0.3, line_width=0)
    fig_trend.add_hrect(y0=0.5, y1=25, fillcolor="#FAEEDA", opacity=0.2, line_width=0)
    fig_trend.add_hrect(y0=25, y1=100, fillcolor="#FCEBEB", opacity=0.15, line_width=0)

    colors_line = ["#185FA5","#BA7517","#E24B4A","#639922","#534AB7","#D85A30"]
    for i, vendor in enumerate(vendors_trend):
        vdf = df[df["Nama Vendor"]==vendor].copy()
        vdf = vdf[vdf["periode_num"].isin(recent_periods["periode_num"])]
        vdf = vdf.sort_values("periode_num")
        if vdf.empty: continue

        x_labels = vdf["Bulan Tagihan"].str[:3] + " " + vdf["Tahun Tagihan"].astype(str)
        skor     = vdf["Skor Risiko (Expert Judgment)"]
        klabel   = vdf["Klasifikasi Risiko"]

        dot_colors = [COLOR_MAP.get(k,"#888") for k in klabel]
        color_line = colors_line[i % len(colors_line)]

        fig_trend.add_trace(go.Scatter(
            x=x_labels, y=skor, mode="lines+markers",
            name=vendor, line=dict(color=color_line, width=2),
            marker=dict(color=dot_colors, size=8, line=dict(color="white",width=1.5)),
            hovertemplate=f"<b>{vendor}</b><br>%{{x}}<br>Skor: %{{y:.1f}}<extra></extra>"
        ))

    fig_trend.update_layout(
        height=320, margin=dict(l=0,r=60,t=20,b=40),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Arial", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        yaxis=dict(title="Skor Risiko", range=[0,100]),
        xaxis=dict(title=""),
        hovermode="x unified"
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    # ── Tabel Detail ────────────────────────────────────────
    st.markdown('<div class="section-label">Tabel Ringkasan Vendor</div>',
                unsafe_allow_html=True)
    rows_html = ""
    for _, row in dff.sort_values("Skor Risiko (Expert Judgment)", ascending=False).iterrows():
        label = row["Klasifikasi Risiko"]
        skor  = row["Skor Risiko (Expert Judgment)"]
        pd_val= row.get("Proportion Delay",0)
        pg_val= row.get("Proportion Gap",0)
        rows_html += f"""<tr>
            <td><b>{row['Nama Vendor']}</b></td>
            <td>{skor:.2f}</td>
            <td>{badge_html(label)}</td>
            <td {'style="color:#E24B4A;font-weight:600"' if pd_val>0 else ''}>{pd_val:.3f}</td>
            <td {'style="color:#E24B4A;font-weight:600"' if pg_val>0 else ''}>{pg_val:.3f}</td>
        </tr>"""

    st.markdown(f"""
    <table class="styled-table">
        <thead><tr>
            <th>Vendor</th><th>Skor Risiko</th><th>Kategori</th>
            <th>Proportion Delay</th><th>Proportion Gap</th>
        </tr></thead>
        <tbody>{rows_html}</tbody>
    </table>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# HALAMAN — PREDICTIONS
# ─────────────────────────────────────────────────────────────

def page_predictions(df, artifact):
    st.markdown('<div class="section-label">Prediksi Risiko Bulan Depan</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        Prediksi menggunakan <b>Random Forest Eks.4</b> dengan carry-forward assumption (LOCF).
        Fitur non-lag bulan depan diestimasi dari nilai aktual bulan ini.
        Garis solid = aktual | Garis putus-putus = prediksi.
    </div>""", unsafe_allow_html=True)

    if df.empty or artifact is None:
        st.warning("Data atau model belum tersedia.")
        return

    years, bulan_per_tahun = get_periode_options(df)

    c1, c2, c3 = st.columns([1,1,2])
    with c1:
        sel_year  = st.selectbox("Tahun", years, key="pred_year")
    with c2:
        sel_bulan = st.selectbox("Bulan", bulan_per_tahun.get(sel_year,[]),
                                 key="pred_bulan")
    with c3:
        vendors_avail = sorted(df[(df["Tahun Tagihan"]==sel_year) &
                                   (df["Bulan Tagihan"]==sel_bulan)]["Nama Vendor"].unique())
        sel_vendor = st.selectbox("Vendor", ["All Vendors"]+vendors_avail,
                                  key="pred_vendor")

    dff = df[(df["Tahun Tagihan"]==sel_year)&(df["Bulan Tagihan"]==sel_bulan)]
    if sel_vendor != "All Vendors":
        dff = dff[dff["Nama Vendor"]==sel_vendor]
    if dff.empty:
        st.info("Tidak ada data untuk periode yang dipilih.")
        return

    bulan_dep, tahun_dep = bulan_berikutnya(sel_bulan, sel_year)

    # Hitung prediksi untuk setiap vendor
    hasil_pred = []
    for _, row in dff.iterrows():
        vendor = row["Nama Vendor"]
        fitur_ini = {f: float(row.get(f,0.0)) for f in FITUR_NON_LAG}
        lag_ini, _= get_lag(df, vendor, sel_bulan, sel_year)

        # Klasifikasi t (monitoring bulan ini — formula)
        skor_t = float(row.get("Skor Risiko (Expert Judgment)",0))
        label_t_formula = hitung_klasifikasi_formula(skor_t)

        # Prediksi t+1 (carry-forward)
        lag_dep = {lag_c: fitur_ini[src_c]
                   for lag_c, src_c in LAG_SOURCE_MAP.items()}
        fitur_pred = {**fitur_ini, **lag_dep}
        label_pred, prob_pred = predict_row(artifact, fitur_pred)

        hasil_pred.append({
            "Vendor"            : vendor,
            "Skor t"            : round(skor_t,2),
            "Label t (formula)" : label_t_formula,
            "Prediksi t+1"      : label_pred,
            "Prob Patuh"        : prob_pred.get("Patuh",0),
            "Prob Waspada"      : prob_pred.get("Waspada",0),
            "Prob Kritis"       : prob_pred.get("Kritis",0),
        })

    df_pred = pd.DataFrame(hasil_pred)

    # ── KPI prediksi ────────────────────────────────────────
    st.markdown(f"<br><b>Prediksi untuk {bulan_dep} {tahun_dep}</b><br><br>",
                unsafe_allow_html=True)
    cols = st.columns(len(df_pred))
    for i, (_, row) in enumerate(df_pred.iterrows()):
        label = row["Prediksi t+1"]
        prob  = row[f"Prob {label}"]
        with cols[i]:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{row['Vendor']}</div>
                <div style="font-size:22px;margin:6px 0">{badge_html(label)}</div>
                <div style="font-size:11px;color:#94A3B8">Keyakinan: {prob:.0%}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Grafik probabilitas ─────────────────────────────────
    fig_prob = go.Figure()
    for kelas, color in COLOR_MAP.items():
        fig_prob.add_trace(go.Bar(
            name=kelas,
            x=df_pred["Vendor"],
            y=df_pred[f"Prob {kelas}"].round(3)*100,
            marker_color=color, opacity=0.85,
        ))
    fig_prob.update_layout(
        title=f"Probabilitas Prediksi per Vendor — {bulan_dep} {tahun_dep}",
        barmode="stack", height=300,
        yaxis_title="Probabilitas (%)", xaxis_title="",
        margin=dict(l=0,r=0,t=40,b=20),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Arial",size=12),
        legend=dict(orientation="h",yanchor="bottom",y=0.97)
    )
    st.plotly_chart(fig_prob, use_container_width=True)

    # ── Timeline t-1 / t / t+1 ─────────────────────────────
    st.markdown('<div class="section-label">Timeline Risiko — t-1 | t (aktual) | t+1 (prediksi)</div>',
                unsafe_allow_html=True)

    # Hitung t-1
    bulan_prev_list = df[(df["Tahun Tagihan"]==sel_year) &
                         (df["Bulan Tagihan"]==sel_bulan)]["periode_num"]
    if not bulan_prev_list.empty:
        periode_prev = bulan_prev_list.iloc[0] - 1
        df_prev = df[df["periode_num"]==periode_prev]
    else:
        df_prev = pd.DataFrame()

    vendors_plot = df_pred["Vendor"].tolist() if sel_vendor=="All Vendors" else [sel_vendor]
    colors_line  = ["#185FA5","#BA7517","#E24B4A","#639922","#534AB7","#D85A30"]

    fig_tl = go.Figure()
    fig_tl.add_hline(y=25, line_dash="dash", line_color="#E24B4A",
                     line_width=1, annotation_text="Batas Kritis",
                     annotation_position="right")
    fig_tl.add_hrect(y0=0,  y1=0.5, fillcolor="#EAF3DE", opacity=0.3, line_width=0)
    fig_tl.add_hrect(y0=0.5,y1=25,  fillcolor="#FAEEDA", opacity=0.2, line_width=0)
    fig_tl.add_hrect(y0=25, y1=100, fillcolor="#FCEBEB", opacity=0.15,line_width=0)

    # Label periode
    bulan_idx = BULAN_ORDER.index(sel_bulan.upper())
    if bulan_idx == 0:
        prev_label = f"DES {sel_year-1}"
    else:
        prev_label = f"{BULAN_ORDER[bulan_idx-1][:3]} {sel_year}"
    curr_label = f"{sel_bulan[:3]} {sel_year} (aktual)"
    next_label = f"{bulan_dep[:3]} {tahun_dep} (prediksi)"
    x_labels   = [prev_label, curr_label, next_label]

    for i, vendor in enumerate(vendors_plot):
        cl = colors_line[i % len(colors_line)]

        # Skor t-1
        skor_prev = 0
        if not df_prev.empty and vendor in df_prev["Nama Vendor"].values:
            skor_prev = float(df_prev[df_prev["Nama Vendor"]==vendor]
                              ["Skor Risiko (Expert Judgment)"].iloc[0])

        # Skor t
        row_t = df_pred[df_pred["Vendor"]==vendor]
        skor_t = float(row_t["Skor t"].iloc[0]) if not row_t.empty else 0

        # Skor t+1 — gunakan midpoint probabilitas sebagai proxy visual
        pred_row = row_t.iloc[0] if not row_t.empty else None
        if pred_row is not None:
            label_t1 = pred_row["Prediksi t+1"]
            skor_t1 = {"Patuh":0,"Waspada":12.5,"Kritis":62.5}.get(label_t1,0)
        else:
            skor_t1 = 0

        # Garis solid (t-1 → t)
        fig_tl.add_trace(go.Scatter(
            x=[prev_label, curr_label], y=[skor_prev, skor_t],
            mode="lines+markers", name=vendor,
            line=dict(color=cl, width=2.5),
            marker=dict(size=9, color=[
                COLOR_MAP.get(hitung_klasifikasi_formula(skor_prev),"#888"),
                COLOR_MAP.get(hitung_klasifikasi_formula(skor_t),"#888")
            ], line=dict(color="white",width=1.5)),
            legendgroup=vendor, showlegend=True,
            hovertemplate=f"<b>{vendor}</b><br>%{{x}}: %{{y:.1f}}<extra></extra>"
        ))

        # Garis putus-putus (t → t+1)
        fig_tl.add_trace(go.Scatter(
            x=[curr_label, next_label], y=[skor_t, skor_t1],
            mode="lines+markers", name=f"{vendor} (prediksi)",
            line=dict(color=cl, width=2.5, dash="dot"),
            marker=dict(size=9, symbol="diamond",
                        color=[COLOR_MAP.get(hitung_klasifikasi_formula(skor_t),"#888"),
                               COLOR_MAP.get(label_t1,"#888")],
                        line=dict(color="white",width=1.5)),
            legendgroup=vendor, showlegend=False,
            hovertemplate=f"<b>{vendor} (prediksi)</b><br>%{{x}}: {label_t1}<extra></extra>"
        ))

    fig_tl.update_layout(
        height=340, margin=dict(l=0,r=80,t=20,b=40),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Arial",size=12),
        yaxis=dict(title="Skor Risiko",range=[-2,105]),
        xaxis=dict(title=""),
        legend=dict(orientation="h",yanchor="bottom",y=1.02),
        hovermode="x unified"
    )

    # Anotasi zona
    for y_mid, label_z, color_z in [(0,"Patuh","#3B6D11"),(12,"Waspada","#854F0B"),(37,"Kritis","#A32D2D")]:
        fig_tl.add_annotation(
            x=1.01, y=y_mid, text=label_z, xref="paper",
            showarrow=False, font=dict(color=color_z, size=10),
            xanchor="left"
        )

    st.plotly_chart(fig_tl, use_container_width=True)

    # ── Tabel prediksi ──────────────────────────────────────
    rows_html = ""
    for _, row in df_pred.iterrows():
        rows_html += f"""<tr>
            <td><b>{row['Vendor']}</b></td>
            <td>{badge_html(row['Label t (formula)'])}</td>
            <td>{row['Skor t']:.2f}</td>
            <td>{badge_html(row['Prediksi t+1'])}</td>
            <td>{row['Prob Patuh']:.0%} / {row['Prob Waspada']:.0%} / {row['Prob Kritis']:.0%}</td>
        </tr>"""

    st.markdown(f"""
    <br><table class="styled-table">
        <thead><tr>
            <th>Vendor</th>
            <th>Aktual {sel_bulan[:3]} {sel_year}</th>
            <th>Skor Aktual</th>
            <th>Prediksi {bulan_dep[:3]} {tahun_dep}</th>
            <th>Prob (Patuh/Waspada/Kritis)</th>
        </tr></thead>
        <tbody>{rows_html}</tbody>
    </table>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# HALAMAN — DETAIL
# ─────────────────────────────────────────────────────────────

def page_detail(df, artifact):
    st.markdown('<div class="section-label">Detail Faktor Risiko Vendor</div>',
                unsafe_allow_html=True)

    if df.empty:
        st.warning("Data belum tersedia.")
        return

    years, bulan_per_tahun = get_periode_options(df)

    c1,c2,c3 = st.columns([1,1,2])
    with c1:
        sel_year  = st.selectbox("Tahun", years, key="det_year")
    with c2:
        sel_bulan = st.selectbox("Bulan", bulan_per_tahun.get(sel_year,[]),
                                 key="det_bulan")
    with c3:
        vendors_avail = sorted(df[(df["Tahun Tagihan"]==sel_year) &
                                   (df["Bulan Tagihan"]==sel_bulan)]["Nama Vendor"].unique())
        sel_vendor = st.selectbox("Vendor", ["All Vendors"]+vendors_avail,
                                  key="det_vendor")

    dff = df[(df["Tahun Tagihan"]==sel_year)&(df["Bulan Tagihan"]==sel_bulan)]
    if sel_vendor != "All Vendors":
        dff = dff[dff["Nama Vendor"]==sel_vendor]
    if dff.empty:
        st.info("Tidak ada data.")
        return

    # Untuk setiap vendor, tampilkan kartu detail
    for _, row in dff.sort_values("Skor Risiko (Expert Judgment)", ascending=False).iterrows():
        vendor = row["Nama Vendor"]
        label  = row["Klasifikasi Risiko"]
        skor   = row["Skor Risiko (Expert Judgment)"]
        pd_val = row.get("Proportion Delay",0)
        pg_val = row.get("Proportion Gap",0)

        # Bahasa bisnis
        def prop_to_bisnis_delay(p):
            n = round(p*4)  # dari 4 jenis kewajiban
            if n==0: return "Seluruh kewajiban terbayar tepat waktu"
            return f"Terdapat keterlambatan pada <b>{n} dari 4 jenis kewajiban</b> hak normatif"

        def prop_to_bisnis_gap(p):
            n = round(p*7)  # dari 7 jenis kewajiban
            if n==0: return "Seluruh kewajiban terbayar sesuai nilai yang ditentukan"
            return f"Terdapat selisih pembayaran pada <b>{n} dari 7 jenis kewajiban</b> hak normatif"

        color_header = {"Patuh":"#EAF3DE","Waspada":"#FAEEDA","Kritis":"#FCEBEB"}
        color_text   = {"Patuh":"#3B6D11","Waspada":"#854F0B","Kritis":"#A32D2D"}

        with st.expander(
            f"{'🟢' if label=='Patuh' else '🟡' if label=='Waspada' else '🔴'}  "
            f"{vendor}  —  Skor: {skor:.2f}  |  {label}",
            expanded=(label!="Patuh")
        ):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div style="background:{color_header.get(label,'#fff')};
                     border-radius:8px;padding:12px 16px;margin-bottom:8px">
                    <div style="font-size:11px;color:#64748B;text-transform:uppercase;
                         letter-spacing:0.5px">Proportion Delay</div>
                    <div style="font-size:24px;font-weight:600;
                         color:{'#E24B4A' if pd_val>0 else '#3B6D11'}">{pd_val:.3f}</div>
                    <div style="font-size:12px;color:#475569;margin-top:4px">
                        {prop_to_bisnis_delay(pd_val)}
                    </div>
                </div>""", unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div style="background:{color_header.get(label,'#fff')};
                     border-radius:8px;padding:12px 16px;margin-bottom:8px">
                    <div style="font-size:11px;color:#64748B;text-transform:uppercase;
                         letter-spacing:0.5px">Proportion Gap</div>
                    <div style="font-size:24px;font-weight:600;
                         color:{'#E24B4A' if pg_val>0 else '#3B6D11'}">{pg_val:.3f}</div>
                    <div style="font-size:12px;color:#475569;margin-top:4px">
                        {prop_to_bisnis_gap(pg_val)}
                    </div>
                </div>""", unsafe_allow_html=True)

            # Breakdown Mean Delay dan Mean Gap
            mean_delay_cols = ["Mean Delay (X01)","Mean Delay (X02)","Mean Delay (X04)"]
            mean_gap_cols   = ["Mean Gap (X11)","Mean Gap (X14)"]

            st.markdown("**Rincian Keterlambatan (hari rata-rata):**")
            delay_labels = {"Mean Delay (X01)":"Gaji","Mean Delay (X02)":"BPJS TK",
                            "Mean Delay (X04)":"THR / DPLK"}
            c1r,c2r,c3r = st.columns(3)
            for col_w, dc in zip([c1r,c2r,c3r], mean_delay_cols):
                val = float(row.get(dc,0))
                col_w.metric(
                    delay_labels.get(dc, dc),
                    f"{val:.1f} hari",
                    delta=None
                )

            st.markdown("**Rincian Selisih Pembayaran:**")
            gap_labels = {"Mean Gap (X11)":"Gaji","Mean Gap (X14)":"BPJS Kes"}
            cg1, cg2 = st.columns(2)
            for col_w, gc in zip([cg1,cg2], mean_gap_cols):
                val = float(row.get(gc,0))
                col_w.metric(
                    gap_labels.get(gc, gc),
                    f"{val:.4f}",
                    delta=None
                )

            if label != "Patuh":
                st.markdown(f"""
                <div class="{'alert-kritis' if label=='Kritis' else 'alert-waspada'}">
                    <b>Rekomendasi:</b>
                    {'Lakukan eskalasi segera ke manajemen dan tinjau ulang kontrak.' if label=='Kritis'
                     else 'Tingkatkan frekuensi pemantauan dan kirimkan surat peringatan tertulis kepada vendor.'}
                </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# HALAMAN — GENERATE (ADMIN ONLY)
# ─────────────────────────────────────────────────────────────

def page_generate(df, artifact):
    st.markdown('<div class="section-label">Generate — Input Data Bulanan</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        Halaman ini hanya dapat diakses oleh akun <b>Admin</b>.
        Upload data bulanan baru, sistem akan melakukan validasi dan
        mengintegrasikan ke dataset secara otomatis.
    </div>""", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📥 Download Template", "📤 Upload Data Baru", "🔄 Status Dataset"])

    # ── TAB 1: Download Template ────────────────────────────
    with tab1:
        st.markdown("#### Download Template Excel")
        st.markdown("Template berisi kolom yang harus diisi untuk setiap vendor aktif.")

        vendors_aktif = ["PT CSMP","PT LJ","PT MS","PT PMS","PT UJP"]
        template_data = {
            "Nama Vendor"      : vendors_aktif,
            "Bulan Tagihan"    : ["JANUARI"]*5,
            "Tahun Tagihan"    : [2026]*5,
            "Proportion Delay" : [0.0]*5,
            "Proportion Gap"   : [0.0]*5,
            "Mean Delay (X01)" : [0.0]*5,
            "Mean Delay (X02)" : [0.0]*5,
            "Mean Delay (X04)" : [0.0]*5,
            "Mean Gap (X11)"   : [0.0]*5,
            "Mean Gap (X14)"   : [0.0]*5,
        }
        df_template = pd.DataFrame(template_data)

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df_template.to_excel(writer, index=False, sheet_name="Input Data")
        buf.seek(0)

        st.download_button(
            label="⬇️  Download Template Excel",
            data=buf,
            file_name="Data_Input_Bulan_Ini.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=False
        )

        st.markdown("""
        **Petunjuk pengisian:**
        - Ubah `Bulan Tagihan` sesuai periode (contoh: FEBRUARI)
        - Ubah `Tahun Tagihan` sesuai tahun
        - Isi nilai fitur untuk setiap vendor (kolom kuning)
        - Kolom lag **tidak perlu diisi** — dihitung otomatis
        """)

    # ── TAB 2: Upload Data ──────────────────────────────────
    with tab2:
        st.markdown("#### Upload Data Bulanan Baru")

        uploaded = st.file_uploader(
            "Pilih file Excel (Data_Input_Bulan_Ini.xlsx)",
            type=["xlsx"],
            key="uploader"
        )

        if uploaded is not None:
            try:
                df_upload = pd.read_excel(uploaded)

                # Bersihkan baris kosong
                df_upload = df_upload.dropna(subset=["Nama Vendor","Tahun Tagihan"])
                df_upload = df_upload[
                    df_upload["Nama Vendor"].astype(str).str.strip() != ""
                ].reset_index(drop=True)

                st.markdown("**Preview data yang diupload:**")
                st.dataframe(df_upload, use_container_width=True)

                # ── Validasi ──────────────────────────────
                errors = []
                required_cols = (["Nama Vendor","Bulan Tagihan","Tahun Tagihan"]
                                 + FITUR_NON_LAG)
                missing_cols = [c for c in required_cols if c not in df_upload.columns]
                if missing_cols:
                    errors.append(f"Kolom tidak ditemukan: {missing_cols}")

                if "Bulan Tagihan" in df_upload.columns:
                    bulan_invalid = df_upload[
                        ~df_upload["Bulan Tagihan"].str.upper().isin(BULAN_ORDER)
                    ]["Bulan Tagihan"].unique().tolist()
                    if bulan_invalid:
                        errors.append(f"Nilai Bulan tidak valid: {bulan_invalid}")

                if not errors:
                    # Cek duplikat dengan data existing
                    df_upload["Bulan Tagihan"] = (df_upload["Bulan Tagihan"]
                                                   .str.upper().str.strip())
                    if not df.empty:
                        check_dup = df_upload.merge(
                            df[["Nama Vendor","Bulan Tagihan","Tahun Tagihan"]],
                            on=["Nama Vendor","Bulan Tagihan","Tahun Tagihan"],
                            how="inner"
                        )
                        if not check_dup.empty:
                            dup_info = check_dup[["Nama Vendor","Bulan Tagihan",
                                                   "Tahun Tagihan"]].to_dict("records")
                            errors.append(f"Data duplikat ditemukan: {dup_info}")

                if errors:
                    for err in errors:
                        st.error(f"❌ {err}")
                else:
                    st.success("✅ Validasi berhasil! Data siap diintegrasikan.")

                    # Hitung skor dan lag sebelum simpan
                    df_upload["Skor Risiko (Expert Judgment)"] = (
                        (0.4*df_upload["Proportion Delay"] +
                         0.6*df_upload["Proportion Gap"])*100
                    ).round(2)
                    df_upload["Klasifikasi Risiko"] = df_upload[
                        "Skor Risiko (Expert Judgment)"
                    ].apply(hitung_klasifikasi_formula)

                    # Hitung lag otomatis
                    current_df = load_data()
                    lag_rows = []
                    for _, row in df_upload.iterrows():
                        lag_vals, _ = get_lag(
                            current_df,
                            row["Nama Vendor"],
                            row["Bulan Tagihan"],
                            int(row["Tahun Tagihan"])
                        )
                        lag_rows.append(lag_vals)
                    df_lag = pd.DataFrame(lag_rows)
                    df_upload = pd.concat([df_upload, df_lag], axis=1)

                    # Simpan preview prediksi
                    st.markdown("**Hasil klasifikasi & prediksi bulan depan:**")
                    if artifact is not None:
                        pred_rows = []
                        for _, row in df_upload.iterrows():
                            fitur_ini = {f: float(row.get(f,0)) for f in FITUR_NON_LAG}
                            lag_dep   = {lag_c: fitur_ini[src_c]
                                         for lag_c, src_c in LAG_SOURCE_MAP.items()}
                            fitur_p   = {**fitur_ini, **lag_dep}
                            label_p, prob_p = predict_row(artifact, fitur_p)
                            bulan_dep, tahun_dep = bulan_berikutnya(
                                row["Bulan Tagihan"], int(row["Tahun Tagihan"])
                            )
                            pred_rows.append({
                                "Vendor"            : row["Nama Vendor"],
                                "Aktual Bulan Ini"  : row["Klasifikasi Risiko"],
                                "Skor"              : row["Skor Risiko (Expert Judgment)"],
                                f"Prediksi {bulan_dep[:3]} {tahun_dep}": label_p,
                                "Keyakinan"         : f"{prob_p.get(label_p,0):.0%}",
                            })
                        st.dataframe(pd.DataFrame(pred_rows), use_container_width=True)

                    # Tombol konfirmasi simpan
                    if st.button("💾  Simpan & Integrasikan ke Dataset",
                                 type="primary", use_container_width=True):
                        try:
                            if Path(DATA_FILE).exists():
                                df_existing = pd.read_excel(DATA_FILE)
                                # Hanya simpan kolom yang match
                                cols_to_keep = [c for c in df_existing.columns
                                                if c in df_upload.columns]
                                df_to_append = df_upload[cols_to_keep] if cols_to_keep else df_upload
                                df_final = pd.concat([df_existing, df_to_append],
                                                     ignore_index=True)
                            else:
                                df_final = df_upload

                            df_final.to_excel(DATA_FILE, index=False)
                            st.cache_data.clear()
                            st.success(f"✅ Data berhasil disimpan ke {DATA_FILE}! "
                                       f"Semua menu telah diperbarui.")
                            st.balloons()
                        except Exception as e:
                            st.error(f"Gagal menyimpan: {e}")

            except Exception as e:
                st.error(f"Gagal membaca file: {e}")

    # ── TAB 3: Status Dataset ────────────────────────────────
    with tab3:
        st.markdown("#### Status Dataset Saat Ini")
        if df.empty:
            st.warning("Dataset kosong.")
        else:
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Total Observasi", len(df))
            c2.metric("Jumlah Vendor", df["Nama Vendor"].nunique())
            c3.metric("Periode Awal", f"{df['Bulan Tagihan'].iloc[0][:3]} "
                      f"{df['Tahun Tagihan'].iloc[0]}")
            c4.metric("Periode Akhir", f"{df['Bulan Tagihan'].iloc[-1][:3]} "
                      f"{df['Tahun Tagihan'].iloc[-1]}")

            st.markdown("**Distribusi kelas keseluruhan:**")
            dist = df["Klasifikasi Risiko"].value_counts().reset_index()
            dist.columns = ["Kategori","Jumlah"]
            dist["Persentase"] = (dist["Jumlah"]/len(df)*100).round(1).astype(str)+"%"
            st.dataframe(dist, use_container_width=True, hide_index=True)

            st.markdown("**5 data terbaru:**")
            st.dataframe(
                df[["Nama Vendor","Bulan Tagihan","Tahun Tagihan",
                    "Skor Risiko (Expert Judgment)","Klasifikasi Risiko"]].tail(5),
                use_container_width=True, hide_index=True
            )

            st.markdown(f"""
            **Info Model:**
            Model: Random Forest Eks.4 (lag + class weight)
            Path: `{MODEL_PATH}`
            Status: {"✅ Ditemukan" if MODEL_PATH.exists() else "❌ Tidak ditemukan"}
            """)


# ─────────────────────────────────────────────────────────────
# SIDEBAR & ROUTING UTAMA
# ─────────────────────────────────────────────────────────────

def main():
    # ── Session state init ───────────────────────────────────
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    # ── Halaman login ────────────────────────────────────────
    if not st.session_state["logged_in"]:
        show_login()
        return

    # ── Load data & model ────────────────────────────────────
    df       = load_data()
    artifact = load_model()

    # ── Sidebar ──────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <h2 style="color:white;font-size:18px;margin-bottom:2px">📊 VendorRisk</h2>
        <p style="color:#94A3B8;font-size:12px;margin-bottom:20px">
            Monitoring & Prediksi Risiko TAD
        </p>
        """, unsafe_allow_html=True)

        role = st.session_state.get("role","")
        user = st.session_state.get("username","")
        st.markdown(f"""
        <div style="background:#253550;border-radius:8px;padding:10px 12px;margin-bottom:16px">
            <div style="font-size:12px;color:#94A3B8">Login sebagai</div>
            <div style="font-size:14px;color:white;font-weight:500">{user}</div>
            <div style="font-size:11px;color:#64B5F6">{role}</div>
        </div>""", unsafe_allow_html=True)

        # Menu navigasi
        menu_options = ["Dashboard", "Predictions", "Detail"]
        if role == "Admin":
            menu_options.append("Generate")

        menu = st.radio("Menu", menu_options, label_visibility="collapsed")

        # Info model
        if artifact:
            cv_f1 = artifact.get("cv_f1",0)
            st.markdown(f"""
            <div style="background:#253550;border-radius:8px;padding:10px 12px;margin-top:16px">
                <div style="font-size:11px;color:#94A3B8">Model Aktif</div>
                <div style="font-size:12px;color:white">RF Eks.4 (lag + CW)</div>
                <div style="font-size:11px;color:#64B5F6">CV Macro F1: {cv_f1:.4f}</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.warning("Model tidak ditemukan.")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Keluar", use_container_width=True):
            st.session_state.clear()
            st.rerun()

    # ── Header ───────────────────────────────────────────────
    st.markdown(f"""
    <div class="dash-header">
        <div>
            <h2>📊 {menu}</h2>
        </div>
        <span>{datetime.now(timezone(timedelta(hours=7))).strftime('%d %B %Y %H:%M WIB')}
              &nbsp;|&nbsp; {role}</span>
    </div>""", unsafe_allow_html=True)

    # ── Routing halaman ──────────────────────────────────────
    if menu == "Dashboard":
        page_dashboard(df, artifact)
    elif menu == "Predictions":
        page_predictions(df, artifact)
    elif menu == "Detail":
        page_detail(df, artifact)
    elif menu == "Generate":
        page_generate(df, artifact)


if __name__ == "__main__":
    main()
