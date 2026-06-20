import os
import numpy as np
import scipy.io as sio
import joblib
import streamlit as st
import matplotlib.pyplot as plt
from scipy import stats
from scipy.fft import fft
from scipy.signal import hilbert

# ============================================================
# CONFIGURATION & PARAMÈTRES PHYSIQUES (Simulation)
# ============================================================
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_PATH, "data")
MODELS_PATH = os.path.join(BASE_PATH, "models")

CLASSES = {0: "Normal", 1: "Inner_Race", 2: "Ball", 3: "Outer_Race"}
SAMPLE_RATE = 12000
WINDOW_SIZE = 1024
STEP = 512

# Paramètres issus de generate_multimodal.py pour la simulation
PARAMS_PHYSIQUES = {
    "Normal": {"temp_base": 65.0, "temp_std": 2.0, "flux_base": 0.85, "flux_std": 0.03},
    "Inner_Race": {"temp_base": 78.0, "temp_std": 4.0, "flux_base": 0.74, "flux_std": 0.06},
    "Ball": {"temp_base": 73.0, "temp_std": 3.5, "flux_base": 0.79, "flux_std": 0.05},
    "Outer_Race": {"temp_base": 85.0, "temp_std": 5.0, "flux_base": 0.68, "flux_std": 0.08},
}

ALERTES = {
    "Normal": {"couleur": "#28a745", "niveau": "OK", "emoji": "✅", "message": "Roulement en bon état"},
    "Inner_Race": {"couleur": "#fd7e14", "niveau": "ALERTE", "emoji": "⚠️",
                   "message": "Défaut bague intérieure détecté !"},
    "Ball": {"couleur": "#ffc107", "niveau": "ALERTE", "emoji": "⚠️", "message": "Défaut de bille détecté !"},
    "Outer_Race": {"couleur": "#dc3545", "niveau": "CRITIQUE", "emoji": "🚨",
                   "message": "Défaut bague extérieure détecté !"},
}


# ============================================================
# EXTRACTION FEATURES (Vibration + Temp + Flux)
# ============================================================
def extract_vibration_features(window):
    rms = np.sqrt(np.mean(window ** 2))
    mean_abs = np.mean(np.abs(window))
    std = np.std(window)
    kurt = stats.kurtosis(window)
    skew = stats.skew(window)
    peak = np.max(np.abs(window))
    crest = peak / (rms + 1e-10)
    shape = rms / (mean_abs + 1e-10)
    impulse = peak / (mean_abs + 1e-10)

    N = len(window)
    fft_vals = np.abs(fft(window))[:N // 2]
    freqs = np.linspace(0, SAMPLE_RATE / 2, N // 2)
    freq_dom = freqs[np.argmax(fft_vals)]

    def band_energy(f_low, f_high):
        mask = (freqs >= f_low) & (freqs < f_high)
        return np.sum(fft_vals[mask] ** 2)

    e1, e2, e3, e4 = band_energy(0, 1000), band_energy(1000, 2000), band_energy(2000, 3000), band_energy(3000, 6000)
    total = e1 + e2 + e3 + e4 + 1e-10
    centroid = np.sum(freqs * fft_vals) / (np.sum(fft_vals) + 1e-10)

    envelope = np.abs(hilbert(window))
    return [rms, mean_abs, std, kurt, skew, peak, crest, shape, impulse, freq_dom,
            e1 / total, e2 / total, e3 / total, e4 / total, centroid,
            np.mean(envelope), np.std(envelope), stats.kurtosis(envelope), np.sqrt(np.mean(envelope ** 2))]


def extract_stat_features(data_window):
    """Features pour Température et Flux"""
    return [np.mean(data_window), np.std(data_window), np.max(data_window),
            np.min(data_window), np.max(data_window) - np.min(data_window)]


# ============================================================
# LOGIQUE DE PRÉDICTION MULTIMODALE
# ============================================================
def predict_multimodal(signal, class_selection, model, scaler):
    # 1. Fenêtrage et features vibratoires
    windows = [signal[i: i + WINDOW_SIZE] for i in range(0, len(signal) - WINDOW_SIZE, STEP)]
    X_vib = np.array([extract_vibration_features(w) for w in windows])

    # 2. Simulation Température & Flux (basée sur la vibration et la classe sélectionnée)
    # On simule ce que les capteurs liraient pour ce type de panne
    p = PARAMS_PHYSIQUES[class_selection]
    n_wins = len(X_vib)
    vib_rms = X_vib[:, 0]
    vib_kurt = X_vib[:, 3]

    temps = (p["temp_base"] + 15.0 * vib_rms + np.random.normal(0, p["temp_std"], n_wins))
    fluxs = np.clip(
        p["flux_base"] - 0.005 * np.clip(vib_kurt, 0, 20) - 0.02 * vib_rms + np.random.normal(0, p["flux_std"], n_wins),
        0.3, 1.2)

    # 3. Features Temp & Flux
    X_temp = []
    X_flux = []
    for i in range(n_wins):
        start, end = max(0, i - 5), min(n_wins, i + 5)
        X_temp.append(extract_stat_features(temps[start:end]))
        X_flux.append(extract_stat_features(fluxs[start:end]))

    # 4. Fusion et Prédiction
    X_total = np.hstack([X_vib, np.array(X_temp), np.array(X_flux)])
    X_scaled = scaler.transform(X_total)

    preds = model.predict(X_scaled)
    probs = model.predict_proba(X_scaled).mean(axis=0)

    unique, counts = np.unique(preds, return_counts=True)
    classe_idx = unique[np.argmax(counts)]

    return {
        "classe": CLASSES[classe_idx],
        "confiance": np.max(counts) / len(preds) * 100,
        "probs": probs,
        "predictions": preds,
        "temp_curve": temps,
        "flux_curve": fluxs,
        "vib_rms": vib_rms
    }


# ============================================================
# INTERFACE STREAMLIT
# ============================================================
st.set_page_config(page_title="Maintenance Multimodale", page_icon="⚙️", layout="wide")


@st.cache_resource
def load_multimodal_assets():
    m = joblib.load(os.path.join(MODELS_PATH, "model_multimodal.pkl"))
    s = joblib.load(os.path.join(MODELS_PATH, "scaler_multimodal.pkl"))
    return m, s


st.title("⚙️ Surveillance Multimodale Haute Précision")
st.markdown("*Fusion Vibration + Thermographie + Flux Magnétique*")

try:
    model, scaler = load_multimodal_assets()
    st.sidebar.success("✅ Modèle Multimodal (29 features) chargé")
except Exception as e:
    st.sidebar.error(f"❌ Erreur : {e}")
    st.stop()

# Sidebar
st.sidebar.header("📂 Données d'entrée")
dossier_choisi = st.sidebar.selectbox("Type de panne (Source)", list(PARAMS_PHYSIQUES.keys()))
fichiers = sorted([f for f in os.listdir(os.path.join(DATA_PATH, dossier_choisi)) if f.endswith(".mat")])
fichier_choisi = st.sidebar.selectbox("Fichier signal", fichiers)

if st.sidebar.button("🔍 Lancer l'Analyse Fusionnée", use_container_width=True):
    filepath = os.path.join(DATA_PATH, dossier_choisi, fichier_choisi)
    mat_data = sio.loadmat(filepath)
    signal_key = [k for k in mat_data.keys() if "DE_time" in k][0]
    signal = mat_data[signal_key].flatten()

    with st.spinner("Fusion des modalités en cours..."):
        res = predict_multimodal(signal, dossier_choisi, model, scaler)

    # Diagnostic
    alerte = ALERTES[res["classe"]]
    st.markdown(f"""
        <div style="background-color:{alerte['couleur']}22; border-left: 6px solid {alerte['couleur']}; padding:20px; border-radius:8px;">
            <h2 style="color:{alerte['couleur']}; margin:0;">{alerte['emoji']} {alerte['message']}</h2>
            <p>Diagnostic : <b>{res['classe']}</b> | Confiance : <b>{res['confiance']:.1f}%</b></p>
        </div>""", unsafe_allow_html=True)

    # Métriques
    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Température Moyenne", f"{res['temp_curve'].mean():.1f} °C", f"{res['temp_curve'].mean() - 65:.1f} °C")
    m2.metric("Flux Magnétique", f"{res['flux_curve'].mean():.3f} T")
    m3.metric("Vibration RMS", f"{res['vib_rms'].mean():.3f} g")

    # Graphiques
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🌡️ Profil Thermique")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(res['temp_curve'], color="tomato", lw=1.5)
        ax.set_ylabel("Température (°C)")
        ax.grid(alpha=0.3)
        st.pyplot(fig)

    with c2:
        st.subheader("🧲 Flux Magnétique")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(res['flux_curve'], color="gold", lw=1.5)
        ax.set_ylabel("Tesla (T)")
        ax.grid(alpha=0.3)
        st.pyplot(fig)

    # Probabilités
    st.subheader("🎯 Analyse de probabilités par modalité")
    cols = st.columns(4)
    for i, (name, prob) in enumerate(zip(CLASSES.values(), res["probs"])):
        cols[i].progress(float(prob))
        cols[i].write(f"{name}: {prob * 100:.1f}%")

else:
    st.info("Sélectionnez un signal pour démarrer l'analyse multimodale.")