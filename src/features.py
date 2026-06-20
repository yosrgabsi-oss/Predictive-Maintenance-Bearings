import os
import numpy as np
from scipy import stats
from scipy.fft import fft
from scipy.signal import hilbert

# ============================================================
# CONFIGURATION
# ============================================================
DATA_PATH    = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
RESULTS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
SAMPLE_RATE  = 12000

# ============================================================
# FEATURES TEMPORELLES
# ============================================================
def extract_time_features(window):
    rms        = np.sqrt(np.mean(window**2))
    mean_abs   = np.mean(np.abs(window))
    std        = np.std(window)
    kurtosis   = stats.kurtosis(window)
    skewness   = stats.skew(window)
    peak       = np.max(np.abs(window))
    crest      = peak / (rms + 1e-10)
    shape      = rms / (mean_abs + 1e-10)
    impulse    = peak / (mean_abs + 1e-10)
    return [rms, mean_abs, std, kurtosis, skewness, peak, crest, shape, impulse]

# ============================================================
# FEATURES FRÉQUENTIELLES
# ============================================================
def extract_freq_features(window, sample_rate=SAMPLE_RATE):
    N        = len(window)
    fft_vals = np.abs(fft(window))[:N//2]
    freqs    = np.linspace(0, sample_rate/2, N//2)

    freq_dom = freqs[np.argmax(fft_vals)]

    def band_energy(f_low, f_high):
        mask = (freqs >= f_low) & (freqs < f_high)
        return np.sum(fft_vals[mask]**2)

    e1 = band_energy(0,    1000)
    e2 = band_energy(1000, 2000)
    e3 = band_energy(2000, 3000)
    e4 = band_energy(3000, 6000)

    total    = e1 + e2 + e3 + e4 + 1e-10
    e1, e2, e3, e4 = e1/total, e2/total, e3/total, e4/total

    centroid = np.sum(freqs * fft_vals) / (np.sum(fft_vals) + 1e-10)

    return [freq_dom, e1, e2, e3, e4, centroid]

# ============================================================
# FEATURES ENVELOPPE
# ============================================================
def extract_envelope_features(window):
    envelope = np.abs(hilbert(window))
    env_mean = np.mean(envelope)
    env_std  = np.std(envelope)
    env_kurt = stats.kurtosis(envelope)
    env_rms  = np.sqrt(np.mean(envelope**2))
    return [env_mean, env_std, env_kurt, env_rms]

# ============================================================
# EXTRACTION COMPLÈTE
# ============================================================
def extract_all_features(window):
    return (extract_time_features(window) +
            extract_freq_features(window) +
            extract_envelope_features(window))

FEATURE_NAMES = [
    "RMS", "Mean_Abs", "STD", "Kurtosis", "Skewness",
    "Peak", "Crest_Factor", "Shape_Factor", "Impulse_Factor",
    "Freq_Dom", "Energy_0_1kHz", "Energy_1_2kHz",
    "Energy_2_3kHz", "Energy_3_6kHz", "Spectral_Centroid",
    "Env_Mean", "Env_STD", "Env_Kurtosis", "Env_RMS"
]

# ============================================================
# CONSTRUIRE LE DATASET
# ============================================================
def build_feature_dataset():
    print("=" * 55)
    print("   Extraction des features")
    print("=" * 55)

    X_raw = np.load(os.path.join(DATA_PATH, "X_raw.npy"))
    y     = np.load(os.path.join(DATA_PATH, "y_labels.npy"))

    print(f"\nX_raw charge : {X_raw.shape}")
    print(f"y charge     : {y.shape}")
    print(f"\nExtraction de {len(FEATURE_NAMES)} features par fenetre...")

    X_features = []
    total = len(X_raw)

    for i, window in enumerate(X_raw):
        features = extract_all_features(window)
        X_features.append(features)
        if (i + 1) % 500 == 0 or (i + 1) == total:
            print(f"  Progression : {i+1}/{total} fenetres traitees")

    X_features = np.array(X_features)

    print(f"\nX_features.shape = {X_features.shape}")
    print(f"Nombre de features : {X_features.shape[1]}")
    print(f"\nFeatures extraites :")
    for i, name in enumerate(FEATURE_NAMES):
        print(f"  [{i+1:02d}] {name}")

    return X_features, y

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    # 1. Extraire
    X_features, y = build_feature_dataset()

    # 2. Sauvegarder
    np.save(os.path.join(DATA_PATH, "X_features.npy"), X_features)
    np.save(os.path.join(DATA_PATH, "y_labels.npy"),   y)

    print("\nFichiers sauvegardes :")
    print(f"  data/X_features.npy -> {X_features.shape}")
    print(f"  data/y_labels.npy   -> {y.shape}")

    # 3. Statistiques par classe
    CLASSES = {0: "Normal", 1: "Inner_Race", 2: "Ball", 3: "Outer_Race"}

    print("\nStatistiques RMS par classe :")
    for label, name in CLASSES.items():
        vals = X_features[y == label, 0]
        print(f"  {name:12} | moyenne={vals.mean():.4f} | std={vals.std():.4f}")

    print("\nStatistiques Kurtosis par classe :")
    for label, name in CLASSES.items():
        vals = X_features[y == label, 3]
        print(f"  {name:12} | moyenne={vals.mean():.4f} | std={vals.std():.4f}")

    print("\nEtape 3 terminee ! Pret pour l etape 4 : entrainement du modele.")