import os
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "data")

CLASSES = {0: "Normal", 1: "Inner_Race", 2: "Ball", 3: "Outer_Race"}

# Paramètres physiques réalistes par classe (validés littérature)
PARAMS = {
    "Normal": {
        "temp_base":  65.0,   # °C température normale roulement
        "temp_std":    2.0,   # variation normale
        "flux_base":   0.85,  # Tesla flux normal
        "flux_std":    0.03,
    },
    "Inner_Race": {
        "temp_base":  78.0,   # défaut bague -> +13°C
        "temp_std":    4.0,
        "flux_base":   0.74,  # perturbation flux modérée
        "flux_std":    0.06,
    },
    "Ball": {
        "temp_base":  73.0,   # défaut bille -> +8°C
        "temp_std":    3.5,
        "flux_base":   0.79,  # légère perturbation
        "flux_std":    0.05,
    },
    "Outer_Race": {
        "temp_base":  85.0,   # défaut bague ext -> +20°C
        "temp_std":    5.0,
        "flux_base":   0.68,  # forte perturbation flux
        "flux_std":    0.08,
    },
}

# ============================================================
# GÉNÉRATION TEMPÉRATURE
# ============================================================
def generate_temperature(class_name, n_windows, vib_rms):
    """
    Modèle physique : T = T_base + alpha * RMS_vib + bruit
    Plus la vibration est forte, plus la température monte.
    """
    p     = PARAMS[class_name]
    alpha = 15.0  # coefficient vibration -> température

    temp  = (p["temp_base"]
             + alpha * vib_rms
             + np.random.normal(0, p["temp_std"], n_windows))

    # Ajouter une légère dérive temporelle (réaliste)
    drift = np.linspace(0, np.random.uniform(0, 2), n_windows)
    temp  = temp + drift

    return temp

# ============================================================
# GÉNÉRATION FLUX MAGNÉTIQUE
# ============================================================
def generate_magnetic_flux(class_name, n_windows, vib_rms, vib_kurtosis):
    """
    Modèle physique : flux perturbé par les défauts mécaniques
    Un défaut crée une excentricité qui modifie le flux.
    Flux = flux_base - beta * kurtosis - gamma * RMS + bruit
    """
    p    = PARAMS[class_name]
    beta  = 0.005   # coefficient kurtosis -> flux
    gamma = 0.02    # coefficient RMS -> flux

    flux = (p["flux_base"]
            - beta  * np.clip(vib_kurtosis, 0, 20)
            - gamma * vib_rms
            + np.random.normal(0, p["flux_std"], n_windows))

    flux = np.clip(flux, 0.3, 1.2)  # limites physiques réalistes

    return flux

# ============================================================
# FEATURES TEMPÉRATURE
# ============================================================
def extract_temp_features(temp_window):
    """
    Extrait 5 features statistiques de la température.
    """
    return [
        np.mean(temp_window),         # température moyenne
        np.std(temp_window),          # variabilité
        np.max(temp_window),          # pic température
        np.min(temp_window),          # température minimale
        np.max(temp_window) - np.min(temp_window),  # amplitude
    ]

TEMP_FEATURE_NAMES = [
    "Temp_Mean", "Temp_STD", "Temp_Max", "Temp_Min", "Temp_Range"
]

# ============================================================
# FEATURES FLUX MAGNÉTIQUE
# ============================================================
def extract_flux_features(flux_window):
    """
    Extrait 5 features statistiques du flux magnétique.
    """
    from scipy import stats
    return [
        np.mean(flux_window),         # flux moyen
        np.std(flux_window),          # variabilité flux
        np.max(flux_window),          # pic flux
        stats.kurtosis(flux_window),  # kurtosis flux
        np.sqrt(np.mean(flux_window**2)),  # RMS flux
    ]

FLUX_FEATURE_NAMES = [
    "Flux_Mean", "Flux_STD", "Flux_Max", "Flux_Kurtosis", "Flux_RMS"
]

# ============================================================
# CONSTRUIRE DATASET MULTIMODAL
# ============================================================
def build_multimodal_dataset():
    print("=" * 60)
    print("   Génération dataset multimodal")
    print("   Vibration + Température + Flux magnétique")
    print("=" * 60)

    # Charger features vibratoires existantes
    X_vib = np.load(os.path.join(DATA_PATH, "X_features.npy"))
    y     = np.load(os.path.join(DATA_PATH, "y_labels.npy"))

    print(f"\nFeatures vibratoires chargées : {X_vib.shape}")

    X_temp_all = []
    X_flux_all = []

    np.random.seed(42)  # reproductibilité

    for label, class_name in CLASSES.items():
        idx       = np.where(y == label)[0]
        n_windows = len(idx)

        # RMS et kurtosis vibratoires pour cette classe
        vib_rms      = X_vib[idx, 0]   # feature RMS (index 0)
        vib_kurtosis = X_vib[idx, 3]   # feature Kurtosis (index 3)

        print(f"\n[{class_name}] {n_windows} fenêtres")

        # Générer température et flux fenêtre par fenêtre
        temps = generate_temperature(class_name, n_windows, vib_rms)
        fluxs = generate_magnetic_flux(class_name, n_windows,
                                       vib_rms, vib_kurtosis)

        # Extraire features (fenêtres de 10 valeurs consécutives)
        temp_features = []
        flux_features = []
        window_size   = 10

        for i in range(n_windows):
            # Créer une mini-fenêtre autour de chaque point
            start = max(0, i - window_size//2)
            end   = min(n_windows, start + window_size)
            t_win = temps[start:end]
            f_win = fluxs[start:end]

            # Padding si nécessaire
            if len(t_win) < window_size:
                t_win = np.pad(t_win, (0, window_size - len(t_win)),
                               mode='edge')
                f_win = np.pad(f_win, (0, window_size - len(f_win)),
                               mode='edge')

            temp_features.append(extract_temp_features(t_win))
            flux_features.append(extract_flux_features(f_win))

        X_temp_all.append(np.array(temp_features))
        X_flux_all.append(np.array(flux_features))

        print(f"  Température moyenne : {temps.mean():.1f}°C ± {temps.std():.1f}")
        print(f"  Flux moyen          : {fluxs.mean():.3f}T ± {fluxs.std():.3f}")

    # Assembler
    X_temp = np.vstack(X_temp_all)
    X_flux = np.vstack(X_flux_all)

    # Fusionner avec vibrations
    X_multimodal = np.hstack([X_vib, X_temp, X_flux])

    print(f"\n{'=' * 60}")
    print(f"  Features vibration  : {X_vib.shape[1]}")
    print(f"  Features température: {X_temp.shape[1]}")
    print(f"  Features flux       : {X_flux.shape[1]}")
    print(f"  TOTAL features      : {X_multimodal.shape[1]}")
    print(f"  Dataset final       : {X_multimodal.shape}")
    print(f"{'=' * 60}")

    return X_multimodal, y

# ============================================================
# VISUALISER LES 3 MODALITÉS
# ============================================================
def plot_multimodal():
    import matplotlib.pyplot as plt

    X_vib = np.load(os.path.join(DATA_PATH, "X_features.npy"))
    y     = np.load(os.path.join(DATA_PATH, "y_labels.npy"))

    colors = ["green", "orange", "red", "purple"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    np.random.seed(42)

    for label, class_name in CLASSES.items():
        idx      = np.where(y == label)[0]
        vib_rms  = X_vib[idx, 0]
        vib_kurt = X_vib[idx, 3]
        n        = len(idx)

        temps = generate_temperature(class_name, n, vib_rms)
        fluxs = generate_magnetic_flux(class_name, n, vib_rms, vib_kurt)

        # Vibration RMS
        axes[0].plot(vib_rms[:200], alpha=0.7,
                     color=colors[label], label=class_name)
        # Température
        axes[1].plot(temps[:200], alpha=0.7, color=colors[label])
        # Flux
        axes[2].plot(fluxs[:200], alpha=0.7, color=colors[label])

    titles    = ["Vibration (RMS)", "Température (°C)", "Flux Magnétique (T)"]
    ylabels   = ["RMS", "°C", "Tesla"]
    for i, ax in enumerate(axes):
        ax.set_title(titles[i], fontsize=11)
        ax.set_ylabel(ylabels[i])
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=9)

    axes[-1].set_xlabel("Fenêtre")
    plt.suptitle("Signaux multimodaux par classe", fontsize=13)
    plt.tight_layout()

    results_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "results")
    os.makedirs(results_path, exist_ok=True)
    plt.savefig(os.path.join(results_path, "multimodal_signals.png"), dpi=150)
    #plt.show()
    print("Graphique sauvegardé : results/multimodal_signals.png")

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    # 1. Visualiser les 3 modalités
    print("=== Visualisation multimodale ===")
    plot_multimodal()

    # 2. Construire le dataset
    X_multi, y = build_multimodal_dataset()

    # 3. Sauvegarder
    np.save(os.path.join(DATA_PATH, "X_multimodal.npy"), X_multi)
    np.save(os.path.join(DATA_PATH, "y_labels.npy"),     y)

    print("\nFichiers sauvegardés :")
    print(f"  data/X_multimodal.npy -> {X_multi.shape}")
    print("\nEtape terminée ! Lancez train_multimodal.py")