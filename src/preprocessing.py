import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
RESULTS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
WINDOW_SIZE  = 1024
OVERLAP      = 0.5
STEP         = int(WINDOW_SIZE * (1 - OVERLAP))  # 512

CLASSES = {
    "Normal":     0,
    "Inner_Race": 1,
    "Ball":       2,
    "Outer_Race": 3,
}

# ============================================================
# ÉTAPE 1 : Charger un fichier .mat
# ============================================================
def load_mat_file(filepath):
    mat_data = sio.loadmat(filepath)

    # Chercher la clé DE_time automatiquement
    signal_key = None
    for key in mat_data.keys():
        if "DE_time" in key:
            signal_key = key
            break

    if signal_key is None:
        raise ValueError(f"Clé DE_time introuvable dans {filepath}")

    signal = mat_data[signal_key].flatten()
    return signal


# ============================================================
# ÉTAPE 2 : Segmenter en fenêtres glissantes
# ============================================================
def segment_signal(signal, window_size=WINDOW_SIZE, step=STEP):
    windows = []
    start = 0
    while start + window_size <= len(signal):
        windows.append(signal[start : start + window_size])
        start += step
    return np.array(windows)


# ============================================================
# ÉTAPE 3 : Normaliser chaque fenêtre
# ============================================================
def normalize_windows(windows):
    normalized = []
    for w in windows:
        std = np.std(w)
        if std > 0:
            normalized.append((w - np.mean(w)) / std)
        else:
            normalized.append(w)
    return np.array(normalized)


# ============================================================
# ÉTAPE 4 : Construire le dataset complet
# ============================================================
def build_dataset():
    X_all = []
    y_all = []

    print("=" * 55)
    print("   Construction du dataset")
    print("=" * 55)

    for class_name, label in CLASSES.items():
        folder = os.path.join(DATA_PATH, class_name)
        mat_files = sorted([f for f in os.listdir(folder) if f.endswith(".mat")])

        print(f"\n[{class_name}] label={label} | {len(mat_files)} fichiers")

        for mat_file in mat_files:
            filepath = os.path.join(folder, mat_file)
            try:
                signal  = load_mat_file(filepath)
                windows = segment_signal(signal)
                windows = normalize_windows(windows)

                X_all.append(windows)
                y_all.append(np.full(len(windows), label))

                print(f"  {mat_file} -> {len(windows)} fenêtres | signal={len(signal)} pts")

            except Exception as e:
                print(f"  ERREUR {mat_file} : {e}")

    X = np.vstack(X_all)
    y = np.concatenate(y_all)

    print("\n" + "=" * 55)
    print(f"  X.shape = {X.shape}")
    print(f"  y.shape = {y.shape}")
    unique, counts = np.unique(y, return_counts=True)
    for u, c in zip(unique, counts):
        nom = [k for k, v in CLASSES.items() if v == u][0]
        print(f"  Classe {u} ({nom}) : {c} fenêtres")
    print("=" * 55)

    return X, y


# ============================================================
# ÉTAPE 5 : Visualiser les 4 classes
# ============================================================
def plot_comparison():
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    colors = ["green", "orange", "red", "purple"]

    for i, (class_name, label) in enumerate(CLASSES.items()):
        folder   = os.path.join(DATA_PATH, class_name)
        mat_files = sorted([f for f in os.listdir(folder) if f.endswith(".mat")])
        filepath  = os.path.join(folder, mat_files[0])

        signal = load_mat_file(filepath)
        axes[i].plot(signal[:2048], linewidth=0.6, color=colors[i])
        axes[i].set_title(f"{class_name}", fontsize=11)
        axes[i].set_ylabel("Amplitude")
        axes[i].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Échantillons")
    plt.suptitle("Comparaison des signaux vibratoires - 4 classes", fontsize=13)
    plt.tight_layout()
    os.makedirs(RESULTS_PATH, exist_ok=True)
    plt.savefig(os.path.join(RESULTS_PATH, "comparaison_signaux.png"), dpi=150)
    #plt.show()
    print("  Graphique sauvegardé : results/comparaison_signaux.png")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    # 1. Visualisation
    print("=== Visualisation des signaux ===")
    plot_comparison()

    # 2. Construction dataset
    X, y = build_dataset()

    # 3. Sauvegarde
    os.makedirs("data", exist_ok=True)
    np.save(os.path.join(DATA_PATH, "X_raw.npy"), X)
    np.save(os.path.join(DATA_PATH, "y_labels.npy"), y)

    print("\nFichiers sauvegardés :")
    print(f"  data/X_raw.npy    -> {X.shape}")
    print(f"  data/y_labels.npy -> {y.shape}")
    print("\nEtape 2 terminée ! Prêt pour l'étape 3 : extraction de features.")