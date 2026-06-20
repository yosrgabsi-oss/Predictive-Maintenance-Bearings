import os
import numpy as np
import scipy.io as sio
import joblib
from scipy import stats
from scipy.fft import fft
from scipy.signal import hilbert
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ============================================================
# CONFIGURATION
# ============================================================
DATA_PATH    = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
MODELS_PATH  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
RESULTS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

CLASSES     = {0: "Normal", 1: "Inner_Race", 2: "Ball", 3: "Outer_Race"}
SAMPLE_RATE = 12000
WINDOW_SIZE = 1024
STEP        = 512

# Couleurs et alertes par classe
ALERTES = {
    "Normal":     {"couleur": "green",  "niveau": "OK",      "message": "Roulement en bon etat"},
    "Inner_Race": {"couleur": "orange", "niveau": "ALERTE",  "message": "Defaut bague interieure detecte !"},
    "Ball":       {"couleur": "orange", "niveau": "ALERTE",  "message": "Defaut de bille detecte !"},
    "Outer_Race": {"couleur": "red",    "niveau": "CRITIQUE","message": "Defaut bague exterieure detecte !"},
}

# ============================================================
# EXTRACTION DES FEATURES (identique à features.py)
# ============================================================
def extract_all_features(window):
    # Temporelles
    rms      = np.sqrt(np.mean(window**2))
    mean_abs = np.mean(np.abs(window))
    std      = np.std(window)
    kurtosis = stats.kurtosis(window)
    skewness = stats.skew(window)
    peak     = np.max(np.abs(window))
    crest    = peak / (rms + 1e-10)
    shape    = rms / (mean_abs + 1e-10)
    impulse  = peak / (mean_abs + 1e-10)

    # Fréquentielles
    N        = len(window)
    fft_vals = np.abs(fft(window))[:N//2]
    freqs    = np.linspace(0, SAMPLE_RATE/2, N//2)
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

    # Enveloppe
    envelope = np.abs(hilbert(window))
    env_mean = np.mean(envelope)
    env_std  = np.std(envelope)
    env_kurt = stats.kurtosis(envelope)
    env_rms  = np.sqrt(np.mean(envelope**2))

    return [rms, mean_abs, std, kurtosis, skewness, peak, crest, shape,
            impulse, freq_dom, e1, e2, e3, e4, centroid,
            env_mean, env_std, env_kurt, env_rms]

# ============================================================
# CHARGER LE MODÈLE
# ============================================================
def load_model():
    model_path  = os.path.join(MODELS_PATH, "best_model.pkl")
    scaler_path = os.path.join(MODELS_PATH, "scaler.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modele introuvable : {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler introuvable : {scaler_path}")

    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("Modele et scaler charges avec succes !")
    return model, scaler

# ============================================================
# PRÉDIRE SUR UN FICHIER .mat
# ============================================================
def predict_file(filepath, model, scaler):
    # Charger le signal
    mat_data   = sio.loadmat(filepath)
    signal_key = None
    for key in mat_data.keys():
        if "DE_time" in key:
            signal_key = key
            break

    if signal_key is None:
        raise ValueError(f"Clé DE_time introuvable dans {filepath}")

    signal = mat_data[signal_key].flatten()
    print(f"\nFichier      : {os.path.basename(filepath)}")
    print(f"Signal       : {len(signal)} points ({len(signal)/SAMPLE_RATE:.2f} sec)")

    # Segmenter en fenêtres
    windows = []
    start = 0
    while start + WINDOW_SIZE <= len(signal):
        windows.append(signal[start : start + WINDOW_SIZE])
        start += STEP
    windows = np.array(windows)
    print(f"Fenetres     : {len(windows)}")

    # Extraire features
    features = np.array([extract_all_features(w) for w in windows])
    features = scaler.transform(features)

    # Prédire
    predictions  = model.predict(features)
    probabilites = model.predict_proba(features)

    # Classe majoritaire
    unique, counts = np.unique(predictions, return_counts=True)
    classe_idx     = unique[np.argmax(counts)]
    classe_name    = CLASSES[classe_idx]
    confiance      = np.max(counts) / len(predictions) * 100
    prob_moyenne   = probabilites.mean(axis=0)

    return {
        "signal"       : signal,
        "predictions"  : predictions,
        "classe"       : classe_name,
        "classe_idx"   : classe_idx,
        "confiance"    : confiance,
        "prob_moyenne" : prob_moyenne,
        "nb_fenetres"  : len(windows),
    }

# ============================================================
# AFFICHER LE RÉSULTAT
# ============================================================
def afficher_resultat(result, filepath):
    classe   = result["classe"]
    alerte   = ALERTES[classe]
    confiance = result["confiance"]

    print("\n" + "=" * 55)
    print(f"  DIAGNOSTIC ROULEMENT")
    print("=" * 55)
    print(f"  Etat detecte  : {classe}")
    print(f"  Niveau        : {alerte['niveau']}")
    print(f"  Message       : {alerte['message']}")
    print(f"  Confiance     : {confiance:.1f}%")
    print("\n  Probabilites par classe :")
    for i, prob in enumerate(result["prob_moyenne"]):
        print(f"    {CLASSES[i]:12} : {prob*100:.1f}%")
    print("=" * 55)

# ============================================================
# VISUALISATION COMPLÈTE
# ============================================================
def plot_diagnostic(result, filepath):
    os.makedirs(RESULTS_PATH, exist_ok=True)
    classe  = result["classe"]
    alerte  = ALERTES[classe]
    signal  = result["signal"]
    fig     = plt.figure(figsize=(14, 10))
    fig.suptitle(f"Diagnostic Roulement - {os.path.basename(filepath)}",
                 fontsize=14, fontweight="bold")

    # 1. Signal temporel
    ax1 = fig.add_subplot(3, 2, (1, 2))
    t   = np.arange(len(signal)) / SAMPLE_RATE
    ax1.plot(t[:3000], signal[:3000], linewidth=0.5, color="steelblue")
    ax1.set_title("Signal vibratoire (3000 premiers points)")
    ax1.set_xlabel("Temps (s)")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, alpha=0.3)

    # 2. FFT
    ax2  = fig.add_subplot(3, 2, 3)
    N    = len(signal[:WINDOW_SIZE])
    fft_vals = np.abs(fft(signal[:WINDOW_SIZE]))[:N//2]
    freqs    = np.linspace(0, SAMPLE_RATE/2, N//2)
    ax2.plot(freqs, fft_vals, linewidth=0.7, color="darkorange")
    ax2.set_title("Spectre FFT")
    ax2.set_xlabel("Frequence (Hz)")
    ax2.set_ylabel("Amplitude")
    ax2.grid(True, alpha=0.3)

    # 3. Probabilités
    ax3    = fig.add_subplot(3, 2, 4)
    labels = list(CLASSES.values())
    probs  = result["prob_moyenne"] * 100
    colors = ["green", "orange", "orange", "red"]
    bars   = ax3.bar(labels, probs, color=colors, edgecolor="white")
    for bar, prob in zip(bars, probs):
        ax3.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.5,
                 f"{prob:.1f}%",
                 ha="center", va="bottom", fontsize=9)
    ax3.set_title("Probabilites par classe")
    ax3.set_ylabel("Probabilite (%)")
    ax3.set_ylim(0, 110)
    ax3.grid(True, alpha=0.3, axis="y")

    # 4. Evolution temporelle des prédictions
    ax4  = fig.add_subplot(3, 2, (5, 6))
    preds = result["predictions"]
    color_map = {0: "green", 1: "orange", 2: "gold", 3: "red"}
    pred_colors = [color_map[p] for p in preds]
    ax4.bar(range(len(preds)), preds, color=pred_colors, width=1.0)
    ax4.set_title("Evolution des predictions par fenetre")
    ax4.set_xlabel("Fenetre")
    ax4.set_ylabel("Classe predite")
    ax4.set_yticks([0, 1, 2, 3])
    ax4.set_yticklabels(["Normal", "Inner", "Ball", "Outer"])
    ax4.grid(True, alpha=0.3, axis="y")

    # Bandeau de diagnostic
    couleur = alerte["couleur"]
    fig.patches.append(mpatches.FancyBboxPatch(
        (0.1, 0.01), 0.8, 0.06,
        boxstyle="round,pad=0.01",
        facecolor=couleur, alpha=0.2,
        transform=fig.transFigure
    ))
    fig.text(0.5, 0.04,
             f"{alerte['niveau']} - {alerte['message']} (confiance: {result['confiance']:.1f}%)",
             ha="center", fontsize=12,
             color=couleur, fontweight="bold")

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    filename = f"diagnostic_{os.path.basename(filepath).replace('.mat', '')}.png"
    plt.savefig(os.path.join(RESULTS_PATH, filename), dpi=150)
    plt.show()
    print(f"Graphique sauvegarde : results/{filename}")

# ============================================================
# TESTER TOUS LES FICHIERS
# ============================================================
def tester_tous_fichiers(model, scaler):
    print("\n" + "=" * 55)
    print("   Test sur tous les fichiers")
    print("=" * 55)

    FICHIERS_TEST = {
        "Normal"    : "Normal_3.mat",
        "Inner_Race": "IR007_3.mat",
        "Ball"      : "B007_3.mat",
        "Outer_Race": "OR007_3.mat",
    }

    resultats = []
    for classe, fichier in FICHIERS_TEST.items():
        filepath = os.path.join(DATA_PATH, classe, fichier)
        if not os.path.exists(filepath):
            print(f"  MANQUANT : {filepath}")
            continue

        result   = predict_file(filepath, model, scaler)
        correct  = result["classe"] == classe
        statut   = "CORRECT" if correct else "ERREUR"

        print(f"\n  [{statut}] Vrai={classe:12} | Predit={result['classe']:12} | "
              f"Confiance={result['confiance']:.1f}%")
        resultats.append(correct)

    precision = sum(resultats) / len(resultats) * 100
    print(f"\n  Precision globale : {precision:.0f}% ({sum(resultats)}/{len(resultats)})")

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    # 1. Charger modèle
    model, scaler = load_model()

    # 2. Tester tous les fichiers
    tester_tous_fichiers(model, scaler)

    # 3. Diagnostic détaillé sur un fichier Inner Race
    print("\n" + "=" * 55)
    print("   Diagnostic detaille : Inner Race")
    print("=" * 55)
    filepath = os.path.join(DATA_PATH, "Inner_Race", "IR007_0.mat")
    result   = predict_file(filepath, model, scaler)
    afficher_resultat(result, filepath)
    plot_diagnostic(result, filepath)

    print("\nEtape 5 terminee ! Projet de maintenance predictive complet.")