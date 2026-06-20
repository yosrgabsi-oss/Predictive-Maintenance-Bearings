import os
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score

# ============================================================
# CONFIGURATION
# ============================================================
DATA_PATH    = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
MODELS_PATH  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
RESULTS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
CLASS_NAMES  = ["Normal", "Inner_Race", "Ball", "Outer_Race"]

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("   Entraînement modèle MULTIMODAL")
    print("   Vibration + Température + Flux magnétique")
    print("=" * 60)

    # 1. Charger
    X = np.load(os.path.join(DATA_PATH, "X_multimodal.npy"))
    y = np.load(os.path.join(DATA_PATH, "y_labels.npy"))
    print(f"\nDataset : {X.shape} | {len(np.unique(y))} classes")
    print(f"Features: 19 vibration + 5 température + 5 flux = {X.shape[1]} total")

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # 3. Normaliser
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # 4. Entraîner
    print("\nEntraînement Random Forest multimodal...")
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    # 5. Évaluer
    y_pred   = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy multimodal : {accuracy*100:.2f}%")
    print("\nRapport de classification :")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    # 6. Matrice de confusion
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred),
                           display_labels=CLASS_NAMES).plot(ax=ax, cmap="Blues")
    ax.set_title("Matrice de confusion - Modèle Multimodal")
    plt.tight_layout()
    os.makedirs(RESULTS_PATH, exist_ok=True)
    plt.savefig(os.path.join(RESULTS_PATH, "confusion_multimodal.png"), dpi=150)
    #plt.show()

    # 7. Importance des features
    feature_names = (
        [f"Vib_{i}" for i in range(19)] +
        ["Temp_Mean", "Temp_STD", "Temp_Max", "Temp_Min", "Temp_Range"] +
        ["Flux_Mean", "Flux_STD", "Flux_Max", "Flux_Kurtosis", "Flux_RMS"]
    )
    importances = rf.feature_importances_
    indices     = np.argsort(importances)[::-1][:15]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(15), importances[indices],
           color=["steelblue" if "Vib" in feature_names[i]
                  else "tomato" if "Temp" in feature_names[i]
                  else "gold" for i in indices])
    ax.set_xticks(range(15))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha="right")
    ax.set_title("Top 15 features les plus importantes")
    ax.set_ylabel("Importance")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, "feature_importance_multimodal.png"), dpi=150)
    #plt.show()

    # 8. Sauvegarder
    os.makedirs(MODELS_PATH, exist_ok=True)
    joblib.dump(rf,     os.path.join(MODELS_PATH, "model_multimodal.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_PATH, "scaler_multimodal.pkl"))
    print(f"\nModèle sauvegardé : models/model_multimodal.pkl")
    print("\nProjet multimodal terminé !")
