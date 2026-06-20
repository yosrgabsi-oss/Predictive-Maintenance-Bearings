import os
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, accuracy_score)

# ============================================================
# CONFIGURATION
# ============================================================
DATA_PATH    = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
MODELS_PATH  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
RESULTS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

CLASSES      = {0: "Normal", 1: "Inner_Race", 2: "Ball", 3: "Outer_Race"}
CLASS_NAMES  = ["Normal", "Inner_Race", "Ball", "Outer_Race"]

# ============================================================
# ÉTAPE 1 : Charger les données
# ============================================================
def load_data():
    print("=" * 55)
    print("   Chargement des donnees")
    print("=" * 55)

    X = np.load(os.path.join(DATA_PATH, "X_features.npy"))
    y = np.load(os.path.join(DATA_PATH, "y_labels.npy"))

    print(f"\nX.shape = {X.shape}")
    print(f"y.shape = {y.shape}")

    unique, counts = np.unique(y, return_counts=True)
    print("\nDistribution des classes :")
    for u, c in zip(unique, counts):
        print(f"  {CLASSES[u]:12} : {c} echantillons ({100*c/len(y):.1f}%)")

    return X, y

# ============================================================
# ÉTAPE 2 : Préparer les données
# ============================================================
def prepare_data(X, y):
    print("\n" + "=" * 55)
    print("   Preparation des donnees")
    print("=" * 55)

    # Split train/test 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Normalisation
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print(f"\nTrain : {X_train.shape[0]} echantillons")
    print(f"Test  : {X_test.shape[0]} echantillons")
    print(f"Ratio : 80% train / 20% test")

    # Sauvegarder le scaler
    os.makedirs(MODELS_PATH, exist_ok=True)
    joblib.dump(scaler, os.path.join(MODELS_PATH, "scaler.pkl"))
    print(f"\nScaler sauvegarde : models/scaler.pkl")

    return X_train, X_test, y_train, y_test, scaler

# ============================================================
# ÉTAPE 3 : Entraîner Random Forest
# ============================================================
def train_random_forest(X_train, y_train):
    print("\n" + "=" * 55)
    print("   Entrainement Random Forest")
    print("=" * 55)

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    )

    print("\nEntrainement en cours...")
    rf.fit(X_train, y_train)
    print("Entrainement termine !")

    # Cross-validation
    print("\nCross-validation (5 folds)...")
    cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring="accuracy")
    print(f"  Scores : {[f'{s:.4f}' for s in cv_scores]}")
    print(f"  Moyenne : {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    return rf

# ============================================================
# ÉTAPE 4 : Entraîner SVM
# ============================================================
def train_svm(X_train, y_train):
    print("\n" + "=" * 55)
    print("   Entrainement SVM")
    print("=" * 55)

    svm = SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        random_state=42,
        probability=True
    )

    print("\nEntrainement en cours...")
    svm.fit(X_train, y_train)
    print("Entrainement termine !")

    # Cross-validation
    print("\nCross-validation (5 folds)...")
    cv_scores = cross_val_score(svm, X_train, y_train, cv=5, scoring="accuracy")
    print(f"  Scores : {[f'{s:.4f}' for s in cv_scores]}")
    print(f"  Moyenne : {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    return svm

# ============================================================
# ÉTAPE 5 : Évaluer un modèle
# ============================================================
def evaluate_model(model, X_test, y_test, model_name):
    print(f"\n{'=' * 55}")
    print(f"   Evaluation : {model_name}")
    print("=" * 55)

    y_pred   = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nAccuracy : {accuracy*100:.2f}%")
    print("\nRapport de classification :")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    return accuracy, y_pred

# ============================================================
# ÉTAPE 6 : Matrice de confusion
# ============================================================
def plot_confusion_matrix(y_test, y_pred, model_name):
    os.makedirs(RESULTS_PATH, exist_ok=True)

    cm  = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=CLASS_NAMES)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Matrice de confusion - {model_name}", fontsize=13)
    plt.tight_layout()

    filename = f"confusion_matrix_{model_name.replace(' ', '_')}.png"
    plt.savefig(os.path.join(RESULTS_PATH, filename), dpi=150)
    plt.show()
    print(f"Matrice sauvegardee : results/{filename}")

# ============================================================
# ÉTAPE 7 : Comparer les modèles
# ============================================================
def plot_comparison(results):
    os.makedirs(RESULTS_PATH, exist_ok=True)

    models   = list(results.keys())
    accuracy = [results[m] * 100 for m in models]
    colors   = ["steelblue", "coral"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(models, accuracy, color=colors, width=0.4, edgecolor="white")

    for bar, acc in zip(bars, accuracy):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.3,
                f"{acc:.2f}%",
                ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_ylim(80, 101)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Comparaison des modeles", fontsize=13)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, "comparaison_modeles.png"), dpi=150)
    plt.show()
    print("Graphique sauvegarde : results/comparaison_modeles.png")

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    # 1. Charger
    X, y = load_data()

    # 2. Préparer
    X_train, X_test, y_train, y_test, scaler = prepare_data(X, y)

    # 3. Random Forest
    rf       = train_random_forest(X_train, y_train)
    acc_rf, y_pred_rf = evaluate_model(rf, X_test, y_test, "Random Forest")
    plot_confusion_matrix(y_test, y_pred_rf, "Random Forest")

    # 4. SVM
    svm      = train_svm(X_train, y_train)
    acc_svm, y_pred_svm = evaluate_model(svm, X_test, y_test, "SVM")
    plot_confusion_matrix(y_test, y_pred_svm, "SVM")

    # 5. Comparaison
    results = {"Random Forest": acc_rf, "SVM": acc_svm}
    plot_comparison(results)

    # 6. Sauvegarder le meilleur modèle
    best_name  = max(results, key=results.get)
    best_model = rf if best_name == "Random Forest" else svm
    os.makedirs(MODELS_PATH, exist_ok=True)
    joblib.dump(best_model, os.path.join(MODELS_PATH, "best_model.pkl"))

    print("\n" + "=" * 55)
    print(f"  Meilleur modele : {best_name}")
    print(f"  Accuracy        : {results[best_name]*100:.2f}%")
    print(f"  Sauvegarde      : models/best_model.pkl")
    print("=" * 55)
    print("\nEtape 4 terminee ! Pret pour l etape 5 : prediction.")