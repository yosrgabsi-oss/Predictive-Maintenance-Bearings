import urllib.request
import os

# Fichiers CWRU à télécharger (numéro CWRU -> dossier destination)
FILES = {
    # Normal (4 charges : 0, 1, 2, 3 HP)
    "Normal": {
        "97.mat":  "https://engineering.case.edu/sites/default/files/97.mat",
        "98.mat":  "https://engineering.case.edu/sites/default/files/98.mat",
        "99.mat":  "https://engineering.case.edu/sites/default/files/99.mat",
        "100.mat": "https://engineering.case.edu/sites/default/files/100.mat",
    },
    # Inner Race - 0.007" - 12k
    "Inner_Race": {
        "IR007_0.mat": "https://engineering.case.edu/sites/default/files/105.mat",
        "IR007_1.mat": "https://engineering.case.edu/sites/default/files/106.mat",
        "IR007_2.mat": "https://engineering.case.edu/sites/default/files/107.mat",
        "IR007_3.mat": "https://engineering.case.edu/sites/default/files/108.mat",
    },
    # Ball - 0.007" - 12k
    "Ball": {
        "B007_0.mat": "https://engineering.case.edu/sites/default/files/118.mat",
        "B007_1.mat": "https://engineering.case.edu/sites/default/files/119.mat",
        "B007_2.mat": "https://engineering.case.edu/sites/default/files/120.mat",
        "B007_3.mat": "https://engineering.case.edu/sites/default/files/121.mat",
    },
    # Outer Race @6:00 - 0.007" - 12k
    "Outer_Race": {
        "OR007_0.mat": "https://engineering.case.edu/sites/default/files/130.mat",
        "OR007_1.mat": "https://engineering.case.edu/sites/default/files/131.mat",
        "OR007_2.mat": "https://engineering.case.edu/sites/default/files/132.mat",
        "OR007_3.mat": "https://engineering.case.edu/sites/default/files/133.mat",
    },
}

def download_all():
    total = sum(len(v) for v in FILES.values())
    count = 0

    for folder_name, file_dict in FILES.items():
        folder_path = os.path.join("data", folder_name)
        os.makedirs(folder_path, exist_ok=True)

        for filename, url in file_dict.items():
            dest = os.path.join(folder_path, filename)

            if os.path.exists(dest):
                print(f"  [OK déjà présent] {filename}")
                count += 1
                continue

            try:
                print(f"  Téléchargement ({count+1}/{total}) : {filename} ...", end="")
                urllib.request.urlretrieve(url, dest)
                size_kb = os.path.getsize(dest) // 1024
                print(f" {size_kb} KB")
                count += 1
            except Exception as e:
                print(f" ERREUR : {e}")

    print(f"\nTerminé ! {count}/{total} fichiers téléchargés.")
    print("\nStructure finale :")
    for folder_name in FILES:
        folder_path = os.path.join("data", folder_name)
        files = os.listdir(folder_path)
        print(f"  data/{folder_name}/ -> {len(files)} fichiers : {files}")

if __name__ == "__main__":
    download_all()
