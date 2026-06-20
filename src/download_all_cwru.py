import urllib.request
import os
import time
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

BASE_URL = "https://engineering.case.edu/sites/default/files"

# ============================================================
# TOUS LES FICHIERS CWRU
# ============================================================
ALL_FILES = {

    # --------------------------------------------------------
    # NORMAL (baseline)
    # --------------------------------------------------------
    "Normal": {
        "Normal_0.mat": f"{BASE_URL}/97.mat",
        "Normal_1.mat": f"{BASE_URL}/98.mat",
        "Normal_2.mat": f"{BASE_URL}/99.mat",
        "Normal_3.mat": f"{BASE_URL}/100.mat",
    },

    # --------------------------------------------------------
    # INNER RACE — 0.007"
    # --------------------------------------------------------
    "Inner_Race": {
        "IR007_0.mat": f"{BASE_URL}/105.mat",
        "IR007_1.mat": f"{BASE_URL}/106.mat",
        "IR007_2.mat": f"{BASE_URL}/107.mat",
        "IR007_3.mat": f"{BASE_URL}/108.mat",
        # 0.014"
        "IR014_0.mat": f"{BASE_URL}/169.mat",
        "IR014_1.mat": f"{BASE_URL}/170.mat",
        "IR014_2.mat": f"{BASE_URL}/171.mat",
        "IR014_3.mat": f"{BASE_URL}/172.mat",
        # 0.021"
        "IR021_0.mat": f"{BASE_URL}/209.mat",
        "IR021_1.mat": f"{BASE_URL}/210.mat",
        "IR021_2.mat": f"{BASE_URL}/211.mat",
        "IR021_3.mat": f"{BASE_URL}/212.mat",
        # 0.028"
        "IR028_0.mat": f"{BASE_URL}/3001.mat",
        "IR028_1.mat": f"{BASE_URL}/3002.mat",
        "IR028_2.mat": f"{BASE_URL}/3003.mat",
        "IR028_3.mat": f"{BASE_URL}/3004.mat",
    },

    # --------------------------------------------------------
    # BALL — 0.007"
    # --------------------------------------------------------
    "Ball": {
        "B007_0.mat": f"{BASE_URL}/118.mat",
        "B007_1.mat": f"{BASE_URL}/119.mat",
        "B007_2.mat": f"{BASE_URL}/120.mat",
        "B007_3.mat": f"{BASE_URL}/121.mat",
        # 0.014"
        "B014_0.mat": f"{BASE_URL}/185.mat",
        "B014_1.mat": f"{BASE_URL}/186.mat",
        "B014_2.mat": f"{BASE_URL}/187.mat",
        "B014_3.mat": f"{BASE_URL}/188.mat",
        # 0.021"
        "B021_0.mat": f"{BASE_URL}/222.mat",
        "B021_1.mat": f"{BASE_URL}/223.mat",
        "B021_2.mat": f"{BASE_URL}/224.mat",
        "B021_3.mat": f"{BASE_URL}/225.mat",
        # 0.028"
        "B028_0.mat": f"{BASE_URL}/3005.mat",
        "B028_1.mat": f"{BASE_URL}/3006.mat",
        "B028_2.mat": f"{BASE_URL}/3007.mat",
        "B028_3.mat": f"{BASE_URL}/3008.mat",
    },

    # --------------------------------------------------------
    # OUTER RACE @6:00 — toutes tailles
    # --------------------------------------------------------
    "Outer_Race": {
        # 0.007"
        "OR007_6_0.mat": f"{BASE_URL}/130.mat",
        "OR007_6_1.mat": f"{BASE_URL}/131.mat",
        "OR007_6_2.mat": f"{BASE_URL}/132.mat",
        "OR007_6_3.mat": f"{BASE_URL}/133.mat",
        # 0.007" @3:00
        "OR007_3_0.mat": f"{BASE_URL}/144.mat",
        "OR007_3_1.mat": f"{BASE_URL}/145.mat",
        "OR007_3_2.mat": f"{BASE_URL}/146.mat",
        "OR007_3_3.mat": f"{BASE_URL}/147.mat",
        # 0.007" @12:00
        "OR007_12_0.mat": f"{BASE_URL}/156.mat",
        "OR007_12_1.mat": f"{BASE_URL}/157.mat",
        "OR007_12_2.mat": f"{BASE_URL}/158.mat",
        "OR007_12_3.mat": f"{BASE_URL}/159.mat",
        # 0.014" @6:00
        "OR014_6_0.mat": f"{BASE_URL}/197.mat",
        "OR014_6_1.mat": f"{BASE_URL}/198.mat",
        "OR014_6_2.mat": f"{BASE_URL}/199.mat",
        "OR014_6_3.mat": f"{BASE_URL}/200.mat",
        # 0.021" @6:00
        "OR021_6_0.mat": f"{BASE_URL}/234.mat",
        "OR021_6_1.mat": f"{BASE_URL}/235.mat",
        "OR021_6_2.mat": f"{BASE_URL}/236.mat",
        "OR021_6_3.mat": f"{BASE_URL}/237.mat",
        # 0.021" @3:00
        "OR021_3_0.mat": f"{BASE_URL}/246.mat",
        "OR021_3_1.mat": f"{BASE_URL}/247.mat",
        "OR021_3_2.mat": f"{BASE_URL}/248.mat",
        "OR021_3_3.mat": f"{BASE_URL}/249.mat",
        # 0.021" @12:00
        "OR021_12_0.mat": f"{BASE_URL}/258.mat",
        "OR021_12_1.mat": f"{BASE_URL}/259.mat",
        "OR021_12_2.mat": f"{BASE_URL}/260.mat",
        "OR021_12_3.mat": f"{BASE_URL}/261.mat",
    },
}

# ============================================================
# TÉLÉCHARGEMENT AVEC REPRISE
# ============================================================
def download_file(url, dest, nb_tentatives=5):
    for tentative in range(1, nb_tentatives + 1):
        try:
            if os.path.exists(dest):
                os.remove(dest)
            urllib.request.urlretrieve(url, dest)
            size_kb = os.path.getsize(dest) // 1024
            if size_kb < 50:
                raise ValueError(f"Fichier trop petit ({size_kb} KB) — probablement invalide")
            return True, size_kb
        except Exception as e:
            if tentative < nb_tentatives:
                time.sleep(tentative * 2)
            else:
                return False, str(e)

def download_all():
    total   = sum(len(v) for v in ALL_FILES.values())
    succes  = 0
    echecs  = []
    count   = 0

    print("=" * 60)
    print("   Téléchargement complet base CWRU")
    print(f"   Total fichiers : {total}")
    print("=" * 60)

    for folder_name, file_dict in ALL_FILES.items():
        folder_path = os.path.join("data", folder_name)
        os.makedirs(folder_path, exist_ok=True)
        print(f"\n[{folder_name}] — {len(file_dict)} fichiers")

        for filename, url in file_dict.items():
            dest = os.path.join(folder_path, filename)
            count += 1

            # Vérifier si déjà téléchargé et valide
            if os.path.exists(dest):
                size_kb = os.path.getsize(dest) // 1024
                if size_kb > 50:
                    print(f"  ({count:02d}/{total}) OK déjà présent : {filename} ({size_kb} KB)")
                    succes += 1
                    continue

            print(f"  ({count:02d}/{total}) Téléchargement : {filename} ...", end="", flush=True)
            ok, info = download_file(url, dest)

            if ok:
                print(f" OK ({info} KB)")
                succes += 1
            else:
                print(f" ECHEC : {info}")
                echecs.append(filename)

    # Résumé final
    print("\n" + "=" * 60)
    print(f"  Résultat : {succes}/{total} fichiers téléchargés")

    if echecs:
        print(f"\n  Fichiers échoués ({len(echecs)}) :")
        for f in echecs:
            print(f"    - {f}")
    else:
        print("  Tous les fichiers téléchargés avec succès !")

    print("\n  Structure finale :")
    for folder_name in ALL_FILES:
        folder_path = os.path.join("data", folder_name)
        files = os.listdir(folder_path)
        expected = len(ALL_FILES[folder_name])
        status = "OK" if len(files) >= expected else f"INCOMPLET {len(files)}/{expected}"
        print(f"  [{status}] data/{folder_name}/ -> {len(files)} fichiers")

    print("=" * 60)

if __name__ == "__main__":
    download_all()