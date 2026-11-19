"""
Télécharge les données brutes SoccerNet/SN-echoes et crée echoes_brut.csv.

À lancer depuis le dossier projet-foot :
    python telecharger_echoes_brut.py

Le CSV regroupe les trois versions Whisper (v1, v2, v3), avant sélection
de la meilleure version et avant nettoyage.
"""
from pathlib import Path

import pandas as pd
from datasets import load_dataset

VERSIONS = ["whisper_v1", "whisper_v2", "whisper_v3"]
SPLIT = "en"
SORTIE = Path(__file__).resolve().parent / "echoes_brut.csv"


def main():
    tableaux = []

    for version in VERSIONS:
        print(f"\n[TÉLÉCHARGEMENT] {version} / {SPLIT}")
        dataset = load_dataset("SoccerNet/SN-echoes", version)
        df = dataset[SPLIT].to_pandas()
        df["version"] = version
        print(f"{len(df):,} lignes récupérées")
        tableaux.append(df)

    brut = pd.concat(tableaux, ignore_index=True)
    brut.to_csv(SORTIE, index=False, encoding="utf-8-sig")

    print("\nTéléchargement terminé.")
    print(f"{len(brut):,} lignes enregistrées dans : {SORTIE}")
    print(f"Colonnes : {', '.join(brut.columns)}")


if __name__ == "__main__":
    main()
