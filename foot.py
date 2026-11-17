"""Petit script de contrôle pour vérifier l'accès au dataset SoccerNet SN-echoes.

Usage : python foot.py
Ce fichier est indépendant du pipeline de prédiction modele_foot.py.
"""
from datasets import load_dataset


def main():
    ds = load_dataset("SoccerNet/SN-echoes", "whisper_v2")
    print(ds)


if __name__ == "__main__":
    main()
