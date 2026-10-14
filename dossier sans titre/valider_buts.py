"""
Valide la détection des buts et le recalage sur le coup d'envoi avec les annotations officielles SoccerNet.
Optionnel : ne sert qu'à mesurer la qualité de l'extraction, pas à prédire (ce serait tricher).

Prérequis : pip install SoccerNet
Au premier lancement, télécharge les fichiers Labels-v2.json (les annotations, pas les vidéos) dans ./SoccerNet.

Usage, dans l'environnement nlp-foot et le dossier du projet :
    python valider_buts.py
"""
import glob
import json
import os

import numpy as np
import pandas as pd

import modele_foot as mf

DOSSIER = "SoccerNet"
TOLERANCE = 2.0  # minutes d'écart tolérées entre l'annotation et la détection


def telecharger():
    from SoccerNet.Downloader import SoccerNetDownloader
    d = SoccerNetDownloader(LocalDirectory=DOSSIER)
    d.downloadGames(files=["Labels-v2.json"], split=["train", "valid", "test"])


def fichiers_annotations():
    return glob.glob(os.path.join(DOSSIER, "*", "*", "*", "Labels-v2.json"))


def lire_annotations():
    lignes = []
    for chemin in fichiers_annotations():
        match = os.path.basename(os.path.dirname(chemin))
        with open(chemin, encoding="utf-8") as f:
            annotations = json.load(f).get("annotations", [])
        for a in annotations:
            if a.get("label") not in ("Goal", "Kick-off"):
                continue
            mi_temps, temps = a["gameTime"].split(" - ")
            mm, ss = temps.split(":")
            lignes.append({"match": match, "label": a["label"], "mi_temps": int(mi_temps),
                           "minute_video": int(mm) + int(ss) / 60, "equipe": a.get("team")})
    return pd.DataFrame(lignes)


def apparier(vrais, detectes):
    """Nombre d'annotations retrouvées (au plus une détection par annotation, à TOLERANCE près)."""
    utilises, ok = set(), 0
    for v in vrais:
        cands = [k for k, d in enumerate(detectes) if k not in utilises and abs(d - v) <= TOLERANCE]
        if cands:
            utilises.add(min(cands, key=lambda k: abs(detectes[k] - v)))
            ok += 1
    return ok


def main():
    if not fichiers_annotations():
        print("Téléchargement des annotations SoccerNet (Labels-v2.json)...")
        telecharger()
    ann = lire_annotations()
    df, _ = mf.preparer_tout()
    communs = set(ann["match"]) & set(df["match"])
    print(f"\n{ann['match'].nunique()} matchs annotés, {len(communs)} en commun avec SN-echoes")
    if not communs:
        return

    # 1. Recalage sur le coup d'envoi
    ko = ann[ann["label"] == "Kick-off"].groupby(["match", "mi_temps"])["minute_video"].min().rename("vrai")
    dec = df.groupby(["match", "mi_temps"])["decalage"].first().rename("estime")
    j = pd.concat([ko, dec], axis=1, join="inner")
    print("\n=== Coup d'envoi ===")
    print(f"Mi-temps comparées : {len(j)}, coup d'envoi détecté dans le texte pour {(j['estime'] > 0).mean():.0%}")
    print(f"Erreur moyenne sur la minute : {(j['vrai'] - j['estime']).abs().mean():.2f} min avec recalage, "
          f"{j['vrai'].abs().mean():.2f} min sans recalage")

    # 2. Détection des buts
    vrais = ann[(ann["label"] == "Goal") & ann["match"].isin(communs)]
    det = df[df["match"].isin(communs)]
    trouves = trouves_eq = 0
    for (match, mt), v in vrais.groupby(["match", "mi_temps"]):
        d = det[(det["match"] == match) & (det["mi_temps"] == mt)]
        trouves += apparier(v["minute_video"].tolist(), d.loc[d["evt_but"], "minute_video"].tolist())
        for equipe, col in (("home", "evt_but_dom"), ("away", "evt_but_ext")):
            trouves_eq += apparier(v.loc[v["equipe"] == equipe, "minute_video"].tolist(),
                                   d.loc[d[col], "minute_video"].tolist())
    n_det = int(det["evt_but"].sum())
    n_det_eq = int(det["evt_but_dom"].sum() + det["evt_but_ext"].sum())
    print("\n=== Buts ===")
    print(f"Buts annotés : {len(vrais)}, détections : {n_det} (dont {n_det_eq} attribuées à une équipe)")
    print(f"Rappel (buts retrouvés) : {trouves / max(len(vrais), 1):.0%} · "
          f"précision (détections justes) : {trouves / max(n_det, 1):.0%}")
    print(f"Avec la bonne équipe : rappel {trouves_eq / max(len(vrais), 1):.0%} · "
          f"précision {trouves_eq / max(n_det_eq, 1):.0%}")

    # 3. Score actuel reconstruit en fin de match
    fin = det.groupby("match").tail(1)
    exact = ((fin["etat_dom"] == fin["buts_dom"]) & (fin["etat_ext"] == fin["buts_ext"])).mean()
    issue = (np.sign(fin["etat_dom"] - fin["etat_ext"]) == fin["resultat"]).mean()
    print("\n=== Score reconstruit en fin de match ===")
    print(f"Score exact : {exact:.0%} · bonne issue : {issue:.0%}")


if __name__ == "__main__":
    main()
