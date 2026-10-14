"""
Prépare les données de la page web (index.html) avec le modèle v3 de modele_foot.py.

Pour chaque match :
- sentiment moyen par tranche de 5 minutes (global, équipe domicile, équipe extérieur) ;
- toutes les 10 minutes et en fin de match : score actuel estimé, probabilités de l'issue finale,
  pronostic de score final, dynamique (0 → X contre la dernière tranche de 10 minutes) et indices ;
- l'Elo des deux équipes avant le match.
Les prédictions sont hors-échantillon (validation croisée) : chaque match est prédit par un
modèle entraîné sans lui.

Usage, dans l'environnement nlp-foot et le dossier du projet :
    python exporter_donnees.py
Produit donnees.js, à placer à côté de index.html.
"""
import json

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error
from sklearn.model_selection import StratifiedKFold, cross_val_predict

import modele_foot as mf

CHECKPOINTS = mf.X_MINUTES   # 10, 20, ..., 90, fin du match
BIN_MIN = 5
SORTIE = "donnees.js"


def r3(v):
    return None if v is None or pd.isna(v) else round(float(v), 3)


def part(a, b):
    return None if a + b == 0 else round(float(a / (a + b)), 3)


def sentiment_par_tranche(df):
    bins = list(range(0, mf.FIN_MATCH, BIN_MIN))
    d = df.assign(b=(df["minute"] // BIN_MIN * BIN_MIN).astype(int))
    res = {}
    for match, g in d.groupby("match", sort=False):
        def stats(sous):
            st = sous.groupby("b")["sent"].agg(["mean", "size"]).reindex(bins)
            return [r3(v) for v in st["mean"]], st["size"].fillna(0).astype(int).tolist()

        moy, n = stats(g)
        moy_dom, n_dom = stats(g[g["dom"]])
        moy_ext, n_ext = stats(g[g["ext"]])
        res[match] = {"bins": bins, "global": moy, "n": n,
                      "dom": moy_dom, "n_dom": n_dom, "ext": moy_ext, "n_ext": n_ext}
    return res


def main():
    df, elo = mf.preparer_tout()
    sent = sentiment_par_tranche(df)
    pile = mf.empiler(df, elo, CHECKPOINTS)

    print(f"Validation croisée du modèle final (mode {mf.MODELE})...")
    cls, proba, lam_d, lam_e, choix = mf.predictions_finales(pile)
    cls = list(cls)

    def colonne(v):
        return proba[:, cls.index(v)] if v in cls else np.zeros(len(pile))

    p_dom, p_nul, p_ext = colonne(1), colonne(0), colonne(-1)
    issue = np.array(cls)[proba.argmax(axis=1)]

    preds, scores = {}, []
    for i, r in enumerate(pile.to_dict("records")):
        ed, ee = int(r["etat_dom"]), int(r["etat_ext"])
        sc = mf.score_estime(float(lam_d[i]), float(lam_e[i]), int(issue[i]), ed, ee)
        scores.append(sc)
        preds.setdefault(r["match"], {})[str(r["fenetre"])] = {
            "p": [round(float(p_dom[i]), 3), round(float(p_nul[i]), 3), round(float(p_ext[i]), 3)],
            "score": sc,
            "lambda": [round(float(lam_d[i]), 2), round(float(lam_e[i]), 2)],
            "etat": [ed, ee],
            "etat_conf_age": round(float(r["etat_conf_age"]), 1),
            "etat_non_attr": int(r["etat_non_attr"]),
            "indices": {
                "buts": [int(r["buts_evt_dom"]), int(r["buts_evt_ext"])],
                "rouges": [int(r["rouge_dom"]), int(r["rouge_ext"])],
                "score_cite": r["score_cite"],
                "score_cite_min": r3(r["score_cite_min"]),
            },
            "dyn": {
                "sent": [r3(r["sent_dom"]), r3(r["sent_ext"])],
                "sent_rec": [r3(r["sent_recent_dom"]), r3(r["sent_recent_ext"])],
                "ment": part(r["ment_dom"], r["ment_ext"]),
                "ment_rec": part(r["ment_recent_dom"], r["ment_recent_ext"]),
            },
        }
    scores = np.array(scores)

    # Fiabilité par fenêtre
    metriques = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=mf.SEED)
    for x in CHECKPOINTS:
        sel = (pile["fenetre"] == x).to_numpy()
        m = pile[sel]
        if len(m) < 30:
            continue
        y = m["resultat"]
        cls_m = np.sort(y.unique())
        base = cross_val_predict(DummyClassifier(strategy="prior"), m, y, cv=cv, method="predict_proba")
        p_elo = cross_val_predict(mf.modele_elo(), m, y, cv=cv, method="predict_proba")
        exact = (scores[sel, 0] == m["buts_dom"].to_numpy()) & (scores[sel, 1] == m["buts_ext"].to_numpy())
        metriques[str(x)] = {
            "matchs": int(sel.sum()),
            "acc_modele": round(accuracy_score(y, issue[sel]), 3),
            "acc_base": round(accuracy_score(y, cls_m[base.argmax(axis=1)]), 3),
            "acc_elo": round(accuracy_score(y, cls_m[p_elo.argmax(axis=1)]), 3),
            "logloss_modele": round(log_loss(y, proba[sel], labels=cls), 3),
            "logloss_base": round(log_loss(y, base, labels=cls_m), 3),
            "mae_modele": round((mean_absolute_error(m["buts_dom"], lam_d[sel])
                                 + mean_absolute_error(m["buts_ext"], lam_e[sel])) / 2, 3),
            "score_exact": round(float(exact.mean()), 3),
            "variante": "modèle unique" if choix.get(x) == "unique" else "un modèle par fenêtre",
        }
        print(f"{mf.nom_fenetre(x)} : {metriques[str(x)]['acc_modele']:.0%} de bonnes issues "
              f"(référence {metriques[str(x)]['acc_base']:.0%}, Elo seul {metriques[str(x)]['acc_elo']:.0%})")

    infos = df.groupby("match", sort=False).agg(
        championnat=("championnat", "first"), saison=("saison", "first"), date=("date", "first"),
        dom=("equipe_dom", "first"), ext=("equipe_ext", "first"),
        buts_dom=("buts_dom", "first"), buts_ext=("buts_ext", "first"))

    matchs = []
    for match, r in infos.iterrows():
        e = elo.loc[match]
        matchs.append({
            "id": match,
            "competition": f"{r['championnat']} · {r['saison']}",
            "date": r["date"], "dom": r["dom"], "ext": r["ext"],
            "buts_dom": int(r["buts_dom"]), "buts_ext": int(r["buts_ext"]),
            "elo": {"dom": round(float(e["elo_dom"])), "ext": round(float(e["elo_ext"])),
                    "attendu_dom": round(float(e["elo_proba_dom"]), 3), "nb_min": int(e["elo_nb_min"])},
            "sentiment": sent[match],
            "predictions": preds.get(match, {}),
        })
    matchs.sort(key=lambda x: (x["competition"], x["date"]))

    donnees = {
        "meta": {"checkpoints": CHECKPOINTS, "fin_match": mf.FIN_MATCH, "bin_min": BIN_MIN,
                 "tranche_recente": mf.TRANCHE_RECENTE, "sentiment": mf.SENTIMENT,
                 "modele": mf.MODELE, "metriques": metriques},
        "matchs": matchs,
    }
    with open(SORTIE, "w", encoding="utf-8") as f:
        f.write("window.DONNEES = " + json.dumps(donnees, ensure_ascii=False) + ";")
    print(f"{len(matchs)} matchs exportés dans {SORTIE}")


if __name__ == "__main__":
    main()
