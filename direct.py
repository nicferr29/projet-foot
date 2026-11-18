"""
Prédire le match « en direct », minute après minute, à partir des seuls commentaires déjà prononcés.

À chaque point de contrôle (0', 10', 20'… mi-temps… 90', fin) :
1. on ne garde que les commentaires prononcés avant cette minute ;
2. le détecteur de buts (detection_buts.py) estime le score courant : loi de probabilité sur 0-0, 1-0, 0-1... ;
   on garde aussi l'explication de ce score (quels passages ont été comptés comme buts, quel score cité les
   confirme) pour l'afficher ;
3. on projette la fin du match : les buts restants suivent une loi de Poisson dont le taux vient de la force des
   équipes avant le match (filtre de Kalman sur les seuls matchs antérieurs) et de la part du temps qui reste
   (mesurée sur les annotations : on marque plus en fin de mi-temps) ;
4. on en déduit les chances de victoire / nul / défaite et le score final le plus probable.

Au coup d'envoi, la prédiction est donc celle de l'avant-match ; elle s'affine à mesure que le match avance.
Tout est évalué en validation croisée par match. Le script écrit match_en_direct.html. Les actions annotées
(buts, cartons, tirs...) servent seulement à animer la page et à montrer le vrai déroulement, jamais à prédire.

Usage (environnement conda nlp-foot, dans le dossier du projet) :
    python direct.py
"""
import json
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import GroupKFold

import detection_buts as db
import donnees
import equipes

PAGE = os.path.join(donnees.ICI, "match_en_direct.html")
GABARIT = os.path.join(donnees.ICI, "gabarit_page.html")
FICHIER_JSON = os.path.join(donnees.ICI, "resultats", "predictions_direct.json")
COMPETITIONS = {"england_epl": "Premier League", "spain_laliga": "LaLiga", "italy_serie-a": "Serie A",
                "germany_bundesliga": "Bundesliga", "france_ligue-1": "Ligue 1",
                "europe_uefa-champions-league": "Ligue des champions"}
POINTS = ["0'", "10'", "20'", "30'", "40'", "Mi-temps", "50'", "60'", "70'", "80'", "90'", "Fin"]
MARGE = 0.25                 # minute : on ignore les 15 dernières secondes (le contexte d'une mention déborde un peu)
ELO_INIT, ELO_K, ELO_DOMICILE = 1500.0, 20.0, 60.0    # Elo classique, gardé comme point de comparaison
# Force « à la Kalman », en buts par match par rapport à une équipe moyenne. Réglage fixé d'avance, pas cherché
# sur les matchs d'évaluation (27 réglages voisins donnent tous de 67 à 69 % de bons vainqueurs au coup d'envoi).
FORCE_VARIANCE_INIT = 0.5    # incertitude sur une équipe jamais vue (écart-type ≈ 0,7 but)
FORCE_DERIVE = 0.1           # variance ajoutée par an : une équipe change d'une saison à l'autre
FORCE_BRUIT = 3.0            # variance de l'écart de buts d'un match autour de l'écart attendu (écart-type ≈ 1,7)
ECART_MAX = 3                # un 7-0 compte comme un 3-0 : un écart énorme n'en dit guère plus
TERRAIN_INIT, TERRAIN_POIDS = 0.3, 10   # avantage du terrain a priori (buts), et son poids en nombre de matchs
N_SORTIE = db.MAX_BUTS + 6


def limites(point):
    """Minutes (dans chaque mi-temps) avant lesquelles les commentaires sont connus à ce point de contrôle."""
    if point == "Fin":
        return np.inf, np.inf
    if point == "Mi-temps":
        return np.inf, -np.inf
    x = int(point.rstrip("'"))
    return (x, -np.inf) if x <= 45 else (np.inf, x - 45)


def avant(df, point, marge=0.0):
    l1, l2 = limites(point)
    return df[((df["mi_temps"] == 1) & (df["minute"] < l1 - marge)) | ((df["mi_temps"] == 2) & (df["minute"] < l2 - marge))]


def minute_de_match(mi_temps, minute):
    """23.4 en 2e mi-temps -> « 69' » ; temps additionnel : « 45+2' », « 90+3' »."""
    m = int(minute) + 1
    if mi_temps == 1:
        return f"{m}'" if m <= 45 else f"45+{m - 45}'"
    return f"{45 + m}'" if m <= 45 else f"90+{m - 45}'"


# ----------------------------------------------------------------------------
# Avant-match : force des équipes et buts attendus
# ----------------------------------------------------------------------------
def calculer_elo(infos):
    """Elo de chaque équipe avant chaque match, calculé avec les seuls matchs antérieurs (tous les matchs
    SoccerNet connus, y compris ceux sans commentaires)."""
    elo, nb, lignes = defaultdict(lambda: ELO_INIT), defaultdict(int), {}
    for match, r in infos.sort_values("date").iterrows():
        ed, ee = elo[r["dom"]], elo[r["ext"]]
        lignes[match] = (ed, ee, min(nb[r["dom"]], nb[r["ext"]]))
        attendu = 1 / (1 + 10 ** ((ee - ed - ELO_DOMICILE) / 400))
        ecart = abs(r["buts_dom"] - r["buts_ext"])
        mult = 1.0 if ecart <= 1 else (1.5 if ecart == 2 else (11 + ecart) / 8)
        delta = ELO_K * mult * ({1: 1.0, 0: 0.5, -1: 0.0}[r["resultat"]] - attendu)
        elo[r["dom"]] += delta
        elo[r["ext"]] -= delta
        nb[r["dom"]] += 1
        nb[r["ext"]] += 1
    e = pd.DataFrame.from_dict(lignes, orient="index", columns=["elo_dom", "elo_ext", "elo_nb_matchs"])
    return e.assign(ecart=(e["elo_dom"] - e["elo_ext"]) / 400)


def calculer_force(infos):
    """Force de chaque équipe avant chaque match, par un filtre de Kalman (seuls les matchs antérieurs).

    Chaque équipe a une force estimée (buts par match de plus qu'une équipe moyenne) et une incertitude.
    - Prédiction : entre deux matchs, l'incertitude grandit (FORCE_DERIVE par an).
    - Mise à jour : l'écart de buts observé (borné à ± ECART_MAX) est comparé à l'écart attendu, force du
      domicile − force de l'extérieur + avantage du terrain. Chaque équipe corrige sa force d'une part de la
      surprise : le gain de Kalman, d'autant plus grand qu'on la connaît mal, et qui diminue match après match.
    L'Elo fait la même correction avec un gain fixe (K) pour toutes les équipes. L'avantage du terrain est la
    moyenne des écarts de buts déjà vus."""
    force, variance, vu = {}, {}, {}
    somme, n = TERRAIN_INIT * TERRAIN_POIDS, TERRAIN_POIDS
    lignes = {}
    ordre = infos.assign(_cle=infos.index.str.split("/").str[2]).sort_values(["date", "_cle"])
    for match, r in ordre.iterrows():
        jour = pd.Timestamp(r["date"])
        for e in (r["dom"], r["ext"]):
            if e not in force:
                force[e], variance[e], vu[e] = 0.0, FORCE_VARIANCE_INIT, jour
            variance[e] += FORCE_DERIVE * (jour - vu[e]).days / 365.25
            vu[e] = jour
        a, b, terrain = r["dom"], r["ext"], somme / n
        lignes[match] = (force[a], force[b], np.sqrt(variance[a]), np.sqrt(variance[b]), terrain)
        ecart = r["buts_dom"] - r["buts_ext"]
        surprise = np.clip(ecart, -ECART_MAX, ECART_MAX) - (force[a] - force[b] + terrain)
        total = variance[a] + variance[b] + FORCE_BRUIT
        force[a] += variance[a] / total * surprise
        force[b] -= variance[b] / total * surprise
        variance[a] -= variance[a] ** 2 / total
        variance[b] -= variance[b] ** 2 / total
        somme, n = somme + ecart, n + 1
    f = pd.DataFrame.from_dict(lignes, orient="index",
                               columns=["force_dom", "force_ext", "incert_dom", "incert_ext", "terrain"])
    return f.assign(ecart=f["force_dom"] - f["force_ext"] + f["terrain"])


class AvantMatch:
    """Buts attendus de chaque équipe sur tout le match : régression de Poisson sur l'écart de force (colonne
    « ecart » de calculer_force, ou de calculer_elo pour la comparaison)."""

    @staticmethod
    def _x(force):
        return force["ecart"].to_numpy().reshape(-1, 1)

    def fit(self, infos, force):
        X = self._x(force.loc[infos.index])
        self.dom = PoissonRegressor(alpha=1e-4).fit(X, infos["buts_dom"])
        self.ext = PoissonRegressor(alpha=1e-4).fit(X, infos["buts_ext"])
        return self

    def predict(self, force):
        X = self._x(force)
        return self.dom.predict(X), self.ext.predict(X)


class TempsRestant:
    """Part des buts d'un match qui restent à marquer à chaque point de contrôle (mesurée sur les annotations)."""

    def fit(self, buts, n_matchs):
        h = np.zeros((2, 60))
        for mt, minute in zip(buts["mi_temps"], buts["minute"]):
            h[mt - 1, min(int(minute), 59)] += 1
        self.h, self.total = h, h.sum()
        return self

    def __call__(self, point):
        l1, l2 = limites(point)
        apres1 = self.h[0, int(min(l1, 60)):].sum() if l1 != np.inf else 0.0
        apres2 = self.h[1].sum() if l2 == -np.inf else (self.h[1, int(min(l2, 60)):].sum() if l2 != np.inf else 0.0)
        return (apres1 + apres2) / self.total


# ----------------------------------------------------------------------------
# Projection du score final
# ----------------------------------------------------------------------------
def matrice_poisson(lam):
    """T[i, k] = P(i - k buts restants) : passe du score courant au score final."""
    pmf = poisson.pmf(np.arange(N_SORTIE), max(lam, 1e-9))
    T = np.zeros((N_SORTIE, db.MAX_BUTS + 1))
    for k in range(db.MAX_BUTS + 1):
        T[k:, k] = pmf[:N_SORTIE - k]
    return T


def projeter(P, lam_dom, lam_ext):
    F = matrice_poisson(lam_dom) @ P @ matrice_poisson(lam_ext).T
    return F / F.sum()


def resumer(F, plancher):
    """P(domicile, nul, extérieur) et pronostic : le score le plus probable parmi ceux qui donnent le résultat
    favori et ne sont pas inférieurs au score courant estimé (plancher)."""
    i, j = np.indices(F.shape)
    p = np.array([F[i > j].sum(), F[i == j].sum(), F[i < j].sum()])
    issue = [1, 0, -1][int(p.argmax())]
    G = np.where((np.sign(i - j) == issue) & (i >= plancher[0]) & (j >= plancher[1]), F + 1e-300, 0)
    return p, np.unravel_index(G.argmax(), G.shape)


def score_reel(buts, match, point):
    b = avant(buts[buts["match"] == match], point)
    return [int((b["equipe"] == "dom").sum()), int((b["equipe"] == "ext").sum())]


def temps(mi_temps, minute):
    """(mi-temps, minute dans la mi-temps) arrondis, pour placer un événement sur l'horloge de la page."""
    return [int(mi_temps), round(float(minute), 2)]


def extrait(texte, n=220):
    texte = re.sub(r"\s+", " ", str(texte)).strip()
    return texte if len(texte) <= n else texte[:n].rsplit(" ", 1)[0] + " …"


def predire_match(det, m, p, cites, lam, restant):
    """Les 12 points de contrôle d'un match (mentions m, probabilités d'annonce p, scores cités notés)."""
    sorties = []
    m = m.assign(p=p)
    for point in POINTS:
        mx, cx = avant(m, point, MARGE), avant(cites, point, MARGE)
        c = db.candidats(mx, mx["p"].to_numpy())
        c["q"] = det.verif.predict_proba(c[db.VARIABLES_CANDIDAT].astype(float))[:, 1] if len(c) else []
        P = db.decoder_score(c, cx)
        r = restant(point)
        F = projeter(P, lam[0] * r, lam[1] * r)
        cour = np.unravel_index(P.argmax(), P.shape)
        proba, prono = resumer(F, cour)
        # Pourquoi ce score : les buts retenus, leur annonce et le score cité qui les confirme
        lus = []
        for b in db.expliquer_score(c, cx, cour):
            g, s = b["candidat"], b["cite"]
            lus.append({"minute": minute_de_match(g.mi_temps, g.minute), "t": temps(g.mi_temps, g.minute),
                        "equipe": b["equipe"], "proba": round(b["proba"], 2), "q": round(float(g.q), 2),
                        "texte": extrait(g.texte),
                        "cite": None if s is None else {"minute": minute_de_match(s.mi_temps, s.minute),
                                                        "score": f"{s.a}-{s.b}", "texte": extrait(s.texte, 160)}})
        sorties.append({
            "point": point, "p": [round(float(v), 3) for v in proba],
            "cour": [int(cour[0]), int(cour[1])], "conf": round(float(P.max()), 3),
            "prono": [int(prono[0]), int(prono[1])], "part_restante": round(r, 3), "buts": lus,
        })
    return sorties


def comparer_avant_match(tous, infos, annotes, inedits, variantes):
    """Bon vainqueur et log loss au coup d'envoi selon la mesure de la force des équipes (même validation
    croisée que le reste) : c'est là seulement que la force compte seule."""
    plis = [list(annotes[te]) for _, te in GroupKFold(db.N_PLIS).split(annotes, groups=annotes)]
    depart = np.zeros((db.MAX_BUTS + 1, db.MAX_BUTS + 1))
    depart[0, 0] = 1.0
    lignes = []
    for nom, f in variantes.items():
        ligne = {"force": nom}
        for groupe, blocs in (("annotés", plis), ("inédits", [list(inedits)])):
            ok, ll = [], []
            for test in blocs:
                lam_d, lam_e = AvantMatch().fit(tous.drop(index=test), f).predict(f.loc[test])
                for match, a, b in zip(test, lam_d, lam_e):
                    p, _ = resumer(projeter(depart, a, b), (0, 0))
                    vrai = infos.loc[match, "resultat"]
                    ok.append([1, 0, -1][int(p.argmax())] == vrai)
                    ll.append(-np.log(max(p[{1: 0, 0: 1, -1: 2}[vrai]], 1e-6)))
            ligne[f"bon vainqueur ({groupe})"] = float(np.mean(ok))
            ligne[f"log loss ({groupe})"] = float(np.mean(ll))
        lignes.append(ligne)
    return pd.DataFrame(lignes).set_index("force")


# ----------------------------------------------------------------------------
# Validation croisée, évaluation et page
# ----------------------------------------------------------------------------
def evaluer(matchs_json, infos, annotes):
    """Par point de contrôle : bon résultat, log loss, pronostic exact, score courant exact (matchs annotés)."""
    lignes = []
    for k, point in enumerate(POINTS):
        y, P, prono_ok, cour_ok, ann = [], [], [], [], []
        for mj in matchs_json:
            pt = mj["points"][k]
            v = infos.loc[mj["id"]]
            y.append(v["resultat"])
            P.append(pt["p"])
            prono_ok.append(pt["prono"] == [int(v["buts_dom"]), int(v["buts_ext"])])
            if mj["reel"] is not None:
                cour_ok.append(pt["cour"] == mj["reel"][k])
        y, P = np.array(y), np.clip(np.array(P), 1e-6, 1)
        pred = np.array([1, 0, -1])[P.argmax(axis=1)]
        idx = np.select([y == 1, y == 0], [0, 1], 2)
        lignes.append({"point": point, "bon_resultat": float((pred == y).mean()),
                       "log_loss": float(-np.log(P[np.arange(len(y)), idx] / P.sum(axis=1)).mean()),
                       "score_final_exact": float(np.mean(prono_ok)),
                       "score_courant_exact": float(np.mean(cour_ok)) if cour_ok else None})
    return lignes


def main():
    pd.set_option("display.width", 200)
    os.makedirs(os.path.dirname(FICHIER_JSON), exist_ok=True)
    print("Chargement...")
    com = donnees.charger_commentaires()
    infos = donnees.infos_matchs(com["match"])
    buts, tous_annotes = donnees.charger_buts()
    evenements = donnees.charger_evenements()
    compos = donnees.charger_compositions(infos)
    annotes = np.array(sorted(set(infos.index) & tous_annotes))
    inedits = sorted(set(infos.index) - set(annotes))
    # Force des équipes : tous les matchs SoccerNet dont on connaît le résultat (nom du dossier)
    tous = donnees.infos_matchs(set(infos.index) | tous_annotes)
    force = calculer_force(tous)
    print("Extraction des mentions...")
    m = db.extraire_mentions(com, compos)
    E = db.representations(m)

    def bloc(det, matchs_test, matchs_train):
        avant_match = AvantMatch().fit(tous.drop(index=list(matchs_test)), force)
        m_ann = set(matchs_train) & set(annotes)
        restant = TempsRestant().fit(buts[buts["match"].isin(m_ann)], len(m_ann))
        sel = m["match"].isin(set(matchs_test)).to_numpy()
        p, _, cites = det.predict(m[sel])
        mt = m[sel].assign(p=p)
        lam_d, lam_e = avant_match.predict(force.loc[list(matchs_test)])
        lam = dict(zip(matchs_test, zip(lam_d, lam_e)))
        cm = {x: g for x, g in cites.groupby("match")}
        res = []
        for match, g in mt.groupby("match"):
            points = predire_match(det, g, g["p"].to_numpy(), cm.get(match, cites.iloc[:0]), lam[match], restant)
            res.append((match, points, lam[match]))
        return res

    resultats = []
    plis = list(GroupKFold(db.N_PLIS).split(annotes, groups=annotes))
    for k, (i_tr, i_te) in enumerate(plis):
        print(f"  pli {k + 1}/{db.N_PLIS}...")
        tr = set(annotes[i_tr])
        det = db.DetecteurDeButs(E).fit(m[m["match"].isin(tr)], buts[buts["match"].isin(tr)])
        resultats += bloc(det, list(annotes[i_te]), tr)
    print("  matchs jamais annotés...")
    det = db.DetecteurDeButs(E).fit(m[m["match"].isin(set(annotes))], buts[buts["match"].isin(set(annotes))])
    resultats += bloc(det, inedits, set(annotes))

    # Durée de chaque mi-temps (temps additionnel compris) : dernière action annotée ou dernier commentaire
    duree = pd.concat([com.groupby(["match", "mi_temps"])["minute"].max(),
                       evenements.groupby(["match", "mi_temps"])["minute"].max()], axis=1).max(axis=1)
    ev_match = {x: g for x, g in evenements.groupby("match")}
    matchs_json = []
    for match, points, lam in resultats:
        v = infos.loc[match]
        e = force.loc[match]
        matchs_json.append({
            "id": match, "competition": COMPETITIONS.get(v["competition"], v["competition"]), "saison": match.split("/")[1],
            "date": v["date"], "dom": equipes.identite(v["dom"]), "ext": equipes.identite(v["ext"]),
            "fin_mt": [round(float(np.clip(duree.get((match, mt), 45.0), 45.0, 55.0)), 2) for mt in (1, 2)],
            "final": [int(v["buts_dom"]), int(v["buts_ext"])],
            # force avant-match (buts par match de plus qu'une équipe moyenne) et son incertitude (écart-type)
            "force": [[round(float(e["force_dom"]), 2), round(float(e["incert_dom"]), 2)],
                      [round(float(e["force_ext"]), 2), round(float(e["incert_ext"]), 2)]],
            "terrain": round(float(e["terrain"]), 2),
            "buts_attendus": [round(float(lam[0]), 2), round(float(lam[1]), 2)],
            "reel": [score_reel(buts, match, pt) for pt in POINTS] if match in set(annotes) else None,
            # actions annotées (affichage seulement) : [mi-temps, minute, code, 0 domicile / 1 extérieur / -1]
            "evenements": [[*temps(a.mi_temps, a.minute), a.code, {"dom": 0, "ext": 1}.get(a.equipe, -1)]
                           for a in ev_match[match].itertuples()] if match in ev_match else None,
            "points": points,
        })
    matchs_json.sort(key=lambda x: (x["competition"], x["date"]))

    ev = evaluer([x for x in matchs_json if x["id"] in set(annotes)], infos, annotes)
    ev_inedits = evaluer([x for x in matchs_json if x["id"] in set(inedits)], infos, inedits)
    t = pd.DataFrame(ev).set_index("point")
    t["bon_resultat (37 inédits)"] = [e["bon_resultat"] for e in ev_inedits]
    print(f"\n=== Prédiction en direct ({len(annotes)} matchs annotés, validation croisée) ===")
    print(t.round(3).to_string())
    print(f"Repère : toujours « victoire à domicile » -> {(infos.loc[annotes, 'resultat'] == 1).mean():.0%}")
    comp = comparer_avant_match(tous, infos, annotes, inedits,
                                {"Kalman": force, f"Elo classique (K = {ELO_K:.0f})": calculer_elo(tous)})
    print("\n=== Force des équipes au coup d'envoi : filtre de Kalman contre Elo classique ===")
    print(comp.round(3).to_string())

    donnees_page = {"points": POINTS, "evaluation": ev, "evaluation_inedits": ev_inedits,
                    "repere_domicile": float((infos.loc[annotes, "resultat"] == 1).mean()),
                    "nb_annotes": len(annotes), "nb_inedits": len(inedits),
                    "comparaison_force": comp.reset_index().to_dict(orient="records"), "matchs": matchs_json}
    with open(FICHIER_JSON, "w", encoding="utf-8") as f:
        json.dump(donnees_page, f, ensure_ascii=False)
    construire_page(donnees_page)


def construire_page(donnees_page):
    with open(GABARIT, encoding="utf-8") as f:
        gabarit = f.read()
    js = json.dumps(donnees_page, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    with open(PAGE, "w", encoding="utf-8") as f:
        f.write(gabarit.replace("/*DONNEES*/null", js))
    print(f"\nPage écrite : {PAGE} ({os.path.getsize(PAGE) / 1e6:.1f} Mo)")


if __name__ == "__main__":
    import sys
    if "--page" in sys.argv:  # reconstruire seulement la page à partir des prédictions déjà calculées
        with open(FICHIER_JSON, encoding="utf-8") as f:
            construire_page(json.load(f))
    else:
        main()
