"""
Prédire le résultat d'un match de foot à partir des X premières minutes de commentaires (version 3).

Nouveautés de la v3 :
1. Score actuel reconstruit : 0-0 au coup d'envoi, +1 à chaque but détecté, corrigé quand un score est
   cité. Le modèle sait donc distinguer « 0-0 à la 80e » (le nul devient probable) de « 0-0 à la 10e »
   (presque aucune information sur le résultat final).
2. Force des équipes avant le match : classement Elo calculé uniquement avec les matchs antérieurs.
3. Détection plus robuste : scores historiques ignorés (« last season it finished 1-3 »), score monotone.
   Le script valider_buts.py mesure la qualité de la détection avec les annotations officielles SoccerNet.
4. Qualité des données : recalage sur le coup d'envoi, choix de la meilleure version Whisper par mi-temps.
5. Sentiment : VADER enrichi d'un lexique football (ou un modèle transformer en option), avec cache disque.
6. Dynamique : sentiment et mentions sur toute la fenêtre (0 → X) ET sur la dernière tranche (X-10 → X).
7. Un modèle unique pour toutes les minutes (la minute est une variable), en plus du modèle par fenêtre.

Installation (environnement nlp-foot) :
    pip install datasets pandas numpy scikit-learn vaderSentiment
Options :
    pip install transformers torch   (SENTIMENT = "transformer")
    pip install SoccerNet            (valider_buts.py)
"""
import os
import re
from collections import defaultdict
from math import exp, factorial

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              VotingClassifier)
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, mean_absolute_error
from sklearn.model_selection import (GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold,
                                     cross_val_predict)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------------
# Paramètres
# ----------------------------------------------------------------------------
WHISPER = "whisper_v2"                  # version utilisée si COMBINER_WHISPER = False
VERSIONS_WHISPER = ["whisper_v1", "whisper_v2", "whisper_v3"]
COMBINER_WHISPER = True                 # garde, pour chaque mi-temps, la version la moins hallucinée
SPLIT = "en"                            # "en" = traduit en anglais
FIN_MATCH = 120                         # « fin du match » : tous les commentaires, temps additionnel compris
X_MINUTES = list(range(10, 100, 10)) + [FIN_MATCH]
MIN_SEGMENTS = 5                        # on ignore les matchs avec trop peu de commentaires
TRANCHE_RECENTE = 10                    # dernière tranche : de X-10 à X minutes
RECALER_COUP_ENVOI = True               # décale les minutes sur le coup d'envoi détecté dans le texte
SENTIMENT = "vader"                     # "vader" (rapide) ou "transformer" (plus fin, beaucoup plus lent)
MODELE_TRANSFORMER = "cardiffnlp/twitter-roberta-base-sentiment-latest"
MODELE = "auto"                         # "unique" (un modèle pour toutes les minutes), "fenetre" (un par
                                        # fenêtre) ou "auto" (le plus précis, fenêtre par fenêtre)
EVALUER_V1 = True                       # compare aussi avec la v1 dans modele_foot.py
ELO_INIT, ELO_K, ELO_AVANTAGE_DOM = 1500.0, 20.0, 60.0
MASQUER_EQUIPES = True                  # remplace les noms d'équipes par "equipedom" / "equipeext"
SEED = 42

PATTERN_MATCH = r"^(\d{4}-\d{2}-\d{2}) - (\d{2}-\d{2}) (.+?) (\d+) - (\d+) (.+)$"
MOTS_GENERIQUES = {
    "fc", "cf", "ac", "as", "sc", "rb", "sv", "vfb", "vfl", "real", "club", "united",
    "city", "athletic", "sporting", "borussia", "de", "la", "el", "the",
}

# Lexiques (les commentaires sont traduits en anglais)
RE_BUT = (r"(?i)\b(?:goal|goals|scores|scored|scoring|nets|netted|back of the net|finds the net|"
          r"equali[sz](?:er|es|ed|ing)|own goal|converts the penalty|penalty converted)\b")
RE_PAS_BUT = r"(?i)\b(?:goal kick|goal line|goal-line|no goal|disallowed|offside|ruled out)\b"
RE_ROUGE = r"(?i)\b(?:red card|sent off|sending off|second yellow)\b"
RE_OCCASION = r"(?i)\b(?:chance|shot|shoots|save|saves|saved|post|crossbar|header|attempt|on target)\b"
RE_EGALITE = r"(?i)\b(?:level|all square|draw|drawing|goalless|nil[- ]nil|parity|equali[sz]er)\b"
RE_MENE = r"(?i)\b(?:lead|leads|leading|ahead|in front|winning)\b"
RE_MENE_PAS = r"(?i)\b(?:trail|trails|trailing|behind|losing)\b"
RE_HISTORIQUE = (r"(?i)\b(?:last (?:season|week|weekend|time|year|meeting|match|game|visit)|previous|"
                 r"first leg|reverse fixture|ago|history|record|head to head|in (?:19|20)\d\d)\b")
RE_COUP_ENVOI = (r"(?i)\b(?:kick[- ]?off|kicks off|kicked off|under ?way|we are off|we're off|"
                 r"the (?:game|match) (?:begins|starts|has started)|second half (?:begins|starts))\b")

NOMBRES = {"nil": 0, "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
RE_SCORE_CHIFFRES = re.compile(r"(?<![\d.:/-])(\d)\s*(?:-|–|:|\bto\b)\s*(\d)(?![\d.:/-])", re.I)
RE_SCORE_MOTS = re.compile(r"\b(nil|zero|one|two|three|four|five)[- ](nil|zero|one|two|three|four|five)\b", re.I)

# Mots de football absents ou mal notés dans VADER (échelle de -4 à +4)
LEXIQUE_FOOT = {
    "goal": 1.5, "goals": 1.2, "scores": 1.5, "scored": 1.5, "equaliser": 1.5, "equalizer": 1.5,
    "lead": 0.8, "clinical": 1.8, "dominating": 1.5, "dominate": 1.2, "dominant": 1.5, "pressing": 0.6,
    "chance": 0.8, "chances": 0.8, "dangerous": 0.6, "threat": 0.5, "superb": 2.5, "stunning": 2.5,
    "cracker": 2.0, "screamer": 2.0, "miss": -1.2, "misses": -1.2, "missed": -1.3, "wasteful": -1.5,
    "wide": -0.4, "blocked": -0.4, "woodwork": -0.5, "crossbar": -0.4, "offside": -0.6,
    "foul": -1.0, "fouled": -0.8, "booked": -1.0, "booking": -1.0, "concede": -1.5, "conceded": -1.5,
    "concedes": -1.5, "blunder": -2.2, "howler": -2.2, "sloppy": -1.5, "clumsy": -1.3, "injured": -1.5,
}

NUM_V1 = ["nb_segments", "sent_moy", "sent_dom", "sent_ext", "sent_diff",
          "ment_dom", "ment_ext", "ment_diff"]
NUM = NUM_V1 + [
    # dernière tranche (X-10 → X) et évolution par rapport à toute la fenêtre
    "sent_recent", "sent_recent_dom", "sent_recent_ext", "sent_recent_diff",
    "ment_recent_dom", "ment_recent_ext", "ment_recent_diff", "occ_recent_diff", "egalite_recent",
    "momentum_sent", "momentum_ment",
    # événements
    "buts_evt", "buts_evt_dom", "buts_evt_ext", "buts_evt_diff",
    "rouge_dom", "rouge_ext", "rouge_diff", "occ_diff",
    # dernier score cité
    "sc_connu", "sc_egal", "sc_ecart", "sc_total", "sc_age", "sc_or_connu", "sc_or_diff", "sc_or_age",
    # score actuel reconstruit
    "etat_dom", "etat_ext", "etat_diff", "etat_total", "etat_nul", "etat_conf_age", "etat_non_attr",
    # temps et force des équipes
    "avancement", "couverture", "elo_dom", "elo_ext", "elo_diff", "elo_proba_dom", "elo_nb_min",
    # interactions (utiles à la régression logistique)
    "etat_diff_x_av", "etat_nul_x_av", "elo_diff_x_reste", "sent_diff_x_av",
]


# ----------------------------------------------------------------------------
# 1. Chargement et préparation
# ----------------------------------------------------------------------------
def charger_version(version):
    ds = load_dataset("SoccerNet/SN-echoes", version)
    return ds[SPLIT].to_pandas().assign(version=version)


def choisir_meilleure_version(df):
    """« Mixed selection » simplifiée : pour chaque mi-temps, la version Whisper la moins hallucinée.
    Hallucination = phrase déjà vue dans les 5 commentaires précédents (boucles « Thank you. » /
    « Thank you for watching. » comprises) ou phrase qui domine la mi-temps.
    À qualité égale, on garde la version WHISPER."""
    df = df.copy()
    df["text"] = df["text"].fillna("").str.strip()
    df = df.sort_values(["game", "version", "segment_index"])
    cle = ["game", "version"]
    g = df.groupby(cle)["text"]
    rep = np.zeros(len(df), dtype=bool)
    for k in range(1, 6):
        rep |= df["text"].eq(g.shift(k)).to_numpy()
    df["_rep"] = rep
    taille = df.groupby(cle)["segment_index"].transform("size")
    df["_top"] = df.groupby(cle + ["text"])["segment_index"].transform("size") / taille
    q = df.groupby(cle).agg(rep=("_rep", "mean"), top=("_top", "max"), uniques=("text", "nunique")).reset_index()
    q["score"] = (q["rep"] + q["top"]).round(4)
    q["defaut"] = (q["version"] != WHISPER).astype(int)
    meilleure = (q.sort_values(["game", "score", "defaut", "uniques"], ascending=[True, True, True, False])
                 .drop_duplicates("game"))
    print("Version Whisper retenue par mi-temps :")
    print(meilleure["version"].value_counts().to_string())
    garder = pd.MultiIndex.from_frame(meilleure[cle])
    masque = pd.MultiIndex.from_frame(df[cle]).isin(garder)
    return df[masque].drop(columns=["_rep", "_top"])


def charger_donnees():
    if COMBINER_WHISPER:
        df = pd.concat([charger_version(v) for v in VERSIONS_WHISPER], ignore_index=True)
        return choisir_meilleure_version(df)
    return charger_version(WHISPER)


def preparer(df):
    """Extrait les infos de la colonne 'game', nettoie le texte, calcule la minute de match."""
    parts = df["game"].str.split("/", expand=True)
    df = df.copy()
    df["championnat"] = parts[0]
    df["saison"] = parts[1]
    df["match"] = parts[2]
    df["mi_temps"] = pd.to_numeric(parts[3], errors="coerce")

    infos = df["match"].str.extract(PATTERN_MATCH)
    infos.columns = ["date", "heure", "equipe_dom", "buts_dom", "buts_ext", "equipe_ext"]
    df = pd.concat([df, infos], axis=1).dropna(subset=["equipe_dom", "mi_temps"])

    df["buts_dom"] = df["buts_dom"].astype(int)
    df["buts_ext"] = df["buts_ext"].astype(int)
    df["mi_temps"] = df["mi_temps"].astype(int)
    df["diff_buts"] = df["buts_dom"] - df["buts_ext"]
    df["resultat"] = np.sign(df["diff_buts"]).astype(int)   # 1 domicile, 0 nul, -1 extérieur

    # Nettoyage + suppression des répétitions consécutives (hallucinations Whisper)
    df["text"] = df["text"].fillna("").str.strip()
    df = df[df["text"] != ""]
    df = df.sort_values(["game", "segment_index"])
    repete = df["text"].eq(df.groupby("game")["text"].shift())
    df = df[~repete].copy()

    # Minute : start_time est en secondes depuis le début de la vidéo de la mi-temps, qui ne commence
    # pas forcément au coup d'envoi. On recale sur la première mention du coup d'envoi (10 premières min).
    df["minute_video"] = df["start_time"] / 60
    if RECALER_COUP_ENVOI:
        envoi = df[df["text"].str.contains(RE_COUP_ENVOI) & (df["start_time"] < 600)]
        decalage = df["game"].map(envoi.groupby("game")["start_time"].min()).fillna(0.0)
    else:
        decalage = 0.0
    df["decalage"] = decalage / 60
    df["minute"] = df["minute_video"] - df["decalage"] + (df["mi_temps"] - 1) * 45
    return df


# ----------------------------------------------------------------------------
# 2. Sentiment (avec cache disque : calculé une seule fois)
# ----------------------------------------------------------------------------
def sentiment_vader(textes):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    sia.lexicon.update(LEXIQUE_FOOT)
    return [sia.polarity_scores(t)["compound"] for t in textes]


def sentiment_transformer(textes):
    import torch
    from transformers import pipeline
    device = "mps" if torch.backends.mps.is_available() else -1
    clf = pipeline("sentiment-analysis", model=MODELE_TRANSFORMER, top_k=None, device=device, truncation=True)
    scores = []
    for i in range(0, len(textes), 512):
        for res in clf(textes[i:i + 512], batch_size=64):
            d = {r["label"].lower(): r["score"] for r in res}
            scores.append(d.get("positive", 0.0) - d.get("negative", 0.0))
        print(f"  sentiment : {min(i + 512, len(textes))}/{len(textes)}")
    return scores


def ajouter_sentiment(df, minute_max=FIN_MATCH):
    """Score de sentiment de -1 (négatif) à +1 (positif) pour chaque commentaire."""
    df = df[df["minute"] < minute_max].copy()
    chemin = f"cache_sentiment_{SENTIMENT}_v3.parquet"
    cache = {}
    if os.path.exists(chemin):
        c = pd.read_parquet(chemin)
        cache = dict(zip(c["text"], c["sent"]))
    manquants = [t for t in df["text"].unique() if t not in cache]
    if manquants:
        print(f"Sentiment ({SENTIMENT}) : {len(manquants)} textes à calculer...")
        f = sentiment_transformer if SENTIMENT == "transformer" else sentiment_vader
        cache.update(zip(manquants, f(manquants)))
        pd.DataFrame({"text": list(cache.keys()), "sent": list(cache.values())}).to_parquet(chemin, index=False)
    df["sent"] = df["text"].map(cache)
    return df


# ----------------------------------------------------------------------------
# 3. Features par commentaire : équipes, événements, score cité, score actuel
# ----------------------------------------------------------------------------
def mots_cles(nom):
    tokens = re.findall(r"[A-Za-zÀ-ÿ]+", nom.lower())
    cles = [t for t in tokens if len(t) > 2 and t not in MOTS_GENERIQUES]
    return set(cles or tokens)


def motif(mots):
    return r"(?i)\b(?:" + "|".join(re.escape(m) for m in sorted(mots)) + r")\b"


def motifs_equipes(dom, ext):
    kd, ke = mots_cles(dom), mots_cles(ext)
    commun = kd & ke  # ex. deux équipes « Manchester ... »
    kd, ke = (kd - commun) or kd, (ke - commun) or ke
    return motif(kd), motif(ke)


def dernier_score(texte):
    """Dernier score cité dans un commentaire : (position, a, b) ou None. Ignore les schémas 4-4-2."""
    trouves = [(m.start(), int(m.group(1)), int(m.group(2))) for m in RE_SCORE_CHIFFRES.finditer(texte)]
    for m in RE_SCORE_MOTS.finditer(texte):
        a, b = NOMBRES[m.group(1).lower()], NOMBRES[m.group(2).lower()]
        if (a, b) != (1, 2):  # « a one-two » est une combinaison de passes
            trouves.append((m.start(), a, b))
    trouves = [t for t in trouves if t[1] + t[2] <= 10]
    return max(trouves) if trouves else None


def orienter(texte, a, b, pos_score, p_dom, p_ext):
    """Score cité converti en (buts domicile, buts extérieur), ou None si on ne sait pas à qui il revient."""
    if a == b:
        return (a, b)
    md, me = re.search(p_dom, texte), re.search(p_ext, texte)
    haut, bas = max(a, b), min(a, b)
    if md and me:  # « Chelsea 2-1 Burnley » : la première équipe citée a le premier chiffre
        return (a, b) if md.start() < me.start() else (b, a)
    if md:
        if re.search(RE_MENE, texte):
            return (haut, bas)
        if re.search(RE_MENE_PAS, texte):
            return (bas, haut)
        if md.start() < pos_score:
            return (a, b)
    if me:
        if re.search(RE_MENE, texte):
            return (bas, haut)
        if re.search(RE_MENE_PAS, texte):
            return (haut, bas)
        if me.start() < pos_score:
            return (b, a)
    return None


def voisinage(df, flag):
    """Vrai si le drapeau est vrai dans le commentaire i ou ses voisins (i-1, i+1, i+2) du même match."""
    s = pd.Series(flag.astype(int), index=df.index)
    g = s.groupby(df["match"])
    out = flag.copy()
    for k in (1, -1, -2):
        out |= g.shift(k).fillna(0).to_numpy() > 0
    return out


def debut_evenement(df, flag, ecart=3.0):
    """Ne garde que le premier commentaire d'une série (mentions répétées d'un même but)."""
    m = df["minute"].where(flag)
    precedent = m.groupby(df["match"]).ffill().groupby(df["match"]).shift(1)
    return flag & ~((df["minute"] - precedent) <= ecart).to_numpy()


def reconstruire_etat(df, sc_a, sc_b, or_dom, or_ext):
    """Score actuel estimé après chaque commentaire, calculé uniquement avec le passé.
    Départ à 0-0, +1 par but détecté et attribué, recalé quand un score plausible est cité."""
    n = len(df)
    e_dom, e_ext, e_conf, e_na = (np.full(n, np.nan) for _ in range(4))
    but = df["evt_but"].to_numpy()
    b_dom = df["evt_but_dom"].to_numpy()
    b_ext = df["evt_but_ext"].to_numpy()
    a_score = ~np.isnan(sc_a)
    matchs = df["match"].to_numpy()
    minutes = df["minute"].to_numpy()
    courant = None
    sd = se = non_attr = 0
    conf = 0.0
    for j in np.flatnonzero(but | a_score):
        if matchs[j] != courant:
            courant, sd, se, conf, non_attr = matchs[j], 0, 0, 0.0, 0
        if b_dom[j]:
            sd += 1
        elif b_ext[j]:
            se += 1
        elif but[j]:
            non_attr += 1
        if a_score[j]:
            if not np.isnan(or_dom[j]):
                cands = [(int(or_dom[j]), int(or_ext[j]))]
            else:
                cands = [(int(sc_a[j]), int(sc_b[j])), (int(sc_b[j]), int(sc_a[j]))]
            dist = [abs(c[0] - sd) + abs(c[1] - se) for c in cands]
            k = int(np.argmin(dist))
            ambigu = len(cands) == 2 and cands[0] != cands[1] and dist[0] == dist[1]
            total, total_cite = sd + se, sum(cands[k])
            plausible = total - 1 <= total_cite <= total + 2 + non_attr  # un score ne baisse pas
            if plausible and not ambigu:
                sd, se = cands[k]
                conf, non_attr = minutes[j], 0
        e_dom[j], e_ext[j], e_conf[j], e_na[j] = sd, se, conf, non_attr
    etat = pd.DataFrame({"etat_dom": e_dom, "etat_ext": e_ext, "etat_conf": e_conf,
                         "etat_non_attr": e_na}, index=df.index)
    etat = etat.groupby(df["match"]).ffill()
    return etat.fillna({"etat_dom": 0, "etat_ext": 0, "etat_conf": 0.0, "etat_non_attr": 0})


def enrichir(df):
    """Ajoute à chaque commentaire : mentions d'équipe, buts, cartons, occasions, score cité, score actuel."""
    df = df.sort_values(["match", "minute"]).reset_index(drop=True)
    txt = df["text"]
    minutes = df["minute"].to_numpy()
    est_but = (txt.str.contains(RE_BUT) & ~txt.str.contains(RE_PAS_BUT)).to_numpy()
    est_rouge = txt.str.contains(RE_ROUGE).to_numpy()
    est_occ = txt.str.contains(RE_OCCASION).to_numpy()
    df["est_egalite"] = txt.str.contains(RE_EGALITE)
    historique = txt.str.contains(RE_HISTORIQUE).to_numpy()

    # Scores cités, hors scores historiques et hors avant-match
    scores = txt.map(dernier_score)
    valide = scores.notna().to_numpy() & ~historique & (minutes >= 2)
    sc_a = np.array([np.nan if s is None else s[1] for s in scores], dtype=float)
    sc_b = np.array([np.nan if s is None else s[2] for s in scores], dtype=float)
    sc_a[~valide] = np.nan
    sc_b[~valide] = np.nan

    dom = np.zeros(len(df), dtype=bool)
    ext = np.zeros(len(df), dtype=bool)
    or_dom = np.full(len(df), np.nan)
    or_ext = np.full(len(df), np.nan)
    texte = np.empty(len(df), dtype=object)
    for _, idx in df.groupby("match", sort=False).indices.items():
        g = df.iloc[idx]
        p_dom, p_ext = motifs_equipes(g["equipe_dom"].iloc[0], g["equipe_ext"].iloc[0])
        dom[idx] = g["text"].str.contains(p_dom).to_numpy()
        ext[idx] = g["text"].str.contains(p_ext).to_numpy()
        t = g["text"]
        if MASQUER_EQUIPES:
            t = t.str.replace(p_dom, " equipedom ", regex=True).str.replace(p_ext, " equipeext ", regex=True)
        texte[idx] = t.to_numpy()
        for j in idx[valide[idx]]:
            o = orienter(txt.iat[j], int(sc_a[j]), int(sc_b[j]), scores.iat[j][0], p_dom, p_ext)
            if o is not None:
                or_dom[j], or_ext[j] = o

    df["dom"], df["ext"], df["texte_masque"] = dom, ext, texte
    df["sc_a"], df["sc_b"], df["sc_or"] = sc_a, sc_b, or_dom - or_ext

    # Attribution d'un événement : équipe citée seule dans le commentaire, sinon dans ses voisins
    v_dom, v_ext = voisinage(df, dom), voisinage(df, ext)
    seul_dom = (dom & ~ext) | (~dom & ~ext & v_dom & ~v_ext)
    seul_ext = (ext & ~dom) | (~dom & ~ext & v_ext & ~v_dom)
    df["evt_but"] = debut_evenement(df, est_but)
    df["evt_but_dom"] = debut_evenement(df, est_but & seul_dom)
    df["evt_but_ext"] = debut_evenement(df, est_but & seul_ext)
    df["evt_rouge_dom"] = debut_evenement(df, est_rouge & seul_dom)
    df["evt_rouge_ext"] = debut_evenement(df, est_rouge & seul_ext)
    df["occ_dom"] = est_occ & dom & ~ext
    df["occ_ext"] = est_occ & ext & ~dom

    etat = reconstruire_etat(df, sc_a, sc_b, or_dom, or_ext)
    return pd.concat([df, etat], axis=1)


def calculer_elo(df):
    """Elo de chaque équipe AVANT chaque match, calculé uniquement avec les matchs antérieurs du dataset."""
    m = df.groupby("match").agg(date=("date", "first"), heure=("heure", "first"),
                                dom=("equipe_dom", "first"), ext=("equipe_ext", "first"),
                                bd=("buts_dom", "first"), be=("buts_ext", "first")).sort_values(["date", "heure"])
    elo = defaultdict(lambda: ELO_INIT)
    nb = defaultdict(int)
    lignes = []
    for match, r in m.iterrows():
        ed, ee = elo[r["dom"]], elo[r["ext"]]
        attendu = 1 / (1 + 10 ** ((ee - ed - ELO_AVANTAGE_DOM) / 400))
        lignes.append((match, ed, ee, ed - ee, attendu, min(nb[r["dom"]], nb[r["ext"]])))
        ecart = abs(r["bd"] - r["be"])
        mult = 1.0 if ecart <= 1 else (1.5 if ecart == 2 else (11 + ecart) / 8)
        reel = 1.0 if r["bd"] > r["be"] else (0.5 if r["bd"] == r["be"] else 0.0)
        delta = ELO_K * mult * (reel - attendu)
        elo[r["dom"]] += delta
        elo[r["ext"]] -= delta
        nb[r["dom"]] += 1
        nb[r["ext"]] += 1
    cols = ["match", "elo_dom", "elo_ext", "elo_diff", "elo_proba_dom", "elo_nb_min"]
    return pd.DataFrame(lignes, columns=cols).set_index("match")


def preparer_tout():
    """Chargement, nettoyage, sentiment, features par commentaire et Elo."""
    df = preparer(charger_donnees())
    df = ajouter_sentiment(df, minute_max=FIN_MATCH)
    df = enrichir(df)
    elo = calculer_elo(df)
    print(f"{df['match'].nunique()} matchs, {len(df)} commentaires après nettoyage")
    return df, elo


# ----------------------------------------------------------------------------
# 4. Une ligne par match pour une fenêtre [0, x_min)
# ----------------------------------------------------------------------------
def construire_matchs(df, x_min, elo):
    d = df[df["minute"] < x_min]
    n = d.groupby("match")["text"].transform("size")
    d = d[n >= MIN_SEGMENTS]
    if x_min >= FIN_MATCH:
        fin = d.groupby("match")["minute"].transform("max")
    else:
        fin = pd.Series(float(x_min), index=d.index)
    rec = (d["minute"] >= fin - TRANCHE_RECENTE).to_numpy()  # ex. X = 30 : de 20 à 30 minutes
    dom, ext, s = d["dom"].to_numpy(), d["ext"].to_numpy(), d["sent"].to_numpy()
    d = d.assign(
        fin=fin, rec=rec, s_dom=s * dom, s_ext=s * ext,
        r_dom=rec & dom, r_ext=rec & ext, r_s=s * rec, r_s_dom=s * (rec & dom), r_s_ext=s * (rec & ext),
        r_occ_dom=rec & d["occ_dom"].to_numpy(), r_occ_ext=rec & d["occ_ext"].to_numpy(),
        r_egal=rec & d["est_egalite"].to_numpy(),
    )
    g = d.groupby("match", sort=False)
    a = g.agg(
        nb_segments=("text", "size"), sent_moy=("sent", "mean"),
        n_dom=("dom", "sum"), n_ext=("ext", "sum"), s_dom=("s_dom", "sum"), s_ext=("s_ext", "sum"),
        n_rec=("rec", "sum"), r_dom=("r_dom", "sum"), r_ext=("r_ext", "sum"), r_s=("r_s", "sum"),
        r_s_dom=("r_s_dom", "sum"), r_s_ext=("r_s_ext", "sum"),
        r_occ_dom=("r_occ_dom", "sum"), r_occ_ext=("r_occ_ext", "sum"), r_egal=("r_egal", "sum"),
        buts_evt=("evt_but", "sum"), buts_evt_dom=("evt_but_dom", "sum"), buts_evt_ext=("evt_but_ext", "sum"),
        rouge_dom=("evt_rouge_dom", "sum"), rouge_ext=("evt_rouge_ext", "sum"),
        occ_dom=("occ_dom", "sum"), occ_ext=("occ_ext", "sum"),
        minute_max=("minute", "max"), fin=("fin", "first"),
        texte=("texte_masque", " ".join),
        buts_dom=("buts_dom", "first"), buts_ext=("buts_ext", "first"),
        diff_buts=("diff_buts", "first"), resultat=("resultat", "first"),
    )
    texte_recent = d[d["rec"]].groupby("match")["texte_masque"].agg(" ".join).reindex(a.index).fillna("")
    dernier = g.tail(1).set_index("match").reindex(a.index)  # dernier commentaire de la fenêtre
    un = lambda s: s.clip(lower=1)

    f = pd.DataFrame(index=a.index)
    nb = a["nb_segments"]
    # Toute la fenêtre (0 → X)
    f["nb_segments"] = nb
    f["sent_moy"] = a["sent_moy"]
    f["sent_dom"] = a["s_dom"] / un(a["n_dom"])
    f["sent_ext"] = a["s_ext"] / un(a["n_ext"])
    f["sent_diff"] = f["sent_dom"] - f["sent_ext"]
    f["ment_dom"] = a["n_dom"] / nb
    f["ment_ext"] = a["n_ext"] / nb
    f["ment_diff"] = f["ment_dom"] - f["ment_ext"]
    # Dernière tranche (X-10 → X)
    f["sent_recent"] = a["r_s"] / un(a["n_rec"])
    f["sent_recent_dom"] = a["r_s_dom"] / un(a["r_dom"])
    f["sent_recent_ext"] = a["r_s_ext"] / un(a["r_ext"])
    f["sent_recent_diff"] = f["sent_recent_dom"] - f["sent_recent_ext"]
    f["ment_recent_dom"] = a["r_dom"] / un(a["n_rec"])
    f["ment_recent_ext"] = a["r_ext"] / un(a["n_rec"])
    f["ment_recent_diff"] = f["ment_recent_dom"] - f["ment_recent_ext"]
    f["occ_recent_diff"] = (a["r_occ_dom"] - a["r_occ_ext"]) / un(a["n_rec"])
    f["egalite_recent"] = a["r_egal"] / un(a["n_rec"])
    f["momentum_sent"] = f["sent_recent_diff"] - f["sent_diff"]
    f["momentum_ment"] = f["ment_recent_diff"] - f["ment_diff"]
    # Événements
    f["buts_evt"] = a["buts_evt"]
    f["buts_evt_dom"] = a["buts_evt_dom"]
    f["buts_evt_ext"] = a["buts_evt_ext"]
    f["buts_evt_diff"] = a["buts_evt_dom"] - a["buts_evt_ext"]
    f["rouge_dom"] = a["rouge_dom"]
    f["rouge_ext"] = a["rouge_ext"]
    f["rouge_diff"] = a["rouge_dom"] - a["rouge_ext"]
    f["occ_diff"] = (a["occ_dom"] - a["occ_ext"]) / nb

    # Dernier score cité (le plus récent dans la fenêtre)
    sc = d[d["sc_a"].notna()].groupby("match").tail(1).set_index("match").reindex(a.index)
    f["sc_connu"] = sc["sc_a"].notna().astype(int)
    f["sc_egal"] = (sc["sc_a"] == sc["sc_b"]).astype(int)
    f["sc_ecart"] = (sc["sc_a"] - sc["sc_b"]).abs()
    f["sc_total"] = sc["sc_a"] + sc["sc_b"]
    f["sc_age"] = (a["minute_max"] - sc["minute"]).fillna(a["minute_max"])
    so = d[d["sc_or"].notna()].groupby("match").tail(1).set_index("match").reindex(a.index)
    f["sc_or_connu"] = so["sc_or"].notna().astype(int)
    f["sc_or_diff"] = so["sc_or"]
    f["sc_or_age"] = (a["minute_max"] - so["minute"]).fillna(a["minute_max"])

    # Score actuel reconstruit à la fin de la fenêtre
    f["etat_dom"] = dernier["etat_dom"]
    f["etat_ext"] = dernier["etat_ext"]
    f["etat_diff"] = f["etat_dom"] - f["etat_ext"]
    f["etat_total"] = f["etat_dom"] + f["etat_ext"]
    f["etat_nul"] = (f["etat_diff"] == 0).astype(int)
    f["etat_conf_age"] = (a["minute_max"] - dernier["etat_conf"]).clip(lower=0)
    f["etat_non_attr"] = dernier["etat_non_attr"]

    # Temps et force des équipes
    av = a["fin"].clip(lower=0, upper=95) / 95
    f["avancement"] = av
    f["couverture"] = a["minute_max"] / un(a["fin"])
    e = elo.reindex(a.index)
    for c in ["elo_dom", "elo_ext", "elo_diff", "elo_proba_dom", "elo_nb_min"]:
        f[c] = e[c]
    f["etat_diff_x_av"] = f["etat_diff"] * av
    f["etat_nul_x_av"] = f["etat_nul"] * av
    f["elo_diff_x_reste"] = f["elo_diff"] * (1 - av)
    f["sent_diff_x_av"] = f["sent_diff"] * av

    m = f[NUM].fillna(0)
    m["texte"] = a["texte"]
    m["texte_recent"] = texte_recent
    m["score_cite"] = [None if pd.isna(x) else f"{int(x)}-{int(y)}" for x, y in zip(sc["sc_a"], sc["sc_b"])]
    m["score_cite_min"] = sc["minute"]
    m["etat_conf"] = dernier["etat_conf"]
    for c in ["buts_dom", "buts_ext", "diff_buts", "resultat"]:
        m[c] = a[c]
    m["fenetre"] = x_min
    return m.reset_index()


def empiler(df, elo, fenetres):
    """Toutes les fenêtres les unes sous les autres (une ligne par match et par fenêtre)."""
    return pd.concat([construire_matchs(df, x, elo) for x in fenetres], ignore_index=True)


# ----------------------------------------------------------------------------
# 5. Modèles
# ----------------------------------------------------------------------------
def vect_texte(n_features):
    """TF-IDF à mémoire bornée (hachage) : indispensable quand on empile toutes les fenêtres."""
    return Pipeline([
        ("hash", HashingVectorizer(ngram_range=(1, 2), stop_words="english", alternate_sign=False,
                                   norm=None, n_features=n_features)),
        ("tfidf", TfidfTransformer(sublinear_tf=True)),
    ])


def modele_v1():
    """La v1, pour comparaison."""
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, sublinear_tf=True, stop_words="english")
    pre = ColumnTransformer([("tfidf", tfidf, "texte"), ("num", StandardScaler(), NUM_V1)])
    return Pipeline([("pre", pre), ("m", LogisticRegression(max_iter=2000, class_weight="balanced"))])


def modele_elo():
    """Avant-match : uniquement la différence d'Elo (avantage du terrain compris)."""
    pre = ColumnTransformer([("num", StandardScaler(), ["elo_diff"])])
    return Pipeline([("pre", pre), ("m", LogisticRegression(max_iter=1000))])


def modele_logistique():
    """Texte de la dernière tranche (le plus informatif et le plus léger) + tous les indicateurs.
    Le texte de toute la fenêtre est résumé par les indicateurs (sentiment, mentions, score...)."""
    pre = ColumnTransformer([
        ("recent", vect_texte(2 ** 17), "texte_recent"),
        ("num", StandardScaler(), NUM),
    ])
    return Pipeline([("pre", pre), ("m", LogisticRegression(max_iter=3000, C=0.5, solver="saga", tol=1e-3))])


def modele_boosting():
    pre = ColumnTransformer([("num", "passthrough", NUM)])
    gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=3,
                                        min_samples_leaf=20, l2_regularization=1.0, random_state=SEED)
    return Pipeline([("pre", pre), ("m", gb)])


def modele_issue():
    """Modèle final : moyenne des probabilités de la logistique et du boosting."""
    return VotingClassifier([("log", modele_logistique()), ("gb", modele_boosting())], voting="soft")


def modele_buts():
    """Buts d'une équipe en fin de match : gradient boosting avec perte de Poisson."""
    pre = ColumnTransformer([("num", "passthrough", NUM)])
    gb = HistGradientBoostingRegressor(loss="poisson", max_iter=300, learning_rate=0.05, max_depth=3,
                                       min_samples_leaf=20, random_state=SEED)
    return Pipeline([("pre", pre), ("m", gb)])


def predictions_oof(pile, unique=True):
    """Prédictions hors-échantillon : chaque match est prédit par un modèle entraîné sans lui.
    Les buts sont prédits comme « score actuel + buts restants à marquer ».
    Renvoie (classes, probabilités, buts attendus domicile, buts attendus extérieur)."""
    y = pile["resultat"]
    cls = np.sort(y.unique())
    proba = np.zeros((len(pile), len(cls)))
    reste_d = (pile["buts_dom"] - pile["etat_dom"]).clip(lower=0)
    reste_e = (pile["buts_ext"] - pile["etat_ext"]).clip(lower=0)
    pr_d, pr_e = np.zeros(len(pile)), np.zeros(len(pile))
    if unique:
        groupes = pile["match"]
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        proba[:] = cross_val_predict(modele_issue(), pile, y, cv=cv, groups=groupes, method="predict_proba")
        cvr = GroupKFold(n_splits=5)
        pr_d[:] = cross_val_predict(modele_buts(), pile, reste_d, cv=cvr, groups=groupes)
        pr_e[:] = cross_val_predict(modele_buts(), pile, reste_e, cv=cvr, groups=groupes)
    else:
        for _, idx in pile.groupby("fenetre").indices.items():
            m = pile.iloc[idx]
            cls_m = list(np.sort(m["resultat"].unique()))
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
            p = cross_val_predict(modele_issue(), m, m["resultat"], cv=cv, method="predict_proba")
            for k, c in enumerate(cls_m):
                proba[idx, list(cls).index(c)] = p[:, k]
            cvr = KFold(n_splits=5, shuffle=True, random_state=SEED)
            pr_d[idx] = cross_val_predict(modele_buts(), m, reste_d.iloc[idx], cv=cvr)
            pr_e[idx] = cross_val_predict(modele_buts(), m, reste_e.iloc[idx], cv=cvr)
    lam_d = pile["etat_dom"].to_numpy() + np.clip(pr_d, 0, None)
    lam_e = pile["etat_ext"].to_numpy() + np.clip(pr_e, 0, None)
    return cls, proba, lam_d, lam_e


def predictions_finales(pile, mode=None):
    """Prédictions du modèle retenu. En mode « auto », pour chaque fenêtre, la variante (modèle unique
    ou modèle par fenêtre) qui a la meilleure précision en validation croisée.
    Renvoie (classes, probabilités, buts attendus dom, buts attendus ext, variante retenue par fenêtre)."""
    mode = mode or MODELE
    variantes = {"unique": True, "fenetre": False}
    if mode != "auto":
        cls, proba, ld, le = predictions_oof(pile, unique=variantes[mode])
        return cls, proba, ld, le, {x: mode for x in pile["fenetre"].unique()}
    oof = {nom: predictions_oof(pile, unique=u) for nom, u in variantes.items()}
    cls = oof["unique"][0]
    proba = np.zeros_like(oof["unique"][1])
    ld, le = np.zeros(len(pile)), np.zeros(len(pile))
    choix = {}
    for x, idx in pile.groupby("fenetre").indices.items():
        y = pile["resultat"].iloc[idx]
        notes = {nom: (metriques(y, cls, o[1][idx])["accuracy"], -metriques(y, cls, o[1][idx])["log_loss"])
                 for nom, o in oof.items()}
        nom = max(notes, key=notes.get)
        choix[x] = nom
        proba[idx], ld[idx], le[idx] = oof[nom][1][idx], oof[nom][2][idx], oof[nom][3][idx]
    return cls, proba, ld, le, choix


def poisson(k, lam):
    return exp(-lam) * lam ** k / factorial(k)


def score_estime(lam_d, lam_e, issue, min_d=0, min_e=0):
    """Pronostic de score final cohérent avec l'issue prédite et jamais inférieur au score actuel.
    1) buts attendus arrondis s'ils donnent l'issue prédite ; 2) sinon le score le plus probable
    (buts indépendants de Poisson) parmi ceux qui la donnent."""
    arr = (max(int(round(max(lam_d, 0))), min_d), max(int(round(max(lam_e, 0))), min_e))
    if np.sign(arr[0] - arr[1]) == issue:
        return list(arr)
    lam_d, lam_e = max(lam_d, 0.05), max(lam_e, 0.05)
    meilleur, p_max = list(arr), -1.0
    for i in range(min_d, min_d + 8):
        for j in range(min_e, min_e + 8):
            if np.sign(i - j) != issue:
                continue
            p = poisson(i, lam_d) * poisson(j, lam_e)
            if p > p_max:
                meilleur, p_max = [i, j], p
    return meilleur


# ----------------------------------------------------------------------------
# 6. Évaluation
# ----------------------------------------------------------------------------
def metriques(y, cls, proba):
    pred = np.asarray(cls)[proba.argmax(axis=1)]
    return {"accuracy": accuracy_score(y, pred), "f1_macro": f1_score(y, pred, average="macro"),
            "log_loss": log_loss(y, proba, labels=cls)}


def evaluer_references(m):
    """Modèles de comparaison, validés fenêtre par fenêtre."""
    y = m["resultat"]
    cls = np.sort(y.unique())
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    modeles = {"Référence (classe majoritaire)": DummyClassifier(strategy="prior"),
               "Avant-match (Elo seul)": modele_elo()}
    if EVALUER_V1:
        modeles["v1 : texte + sentiment"] = modele_v1()
    return {nom: metriques(y, cls, cross_val_predict(mod, m, y, cv=cv, method="predict_proba"))
            for nom, mod in modeles.items()}


def diagnostic(m):
    """Contrôle de l'extraction en fin de match."""
    connus = m[m["sc_connu"] == 1]
    print(f"Score cité trouvé dans {len(connus)}/{len(m)} matchs")
    if len(connus):
        ok = ((connus["sc_egal"] == 1) == (connus["resultat"] == 0)).mean()
        print(f"  « dernier score cité à égalité » ⇔ « match nul » : {ok:.0%}")
    orientes = m[m["sc_or_connu"] == 1]
    if len(orientes):
        ok = (np.sign(orientes["sc_or_diff"]) == orientes["resultat"]).mean()
        print(f"  Score attribué à une équipe dans {len(orientes)} matchs, bon vainqueur : {ok:.0%}")
    exact = ((m["etat_dom"] == m["buts_dom"]) & (m["etat_ext"] == m["buts_ext"])).mean()
    signe = (np.sign(m["etat_diff"]) == m["resultat"]).mean()
    print(f"Score actuel reconstruit en fin de match : score exact {exact:.0%}, bonne issue {signe:.0%}")


def nom_fenetre(x):
    return "Fin du match" if x >= FIN_MATCH else f"{x} premières minutes"


def main():
    df, elo = preparer_tout()
    pile = empiler(df, elo, X_MINUTES)
    print(f"{pile['match'].nunique()} matchs, {len(pile)} lignes (match × fenêtre)")

    variantes = {"v3 : un modèle par fenêtre": False, "v3 : modèle unique": True}
    oof = {}
    for nom, unique in variantes.items():
        print(f"Validation croisée : {nom}...")
        oof[nom] = predictions_oof(pile, unique=unique)

    lignes = []
    for x in X_MINUTES:
        sel = (pile["fenetre"] == x).to_numpy()
        m = pile[sel]
        print(f"\n=== {nom_fenetre(x)} : {len(m)} matchs ===")
        if x >= FIN_MATCH:
            diagnostic(m)
        for nom, r in evaluer_references(m).items():
            lignes.append({"fenetre": nom_fenetre(x), "modele": nom, **r})
        for nom, (cls, proba, ld, le) in oof.items():
            r = metriques(m["resultat"], cls, proba[sel])
            r["mae_buts"] = (mean_absolute_error(m["buts_dom"], ld[sel]) + mean_absolute_error(m["buts_ext"], le[sel])) / 2
            lignes.append({"fenetre": nom_fenetre(x), "modele": nom, **r})
        print(pd.DataFrame([l for l in lignes if l["fenetre"] == nom_fenetre(x)]).round(3).to_string(index=False))

    res = pd.DataFrame(lignes)
    pd.set_option("display.width", 220)
    ordre = [nom_fenetre(x) for x in X_MINUTES]
    for col, titre in [("accuracy", "Précision (part de bonnes issues)"), ("log_loss", "Log loss (plus bas = mieux)")]:
        print(f"\n=== {titre} ===")
        print(res.pivot(index="fenetre", columns="modele", values=col).reindex(ordre).round(3).to_string())
    return df, res


if __name__ == "__main__":
    main()
