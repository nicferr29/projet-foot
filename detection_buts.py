"""
Détecter les buts dans les commentaires sans se faire piéger par le double sens de « goal ».
Usage (environnement conda nlp-foot, dans le dossier du projet) :
    python detection_buts_2.py
"""
import hashlib
import os
import pickle
import re

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import donnees

DOSSIER_RESULTATS = os.path.join(donnees.ICI, "resultats")
N_PLIS = 5
SEED = 42
FENETRE = 8                  # mots de contexte de chaque côté d'une mention[cite: 6]
FENETRE_TEXTE = 24           # contexte plus large, en mots naturels, pour le modèle de langue[cite: 6]
EN_CLAIR = {"<gardien>": "goalkeeper", "<joueur>": "player", "<equipe>": "team", "<autre_equipe>": "another team",
            "_poss_": "'s"}
REGROUPEMENT = 2.5           # minutes : deux annonces plus proches que ça décrivent le même but[cite: 6]
TOLERANCE = (-1.0, 2.5)      # un but détecté entre 1 min avant et 2 min 30 après le vrai but est correct[cite: 6]
MAX_BUTS = 10
POIDS_SCORE_CITE = 0.5       # un score cité compte pour moitié : les commentateurs se répètent[cite: 6]
MODELE_LANGUE = "distilroberta-base"   # modèle de langue pré-entraîné (None pour s'en passer)[cite: 6]
POIDS_LANGUE = 0.15          # poids de ses vecteurs à côté des mots typés[cite: 6]
FICHIER_REPRESENTATIONS = os.path.join(donnees.ICI, "cache", "representations_langue.npz")
CACHE_CV = os.path.join(donnees.ICI, "cache", "cv_detection_buts_v2.pkl")

# Familles de mots qui peuvent annoncer un but[cite: 6]
FAMILLES = [
    ("goal", re.compile(r"^go+a+l+s?$")),            # goal, goals, gooooal[cite: 6]
    ("marquer", re.compile(r"^scor(?:e|es|ed|ing)$")),
    ("filet", re.compile(r"^net(?:s|ted)?$")),
    ("egaliser", re.compile(r"^equali[sz]\w*$")),
    ("score cité", re.compile(r"^_score_$")),        # « 1-0 », « 2:1 »[cite: 6]
]
TOURNURES = [
    ("dedans", re.compile(
        r"\b(?:ball|it|that|shot|header) (?:is|_poss_|was|goes|went|rolls|flies|ends up|slides|creeps) (?:in|inside)\b"
        r"(?! (?:the|his|her|their|a|an|play|possession|touch|front|behind|for|my|our|your|there|here|position|charge))"
        r"|\bin off the (?:post|bar|crossbar|keeper|defender)\b|\b(?:under|beneath|off the underside of) the (?:bar|crossbar)\b")),
    ("placement", re.compile(
        r"\b(?:into|in|inside|to) the (?:near|far|top|bottom|left|right|lower|upper|long|short|low|high) (?:corner|post)\b")),
    ("buteur", re.compile(
        r"\b(?:goalscorer|goal scorer|scorers?|brace|hat trick|hattrick|opener)\b"
        r"|\b(?:doubles? (?:the|their|his) (?:lead|advantage)|makes it (?:one|two|three|four|five|six)"
        r"|pulls? one back|pulled one back|gets? one back|reduces? the (?:deficit|gap|arrears))\b")),
    ("avance", re.compile(
        r"\b(?:takes?|took|taken) the lead\b|\b(?:in|into) the lead\b|\b(?:lead|leads|leading|ahead|in front) (?:through|thanks to)\b"
        r"|\b(?:levels?|levelled|leveled) (?:it|the score|things|the match|the game)\b|\bback on level terms\b|\bback level\b")),
    ("fête", re.compile(
        r"\b(?:celebrat\w*|jubil\w*|erupts?|erupted|explodes?|exploded|delirium|euphori\w*|ecsta\w*"
        r"|(?:going|goes|go|went) (?:crazy|wild|mad|berserk))\b")),
]
FAMILLES_MOTS = [nom for nom, _ in FAMILLES]
RE_SCORE = re.compile(r"(?<![\d.:/,\-–])(\d)\s*(?:[-–:]|\bto\b)\s*(\d)(?![\d]|[.:/,]\d|\s*[-–]\s*\d)")
# Détection des buts contre son camp (own goal) pour inverser l'attribution de l'équipe
RE_CSC = re.compile(
    r"\b(?:own goals?|own net|into (?:his|their) own|in (?:his|their) own|past (?:his|their) own keeper"
    r"|deflected into (?:his|their) own|unfortunate (?:deflection|touch) into)\b",
    re.IGNORECASE,
)

AVANT_CAGE = {"on", "at", "towards", "toward", "in", "inside", "near", "from", "across", "wide", "over",
              "defend", "defends", "defending", "guard", "guards", "guarding", "protect", "protects", "front",
              "attack", "attacking", "approaching", "approaches"}
APRES_CAGE = {"kick", "kicks", "line", "lines", "area", "mouth", "frame", "post", "posts", "keeper", "keepers",
              "net", "box", "difference", "average", "bar", "crossbar"}
NEGATION = {"disallowed", "annulled", "cancelled", "canceled", "offside", "ruled", "var", "chalked"}
RETRO = {"season", "seasons", "career", "year", "years", "week", "weeks", "previous", "record", "tally"}


# ----------------------------------------------------------------------------
# 1. Mots, entités et mentions[cite: 6]
# ----------------------------------------------------------------------------
def tokeniser(texte):
    t = donnees.normaliser(str(texte).replace("’", "'"))
    valeurs = [(int(a), int(b)) for a, b in RE_SCORE.findall(t)]
    t = RE_SCORE.sub(" _score_ ", t)
    t = re.sub(r"(\w)'s\b", r"\1 _poss_", t)
    t = re.sub(r"(\w)s'(?=[\s.,!?]|$)", r"\1s _poss_", t)
    return re.findall(r"_poss_|_score_|\d+|[a-z]+|!", t), valeurs


def dictionnaire_entites(compo, autres_equipes):
    ent = {mot: ("<autre_equipe>", None) for mot in autres_equipes}
    for cote in ("dom", "ext"):
        for cle, typ in (("equipe", "<equipe>"), ("joueurs", "<joueur>"), ("gardien", "<gardien>")):
            for mot in compo[cote][cle]:
                ent[mot] = (typ, cote)
    return ent


def famille(mot):
    for nom, motif in FAMILLES:
        if motif.match(mot):
            return nom
    return None


def regle(g, d, mot, fam):
    g1, g2, g3 = (g + ["", "", ""])[:3]
    d1, d2, d3 = (d + ["", "", ""])[:3]
    proches = set(g[:6] + d[:6])
    if fam == "goal":
        if (g1 == "_poss_" and g2 == "<gardien>") or (d1 in {"of", "from"} and "<gardien>" in (d2, d3)):
            return "cage d'un gardien"
        if g1 in AVANT_CAGE or (g1 == "the" and g2 in AVANT_CAGE) or (g1 in {"of", "the"} and "front" in (g2, g3)):
            return "cage (préposition)"
        if d1 in APRES_CAGE:
            return "cage (nom composé)"
    if g1 == "no" or NEGATION & proches:
        return "but refusé"
    if mot == "goals" or RETRO & proches:
        return "statistique / passé"
    if fam == "marquer" and mot == "score" and g1 in {"the", "a", "final", "current", "level"}:
        return "le score (nom)"
    return None


def extraire_mentions(com, compos):
    lignes = []
    toutes_equipes = {mot for c in compos.values() for cote in c.values() for mot in cote["equipe"]}
    for (match, mt), seg in com.groupby(["match", "mi_temps"], sort=False):
        ent = dictionnaire_entites(compos[match], toutes_equipes)
        types, cotes, temps, idx_seg, mots, scores = [], [], [], [], [], []
        textes = seg["texte"].tolist()
        for i, (texte, minute) in enumerate(zip(textes, seg["minute"])):
            jetons, valeurs = tokeniser(texte)
            valeurs = iter(valeurs)
            for m in jetons:
                typ, cote = ent.get(m, (m, None))
                if typ == "<gardien>":
                    cote = {"dom": "ext", "ext": "dom"}[cote]
                mots.append(m)
                types.append(typ)
                cotes.append(cote)
                temps.append(minute)
                idx_seg.append(i)
                scores.append(next(valeurs) if m == "_score_" else (np.nan, np.nan))
        clair = [f"{int(sc[0])}-{int(sc[1])}" if m == "_score_" else EN_CLAIR.get(t, t)
                 for m, t, sc in zip(mots, types, scores)]
        ancres = [(k, fam) for k, fam in enumerate(map(famille, mots)) if fam]
        phrase = " ".join(mots)
        debuts = np.cumsum([0] + [len(w) + 1 for w in mots[:-1]])
        for nom, motif in TOURNURES:
            for x in motif.finditer(phrase):
                ancres.append((int(np.searchsorted(debuts, x.end() - 1, side="right") - 1), nom))
        for k, fam in sorted(ancres):
            m = mots[k]
            g, d, i = types[max(0, k - FENETRE):k][::-1], types[k + 1:k + 1 + FENETRE], idx_seg[k]
            autour = [c for c in cotes[max(0, k - 30):k + 40] if c is not None]
            extrait = " ".join(textes[max(0, i - 1):i + 2])
            lignes.append({
                "match": match, "mi_temps": mt, "minute": temps[k], "mot": m, "famille": fam,
                "a": scores[k][0], "b": scores[k][1],
                "contexte_clair": clair[max(0, k - FENETRE_TEXTE):k + 1 + FENETRE_TEXTE],
                "position": min(k, FENETRE_TEXTE),
                "gauche": g, "droite": d, "regle": regle(g, d, m, fam),
                "texte": extrait,
                "seg_excl": "!" in textes[i], "seg_court": len(textes[i].split()) <= 4,
                "cote_dom": autour.count("dom"), "cote_ext": autour.count("ext"),
                "est_csc": bool(RE_CSC.search(extrait)),
            })
    m = pd.DataFrame(lignes).sort_values(["match", "mi_temps", "minute"], kind="stable").reset_index(drop=True)
    m["n_mentions"] = m["n_scores"] = m["n_avant"] = 0.0
    est_score = (m["famille"] == "score cité").to_numpy()
    for _, idx in m.groupby(["match", "mi_temps"], sort=False).indices.items():
        t = m["minute"].to_numpy()[idx]
        a, b = np.searchsorted(t, t - 0.5), np.searchsorted(t, t + 0.5, side="right")
        sc = np.cumsum(np.r_[0, est_score[idx]])
        m.loc[idx, "n_mentions"] = b - a - 1
        m.loc[idx, "n_scores"] = sc[b] - sc[a]
        m.loc[idx, "n_avant"] = a - np.searchsorted(t, t - 3)
    return m


def ecart_au_but(m, buts):
    dt = np.full(len(m), 99.0)
    bm = {k: g["minute"].to_numpy() for k, g in buts.groupby(["match", "mi_temps"])}
    for k, idx in m.groupby(["match", "mi_temps"]).indices.items():
        b = bm.get(k)
        if b is not None and len(b):
            e = m["minute"].to_numpy()[idx][:, None] - b[None, :]
            dt[idx] = e[np.arange(len(idx)), np.abs(e).argmin(axis=1)]
    return dt


def etiqueter(m, buts):
    dt = ecart_au_but(m, buts)
    ambigu = ((dt >= -1) & (dt < -0.25)) | ((dt > 1.5) & (dt <= 4))
    return np.where((dt >= -0.25) & (dt <= 1.5), 1.0, np.where(ambigu, np.nan, 0.0))


# ----------------------------------------------------------------------------
# 2. Étape 1 : sens de chaque mention[cite: 6]
# ----------------------------------------------------------------------------
def caracteristiques(r):
    g, d = r["gauche"], r["droite"]
    forme = "gooal" if r["famille"] == "goal" and len(r["mot"]) > 5 else r["mot"]
    f = {f"famille={r['famille']}": 1, f"forme={forme}": 1}
    for i in range(3):
        if i < len(g):
            f[f"g{i + 1}={g[i]}"] = 1
        if i < len(d):
            f[f"d{i + 1}={d[i]}"] = 1
    if len(g) >= 2:
        f[f"g2g1={g[1]}_{g[0]}"] = 1
    if len(d) >= 2:
        f[f"d1d2={d[0]}_{d[1]}"] = 1
    for w in g:
        f[f"G:{w}"] = 1
    for w in d:
        f[f"D:{w}"] = 1
    if r["regle"]:
        f[f"regle={r['regle']}"] = 1
    f["meme_famille_autour"] = sum(famille(w) == r["famille"] for w in g[:5] + d[:5])
    f["seg_excl"] = float(r["seg_excl"])
    f["seg_court"] = float(r["seg_court"])
    f["est_csc"] = float(r.get("est_csc", False))
    for c in ("n_mentions", "n_scores", "n_avant"):
        f[c] = np.log1p(r[c])
    return f


def representations(m):
    if MODELE_LANGUE is None:
        return None
    signature = hashlib.sha1("\n".join(f"{a}|{b}|{c:.4f}|{d}|{' '.join(e)}" for a, b, c, d, e in
                                        zip(m["match"], m["mi_temps"], m["minute"], m["mot"],
                                            m["contexte_clair"])).encode()).hexdigest()
    if os.path.exists(FICHIER_REPRESENTATIONS):
        z = np.load(FICHIER_REPRESENTATIONS)
        if str(z["signature"]) == signature:
            print("Chargement des représentations depuis le cache...")
            return z["E"]
    import torch
    from transformers import AutoModel, AutoTokenizer, logging
    logging.set_verbosity_error()
    torch.set_num_threads(os.cpu_count() or 4)
    options = {"add_prefix_space": True} if "roberta" in MODELE_LANGUE else {}
    tok = AutoTokenizer.from_pretrained(MODELE_LANGUE, **options)
    mod = AutoModel.from_pretrained(MODELE_LANGUE).eval()
    E = np.zeros((len(m), 2 * mod.config.hidden_size), dtype=np.float32)
    print(f"Lecture des {len(m)} mentions par {MODELE_LANGUE} (une seule fois, quelques minutes)...")
    with torch.no_grad():
        for i in range(0, len(m), 64):
            mots, pos = list(m["contexte_clair"].iloc[i:i + 64]), m["position"].iloc[i:i + 64].to_numpy()
            enc = tok(mots, is_split_into_words=True, truncation=True, max_length=96, padding=True, return_tensors="pt")
            h = mod(**enc).last_hidden_state.numpy()
            masque = enc["attention_mask"].numpy()
            for j in range(len(mots)):
                cible = [t for t, w in enumerate(enc.word_ids(j)) if w == pos[j]]
                E[i + j] = np.r_[h[j, cible].mean(0) if cible else h[j, 0],
                                 (h[j] * masque[j][:, None]).sum(0) / masque[j].sum()]
            if i % 6400 == 0:
                print(f"  {i}/{len(m)}", flush=True)
    os.makedirs(os.path.dirname(FICHIER_REPRESENTATIONS), exist_ok=True)
    np.savez(FICHIER_REPRESENTATIONS, E=E, signature=signature)
    return E


class SensDesMentions:
    def __init__(self, E=None):
        self.E = E

    def _X(self, m, fit=False):
        X = (self.v.fit_transform if fit else self.v.transform)([caracteristiques(r) for _, r in m.iterrows()])
        if self.E is None:
            return X
        L = self.E[m.index.to_numpy()]
        if fit:
            self.sc = StandardScaler().fit(L)
        return sp.hstack([X, sp.csr_matrix(self.sc.transform(L) * POIDS_LANGUE)]).tocsr()

    def fit(self, m, y):
        ok = ~np.isnan(y)
        self.v = DictVectorizer()
        self.lr = LogisticRegression(C=0.3, max_iter=5000).fit(self._X(m[ok], fit=True), y[ok].astype(int))
        return self

    def predict(self, m):
        return self.lr.predict_proba(self._X(m))[:, 1]

    def poids(self, n=12):
        noms = self.v.get_feature_names_out()
        w = pd.Series(self.lr.coef_[0][:len(noms)], index=noms).sort_values()
        return w.tail(n)[::-1], w.head(n)


# ----------------------------------------------------------------------------
# 3. Étape 2 : des mentions aux buts candidats, et vérification[cite: 6]
# ----------------------------------------------------------------------------
VARIABLES_CANDIDAT = ["indice", "p_max", "top3", "somme_p", "n_mentions", "n_p50", "n_familles", "score_cite",
                      "excl", "minute", "mi_temps", "debut", "ecart_prec", "indice_prec", "rang",
                      "nouveau_score", "score_coherent"]


def candidats(m, p, seuil=0.05):
    m = m.assign(p=p)
    sorties = []
    for (match, mt), g in m.groupby(["match", "mi_temps"], sort=False):
        t, q, fam = g["minute"].to_numpy(), g["p"].to_numpy(), g["famille"].to_numpy()
        a, b = np.searchsorted(t, t - 0.5), np.searchsorted(t, t + 0.5, side="right")
        cum = np.cumsum(np.r_[0, np.log1p(-np.clip(q, 0, 1 - 1e-9))])
        indice = 1 - np.exp(cum[b] - cum[a])
        pris = []
        for k in np.argsort(-indice, kind="stable"):
            if indice[k] < seuil:
                break
            if any(abs(t[k] - u) < REGROUPEMENT for u in pris):
                continue
            pris.append(t[k])
            w0, w1 = np.searchsorted(t, t[k] - 0.5), np.searchsorted(t, t[k] + 2, side="right")
            dom, ext = g["cote_dom"].iloc[a[k]:b[k]].sum(), g["cote_ext"].iloc[a[k]:b[k]].sum()

            # Correction CSC : inversion de l'équipe si own goal détecté dans la rafale
            texte_rafale = " ".join(g["texte"].iloc[w0:w1].astype(str))
            if g["est_csc"].iloc[w0:w1].any() or RE_CSC.search(texte_rafale):
                dom, ext = ext, dom

            sorties.append({
                "match": match, "mi_temps": mt, "minute": t[k], "debut_rafale": t[w0],
                "indice": indice[k], "p_max": q[a[k]:b[k]].max(), "somme_p": q[w0:w1].sum(),
                "top3": np.sort(q[w0:w1])[-3:].mean(),
                "n_mentions": w1 - w0, "n_p50": int((q[w0:w1] >= 0.5).sum()), "n_familles": len(set(fam[w0:w1])),
                "score_cite": int((fam[w0:w1] == "score cité").sum()), "excl": int(g["seg_excl"].iloc[w0:w1].sum()),
                "debut": float(t[k] < 1.5),
                "p_dom": (dom + 1) / (dom + ext + 2), "texte": g["texte"].iloc[k],
            })
    c = pd.DataFrame(sorties, columns=["match", "mi_temps", "minute", "debut_rafale", "indice", "p_max", "top3",
                                       "somme_p", "n_mentions", "n_p50", "n_familles", "score_cite", "excl", "debut",
                                       "p_dom", "texte"])
    c = c.sort_values(["match", "mi_temps", "minute"]).reset_index(drop=True)
    gr = c.groupby(["match", "mi_temps"])
    c["ecart_prec"] = gr["minute"].diff().fillna(10).clip(upper=10)
    c["indice_prec"] = gr["indice"].shift().fillna(0)
    c["rang"] = c.groupby("match")["indice"].rank(ascending=False)
    c["equipe"] = np.where(c["p_dom"] >= 0.5, "dom", "ext")
    c["nouveau_score"], c["score_coherent"] = indices_scores_cites(c, m)
    return c


def indices_scores_cites(c, m):
    s = m[m["famille"] == "score cité"]
    s = s.assign(t=(s["mi_temps"] - 1) * 100 + s["minute"], total=s["a"] + s["b"])
    par_match = {k: g for k, g in s.groupby("match")}
    nouveau, coherent, surs = [], [], {}
    for r in c.itertuples():
        t = (r.mi_temps - 1) * 100 + r.minute
        g = par_match.get(r.match, s.iloc[:0])
        rafale = g["total"][(g["t"] >= t - 0.5) & (g["t"] <= t + 2)]
        avant = g["total"][(g["t"] < t - 0.5) & (g["t"] >= t - 20)]
        dernier = avant.iloc[-1] if len(avant) else 0
        nouveau.append(int(np.clip(rafale.max() - dernier, -3, 3)) if len(rafale) else 0)
        coherent.append(int((rafale == surs.get(r.match, 0) + 1).any()))
        if r.indice >= 0.9:
            surs[r.match] = surs.get(r.match, 0) + 1
    return nouveau, coherent


def apparier(detectes, buts, matchs):
    vp = equipe_ok = n_det = n_vrais = 0
    corrects = []
    det = {k: g for k, g in detectes.groupby(["match", "mi_temps"])}
    vrais = {k: g for k, g in buts.groupby(["match", "mi_temps"])}
    for match in matchs:
        for mt in (1, 2):
            d = det.get((match, mt), detectes.iloc[:0])
            b = vrais.get((match, mt), buts.iloc[:0])
            n_det += len(d)
            n_vrais += len(b)
            dm, de, libres = d["minute"].to_numpy(), d["equipe"].to_numpy(), list(range(len(d)))
            for minute, equipe in zip(b["minute"], b["equipe"]):
                cands = [k for k in libres if TOLERANCE[0] <= dm[k] - minute <= TOLERANCE[1]]
                if cands:
                    k = min(cands, key=lambda k: abs(dm[k] - minute))
                    libres.remove(k)
                    corrects.append(d.index[k])
                    vp += 1
                    equipe_ok += de[k] == equipe
    rappel, precision = vp / max(n_vrais, 1), vp / max(n_det, 1)
    return {"buts_annotes": n_vrais, "detections": n_det, "corrects": vp,
            "faux_par_match": (n_det - vp) / max(len(matchs), 1), "rappel": rappel, "precision": precision,
            "f1": 2 * rappel * precision / max(rappel + precision, 1e-9),
            "bonne_equipe": equipe_ok / max(vp, 1)}, corrects


class DetecteurDeButs:
    def __init__(self, E=None):
        self.E = E

    def fit(self, m, buts):
        matchs = np.array(sorted(m["match"].unique()))
        y = etiqueter(m, buts)
        self.sens = SensDesMentions(self.E).fit(m, y)
        p = np.zeros(len(m))
        for i_tr, i_te in GroupKFold(4).split(matchs, groups=matchs):
            te = m["match"].isin(set(matchs[i_te])).to_numpy()
            p[te] = SensDesMentions(self.E).fit(m[~te], y[~te]).predict(m[te])
        c = candidats(m, p)
        cible = np.zeros(len(c))
        cible[apparier(c, buts, matchs)[1]] = 1
        self.verif = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300,
                                                    min_samples_leaf=30, random_state=SEED)
        self.verif.fit(c[VARIABLES_CANDIDAT].astype(float), cible)
        self.fiabilite = FiabiliteScoresCites().fit(m, buts)
        return self

    def predict(self, m):
        p = self.sens.predict(m)
        c = candidats(m, p)
        c["q"] = self.verif.predict_proba(c[VARIABLES_CANDIDAT].astype(float))[:, 1] if len(c) else []
        return p, c, self.fiabilite.predict(m[m["famille"] == "score cité"])


# ----------------------------------------------------------------------------
# 4. Score du match : buts candidats + scores cités, lus dans l'ordre[cite: 6]
# ----------------------------------------------------------------------------
class FiabiliteScoresCites:
    @staticmethod
    def classes(s, buts):
        y = np.zeros(len(s), dtype=int)
        bm = {k: g for k, g in buts.groupby("match")}
        for i, r in enumerate(s.itertuples()):
            b = bm.get(r.match, buts.iloc[:0])
            avant = (b["mi_temps"] < r.mi_temps) | ((b["mi_temps"] == r.mi_temps) & (b["minute"] <= r.minute))
            sd, se = int((b["equipe"][avant] == "dom").sum()), int((b["equipe"][avant] == "ext").sum())
            y[i] = 1 if (r.a, r.b) == (sd, se) else (2 if (r.b, r.a) == (sd, se) else 0)
        return y

    @staticmethod
    def _f(r):
        f = caracteristiques(r)
        proches = set(r["gauche"][:4] + r["droite"][:4])
        f.update({
            "egalite": float(r["a"] == r["b"]),
            "total": r["a"] + r["b"],
            "ecart": abs(r["a"] - r["b"]),
            "equipe_proche": float("<equipe>" in proches),
        })
        return f

    def fit(self, m, buts):
        s = m[m["famille"] == "score cité"]
        y = self.classes(s, buts)
        self.v = DictVectorizer()
        self.lr = LogisticRegression(C=0.3, max_iter=5000).fit(
            self.v.fit_transform([self._f(r) for _, r in s.iterrows()]), y)
        bruit = np.ones((10, 10))
        np.add.at(bruit, (s["a"].to_numpy(int)[y == 0], s["b"].to_numpy(int)[y == 0]), 1)
        self.loi_bruit = bruit / bruit.sum()
        self.taux = {"exact": np.mean(y == 1), "inverse": np.mean(y == 2)}
        return self

    def predict(self, s):
        P = np.zeros((len(s), 3))
        if len(s):
            P[:, self.lr.classes_] = self.lr.predict_proba(self.v.transform([self._f(r) for _, r in s.iterrows()]))
        return s.assign(a=s["a"].astype(int), b=s["b"].astype(int), p_bruit=P[:, 0], p_exact=P[:, 1],
                        p_inverse=P[:, 2], f_bruit=self.loi_bruit[s["a"].astype(int), s["b"].astype(int)])


def decoder_score(c, cites):
    n = MAX_BUTS + 1
    P = np.zeros((n, n))
    P[0, 0] = 1.0
    for genre, r in _evenements(c, cites):
        if genre == 0:
            P = (1 - r.q) * P + r.q * r.p_dom * _decaler(P, 0) + r.q * (1 - r.p_dom) * _decaler(P, 1)
        else:
            P = P * _vraisemblance_cite(r, n)
            P /= P.sum()
    return P


def _evenements(c, cites):
    ev = [(r.mi_temps, r.debut_rafale, 0, r) for r in c.itertuples()]
    ev += [(r.mi_temps, r.minute, 1, r) for r in cites.itertuples()]
    return [(genre, r) for _, _, genre, r in sorted(ev, key=lambda e: (e[0], e[1], e[2]))]


def _decaler(P, cote):
    S = np.zeros_like(P)
    if cote == 0:
        S[1:, :] = P[:-1, :]
    else:
        S[:, 1:] = P[:, :-1]
    return S


def _remonter(B, cote):
    S = np.zeros_like(B)
    if cote == 0:
        S[:-1, :] = B[1:, :]
    else:
        S[:, :-1] = B[:, 1:]
    return S


def _vraisemblance_cite(r, n):
    L = np.full((n, n), r.p_bruit * r.f_bruit)
    L[r.a, r.b] += r.p_exact
    if r.a != r.b:
        L[r.b, r.a] += r.p_inverse
    return L ** POIDS_SCORE_CITE


def expliquer_score(c, cites, final):
    n = MAX_BUTS + 1
    ev = _evenements(c, cites)
    alpha = [np.zeros((n, n))]
    alpha[0][0, 0] = 1.0
    delta, retours = alpha[0].copy(), []
    for genre, r in ev:
        A = alpha[-1]
        if genre == 0:
            pd_, pe = r.q * r.p_dom, r.q * (1 - r.p_dom)
            A = (1 - r.q) * A + pd_ * _decaler(A, 0) + pe * _decaler(A, 1)
            options = np.stack([(1 - r.q) * delta, pd_ * _decaler(delta, 0), pe * _decaler(delta, 1)])
            retours.append(options.argmax(0))
            delta = options.max(0)
        else:
            L = _vraisemblance_cite(r, n)
            A, delta = A * L, delta * L
            retours.append(None)
        alpha.append(A / A.sum())
        delta = delta / delta.max()
    beta = [None] * len(ev) + [np.ones((n, n))]
    for e in range(len(ev) - 1, -1, -1):
        genre, r = ev[e]
        B = beta[e + 1]
        if genre == 0:
            B = (1 - r.q) * B + r.q * r.p_dom * _remonter(B, 0) + r.q * (1 - r.p_dom) * _remonter(B, 1)
        else:
            B = _vraisemblance_cite(r, n) * B
        beta[e] = B / B.max()
    etat, buts = (int(final[0]), int(final[1])), []
    for e in range(len(ev) - 1, -1, -1):
        if retours[e] is None:
            continue
        choix = int(retours[e][etat])
        if choix:
            buts.append((e, choix - 1, etat))
            etat = (etat[0] - 1, etat[1]) if choix == 1 else (etat[0], etat[1] - 1)
    buts.reverse()
    sorties = []
    for k, (e, cote, apres) in enumerate(buts):
        r, A, B = ev[e][1], alpha[e], beta[e + 1]
        pd_, pe = r.q * r.p_dom, r.q * (1 - r.p_dom)
        total = (A * ((1 - r.q) * B + pd_ * _remonter(B, 0) + pe * _remonter(B, 1))).sum()
        proba = (A * (pd_ if cote == 0 else pe) * _remonter(B, cote)).sum() / total
        fin = buts[k + 1][0] if k + 1 < len(buts) else len(ev)
        cite = next((s for g, s in ev[e + 1:fin] if g == 1 and (s.a, s.b) == apres and s.p_exact >= 0.4), None)
        sorties.append({"candidat": r, "equipe": "dom" if cote == 0 else "ext", "proba": float(proba), "cite": cite})
    return sorties


def resume_score(P):
    i, j = np.indices(P.shape)
    proba = np.array([P[i > j].sum(), P[i == j].sum(), P[i < j].sum()])
    return np.unravel_index(P.argmax(), P.shape), proba


def evaluer_scores(predits, infos, matchs):
    p = predits.reindex(list(matchs)).fillna(0)
    v = infos.loc[list(matchs)]
    exact = (p["buts_dom"].to_numpy() == v["buts_dom"].to_numpy()) & (p["buts_ext"].to_numpy() == v["buts_ext"].to_numpy())
    total = (p["buts_dom"] + p["buts_ext"]).to_numpy() == (v["buts_dom"] + v["buts_ext"]).to_numpy()
    res = np.sign(p["buts_dom"] - p["buts_ext"]).to_numpy() == v["resultat"].to_numpy()
    ecart = np.abs((p["buts_dom"] + p["buts_ext"]).to_numpy() - (v["buts_dom"] + v["buts_ext"]).to_numpy())
    return {"matchs": len(matchs), "score_exact": exact.mean(), "nb_buts_exact": total.mean(),
            "nb_buts_a_1_pres": (ecart <= 1).mean(), "bon_resultat": res.mean()}


def compter(detectes):
    c = detectes.groupby(["match", "equipe"]).size().unstack(fill_value=0)
    return c.reindex(columns=["dom", "ext"], fill_value=0).rename(columns={"dom": "buts_dom", "ext": "buts_ext"})


# ----------------------------------------------------------------------------
# 5. Évaluation complète[cite: 6]
# ----------------------------------------------------------------------------
def main():
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 150)
    os.makedirs(DOSSIER_RESULTATS, exist_ok=True)

    print("Chargement et extraction des mentions...")
    com = donnees.charger_commentaires()
    infos = donnees.infos_matchs(com["match"])
    buts, annotes = donnees.charger_buts()
    compos = donnees.charger_compositions(infos)

    annotes = np.array(sorted(set(infos.index) & set(annotes)))
    inedits = sorted(set(infos.index) - set(annotes))

    if len(annotes) == 0:
        raise ValueError("Aucun match annoté trouvé : impossible de poursuivre la validation.")

    m = extraire_mentions(com, compos)
    ann = m["match"].isin(set(annotes)).to_numpy()
    y = np.where(ann, etiqueter(m, buts), np.nan)

    if os.path.exists(CACHE_CV):
        print("Chargement des résultats de validation croisée depuis le cache...")
        with open(CACHE_CV, "rb") as f:
            p, cand, cites, final = pickle.load(f)
    else:
        E = representations(m)
        p = np.full(len(m), np.nan)
        cand, cites = [], []
        plis = list(GroupKFold(N_PLIS).split(annotes, groups=annotes))
        for k, (i_tr, i_te) in enumerate(plis):
            print(f"  pli {k + 1}/{N_PLIS}...")
            tr, te = set(annotes[i_tr]), m["match"].isin(set(annotes[i_te])).to_numpy()
            det = DetecteurDeButs(E).fit(m[m["match"].isin(tr)], buts[buts["match"].isin(tr)])
            p[te], c, sc = det.predict(m[te])
            cand.append(c)
            cites.append(sc)
        print("  matchs jamais annotés...")
        final = DetecteurDeButs(E).fit(m[ann], buts[buts["match"].isin(set(annotes))])
        p[~ann], c, sc = final.predict(m[~ann])
        cand.append(c)
        cites.append(sc)
        cand = pd.concat(cand, ignore_index=True)
        cites = pd.concat(cites, ignore_index=True)

        os.makedirs(os.path.dirname(CACHE_CV), exist_ok=True)
        with open(CACHE_CV, "wb") as f:
            pickle.dump((p, cand, cites, final), f)

    m["p"] = p

    # --- Étape 1 : sens des mentions[cite: 6]
    lab = ~np.isnan(y)
    mots_cles = m["famille"].isin(FAMILLES_MOTS).to_numpy()
    lm, lt = lab & mots_cles, lab & ~mots_cles
    print(f"\n=== Étape 1 : sens de chaque mention (validation croisée) ===")
    print(f"Le modèle sépare annonces et autres sens avec une AUC de {roc_auc_score(y[lm], p[lm]):.2f} sur les mots-clés "
          f"(règles écrites à la main : {roc_auc_score(y[lm], m['regle'].isna()[lm]):.2f} ; 0,5 = hasard)")
    print(f"Tournures sans mot-clé ({lt.sum()} mentions, {np.mean(y[lt]):.0%} juste après un vrai but) : "
          f"AUC {roc_auc_score(y[lt], p[lt]):.2f}")
    plus, moins = final.sens.poids()
    print("Indices d'une annonce :", ", ".join(plus.index))
    print("Indices du contraire  :", ", ".join(moins.index))

    # --- Étape 2 : buts détectés[cite: 6]
    approches = {
        "naïve (tout « goal » est un but)": candidats(m, mots_cles.astype(float), 0.5),
        "règles écrites à la main": candidats(m, (mots_cles & m["regle"].isna().to_numpy()).astype(float), 0.5),
        "modèle, étape 1 seule": cand[cand["indice"] >= 0.5],
        "modèle, étapes 1 + 2": cand[cand["q"] >= 0.5],
    }
    print(f"\n=== Buts détectés contre buts annotés ({len(annotes)} matchs, validation croisée) ===")
    lignes = []
    for nom, d in approches.items():
        lignes.append({"approche": nom, **apparier(d, buts, annotes)[0]})
    print(pd.DataFrame(lignes).set_index("approche").round(2).to_string())

    # --- Score du match[cite: 6]
    sc = cites.assign(y=FiabiliteScoresCites.classes(cites, buts))
    sc = sc[sc["match"].isin(set(annotes))]
    print(f"\n=== Scores cités ({len(sc)} dans les matchs annotés) ===")
    print(f"Score du moment : {(sc['y'] == 1).mean():.0%}, inversé : {(sc['y'] == 2).mean():.0%}, "
          f"bruit : {(sc['y'] == 0).mean():.0%}. Le modèle repère les citations justes avec une AUC de "
          f"{roc_auc_score(sc['y'] > 0, sc['p_exact'] + sc['p_inverse']):.2f}")
    dec = {}
    par_match = {x: g for x, g in cites.groupby("match")}
    for match, g in cand.groupby("match"):
        P = decoder_score(g, par_match.get(match, cites.iloc[:0]))
        (bd, be), proba = resume_score(P)
        dec[match] = {"buts_dom": bd, "buts_ext": be, "p_dom": proba[0], "p_nul": proba[1], "p_ext": proba[2]}
    dec = pd.DataFrame.from_dict(dec, orient="index")
    print(f"\n=== Score final reconstitué ===")
    lignes = []
    for nom, d in approches.items():
        lignes.append({"approche": nom, "matchs": "annotés", **evaluer_scores(compter(d), infos, annotes)})
    lignes.append({"approche": "modèle + scores cités", "matchs": "annotés", **evaluer_scores(dec, infos, annotes)})
    lignes.append({"approche": "modèle + scores cités", "matchs": "jamais annotés",
                   **evaluer_scores(dec, infos, inedits)})
    print(pd.DataFrame(lignes).round(2).to_string(index=False))

    # --- Fichiers pour regarder les erreurs[cite: 6]
    d = cand[cand["q"] >= 0.5].merge(infos[["dom", "ext"]], left_on="match", right_index=True)
    d["equipe"] = np.where(d["equipe"] == "dom", d["dom"], d["ext"])
    d["minute_match"] = (d["minute"] + 45 * (d["mi_temps"] - 1)).round(1)
    corrects = set(apparier(cand[cand["q"] >= 0.5], buts, annotes)[1])
    d["verdict"] = np.where(~d["match"].isin(set(annotes)), "match non annoté",
                            np.where(d.index.isin(corrects), "vrai but", "faux positif"))
    d[["match", "minute_match", "equipe", "q", "verdict", "texte"]].to_csv(
        os.path.join(DOSSIER_RESULTATS, "buts_detectes.csv"), index=False, encoding="utf-8-sig")
    dec.join(infos[["dom", "ext", "buts_dom", "buts_ext"]], rsuffix="_reel").round(3).to_csv(
        os.path.join(DOSSIER_RESULTATS, "scores_reconstitues.csv"), encoding="utf-8-sig")
    m.assign(y=y)[["match", "mi_temps", "minute", "mot", "regle", "p", "y", "texte"]].round(3).to_csv(
        os.path.join(DOSSIER_RESULTATS, "mentions.csv"), index=False, encoding="utf-8-sig")
    print(f"\nDétails : {DOSSIER_RESULTATS}/buts_detectes.csv, scores_reconstitues.csv, mentions.csv")


if __name__ == "__main__":
    main()