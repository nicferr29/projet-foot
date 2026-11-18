"""
Chargement des données du projet. Aucun modèle ici, seulement de la lecture et du nettoyage.

- echoes_brut.csv : commentaires SoccerNet-Echoes (Whisper v2, traduits en anglais), un segment par ligne.
  start_time est en secondes depuis le coup d'envoi de la mi-temps (les vidéos SoccerNet commencent au
  coup d'envoi).
- SoccerNet/<championnat>/<saison>/<match>/Labels-v2.json : annotations officielles, dont chaque but
  (mi-temps, minute, équipe). C'est la vérité terrain.
- SoccerNet/.../Labels-caption.json : compositions officielles (connues avant le match). On n'en garde que
  les noms des joueurs, le gardien titulaire et les noms de l'équipe. Le champ « facts » (buts, cartons) est
  ignoré : il révélerait le déroulement du match.
"""
import glob
import json
import os
import re
import unicodedata

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

ICI = os.path.dirname(os.path.abspath(__file__))
FICHIER_COMMENTAIRES = os.path.join(ICI, "echoes_brut.csv")
DOSSIER_SOCCERNET = os.path.join(ICI, "SoccerNet")
PATTERN_MATCH = r"^(\d{4}-\d{2}-\d{2}) - (\d{2}-\d{2}) (.+?) (\d+) - (\d+) (.+)$"

# Mots trop courants pour servir de nom propre (« Young », « Can », « Son »...)
MOTS_COMMUNS = set(ENGLISH_STOP_WORDS) | {
    "ball", "goal", "game", "team", "match", "area", "time", "real", "club", "city", "united", "athletic",
    "sporting", "borussia", "olympique", "stade", "inter", "atl", "san", "saint", "santa", "son", "can",
    "will", "young", "long", "good", "king", "rose", "costa", "white", "black", "green", "blue", "red",
    "free", "kick", "left", "right", "half", "line", "post", "box", "shot", "head", "corner",
}


# Une même équipe porte parfois deux noms dans les dossiers SoccerNet : sans ce regroupement, « Bayern » et
# « Bayern Munich » auraient chacune leur Elo et leur effectif.
ALIAS = {"Bayern": "Bayern Munich", "Frankfurt": "Eintracht Frankfurt", "Hamburger": "Hamburger SV",
         "Hertha": "Hertha Berlin", "Augsburg": "FC Augsburg", "Leipzig": "RB Leipzig",
         "Borussia": "B. Monchengladbach",  # Leipzig 1-1 Mönchengladbach, 21/09/2016
         "Atletico Madrid": "Atl. Madrid", "Granada": "Granada CF", "Saint Etienne": "St Etienne",
         "Inter": "Internazionale", "Ludogorets": "Ludogorets Razgrad"}

# Actions annotées montrées sur la page (jamais utilisées pour prédire) : code court pour alléger la page.
EVENEMENTS_AFFICHES = {"Goal": "B", "Penalty": "P", "Shots on target": "S", "Shots off target": "W", "Corner": "C",
                       "Yellow card": "J", "Red card": "R", "Yellow->red card": "Y", "Substitution": "X",
                       "Offside": "O", "Direct free-kick": "F", "Throw-in": "T", "Kick-off": "K"}


def normaliser(texte):
    """Minuscules, sans accents."""
    return unicodedata.normalize("NFKD", str(texte)).encode("ascii", "ignore").decode().lower()


def mots_du_nom(nom):
    return {t for t in re.findall(r"[a-z]+", normaliser(nom)) if len(t) >= 3 and t not in MOTS_COMMUNS}


def nom_relatif(chemin):
    return os.path.relpath(os.path.dirname(chemin), DOSSIER_SOCCERNET).replace(os.sep, "/")


def charger_commentaires():
    """Un segment par ligne : match, mi_temps, segment, minute (depuis le coup d'envoi de la mi-temps), texte.
    Les répétitions en boucle de Whisper (même phrase que l'un des 3 segments précédents) sont retirées."""
    df = pd.read_csv(FICHIER_COMMENTAIRES, encoding="utf-8-sig")
    df["match"] = df["game"].str.rsplit("/", n=1).str[0]
    df["mi_temps"] = df["game"].str.rsplit("/", n=1).str[1].astype(int)
    df["texte"] = df["text"].fillna("").astype(str).str.strip()
    df = df[df["texte"] != ""].sort_values(["match", "mi_temps", "start_time", "segment_index"])
    g = df.groupby(["match", "mi_temps"])["texte"]
    repete = np.zeros(len(df), dtype=bool)
    for k in (1, 2, 3):
        repete |= (df["texte"] == g.shift(k)).to_numpy()
    df = df[~repete]
    df = df.assign(minute=df["start_time"] / 60).rename(columns={"segment_index": "segment"})
    return df[["match", "mi_temps", "segment", "minute", "texte"]].reset_index(drop=True)


def infos_matchs(noms):
    """Date, équipes et score final, lus dans le nom du dossier (« 2015-02-21 - 18-00 Chelsea 1 - 1 Burnley »)."""
    s = pd.Series(sorted(set(noms)))
    i = s.str.split("/").str[2].str.extract(PATTERN_MATCH)
    infos = pd.DataFrame({"date": i[0].values, "dom": i[2].replace(ALIAS).values, "ext": i[5].replace(ALIAS).values,
                          "buts_dom": i[3].astype(int).values, "buts_ext": i[4].astype(int).values},
                         index=pd.Index(s.values, name="match"))
    infos["competition"] = s.str.split("/").str[0].values
    infos["resultat"] = np.sign(infos["buts_dom"] - infos["buts_ext"]).astype(int)  # 1 / 0 / -1
    return infos


def charger_buts():
    """Buts annotés : match, mi_temps, minute (depuis le coup d'envoi de la mi-temps), equipe (dom / ext).
    Renvoie aussi l'ensemble des matchs annotés (y compris ceux sans but)."""
    lignes, annotes = [], set()
    for chemin in glob.glob(os.path.join(DOSSIER_SOCCERNET, "*", "*", "*", "Labels-v2.json")):
        match = nom_relatif(chemin)
        annotes.add(match)
        with open(chemin, encoding="utf-8") as f:
            for a in json.load(f).get("annotations", []):
                if a.get("label") == "Goal":
                    mt, temps = a["gameTime"].split(" - ")
                    mm, ss = temps.split(":")
                    lignes.append((match, int(mt), int(mm) + int(ss) / 60,
                                   "dom" if a.get("team") == "home" else "ext"))
    buts = pd.DataFrame(lignes, columns=["match", "mi_temps", "minute", "equipe"])
    return buts.sort_values(["match", "mi_temps", "minute"]).reset_index(drop=True), annotes


def charger_evenements():
    """Actions annotées de chaque match (buts, tirs, corners, cartons, changements...), pour l'affichage
    seulement : match, mi_temps, minute, code (voir EVENEMENTS_AFFICHES), equipe (dom / ext / None)."""
    lignes = []
    for chemin in glob.glob(os.path.join(DOSSIER_SOCCERNET, "*", "*", "*", "Labels-v2.json")):
        match = nom_relatif(chemin)
        with open(chemin, encoding="utf-8") as f:
            for a in json.load(f).get("annotations", []):
                code = EVENEMENTS_AFFICHES.get(a.get("label"))
                if code:
                    mt, temps = a["gameTime"].split(" - ")
                    mm, ss = temps.split(":")
                    lignes.append((match, int(mt), int(mm) + int(ss) / 60, code,
                                   {"home": "dom", "away": "ext"}.get(a.get("team"))))
    ev = pd.DataFrame(lignes, columns=["match", "mi_temps", "minute", "code", "equipe"])
    return ev.sort_values(["match", "mi_temps", "minute"]).reset_index(drop=True)


def lire_composition(chemin):
    """{"dom": {"equipe", "joueurs", "gardien"}, "ext": {...}} d'un fichier Labels-caption.json."""
    with open(chemin, encoding="utf-8") as f:
        j = json.load(f)
    c = {}
    for cote, s in (("dom", "home"), ("ext", "away")):
        c[cote] = {"equipe": set().union(*[mots_du_nom(n) for n in j[s].get("names", [])]),
                   "joueurs": set(), "gardien": set()}
        for p in j["lineup"][s]["players"]:
            noms = mots_du_nom(p.get("short_name") or "") | mots_du_nom(p.get("long_name") or "")
            cle = "gardien" if p.get("starting") and p.get("lineup") == 1 else "joueurs"
            c[cote][cle] |= noms
    return c


def effectifs_par_saison(fichiers):
    """(équipe, saison) -> joueurs et gardiens vus dans les compositions de tous les matchs SoccerNet.
    Sert quand un match n'a pas de composition officielle : l'effectif d'une équipe est public avant le
    match. Le nom de l'équipe est celui du dossier SoccerNet (« Paris SG »), identique d'un match à l'autre."""
    eff = {}
    noms = infos_matchs(list(fichiers))
    for match, chemin in fichiers.items():
        c = lire_composition(chemin)
        saison = match.split("/")[1]
        for cote in ("dom", "ext"):
            e = eff.setdefault((noms.at[match, cote], saison), {"joueurs": set(), "gardien": set()})
            e["joueurs"] |= c[cote]["joueurs"]
            e["gardien"] |= c[cote]["gardien"]
    return eff


def charger_compositions(infos):
    """Pour chaque match : {"dom": {"equipe", "joueurs", "gardien"}, "ext": {...}}, ensembles de mots.
    Sans composition officielle, on prend l'effectif de l'équipe vu dans ses autres matchs de la saison ;
    à défaut, on n'a que le nom de l'équipe."""
    compos = {}
    fichiers = {nom_relatif(c): c for c in glob.glob(os.path.join(DOSSIER_SOCCERNET, "*", "*", "*",
                                                                   "Labels-caption.json"))}
    effectifs = effectifs_par_saison(fichiers)
    for match, r in infos.iterrows():
        c = {"dom": {"equipe": mots_du_nom(r["dom"]), "joueurs": set(), "gardien": set()},
             "ext": {"equipe": mots_du_nom(r["ext"]), "joueurs": set(), "gardien": set()}}
        if match in fichiers:
            officielle = lire_composition(fichiers[match])
            for cote in c:
                for cle in c[cote]:
                    c[cote][cle] |= officielle[cote][cle]
        else:
            saison = match.split("/")[1]
            for cote in ("dom", "ext"):
                e = effectifs.get((r[cote], saison))
                if e:
                    c[cote]["joueurs"] |= e["joueurs"]
                    c[cote]["gardien"] |= e["gardien"]
        # un mot présent des deux côtés (« madrid ») ne désigne personne
        tous_dom = set().union(*c["dom"].values())
        tous_ext = set().union(*c["ext"].values())
        commun = tous_dom & tous_ext
        for cote in c:
            for cle in c[cote]:
                c[cote][cle] -= commun
            c[cote]["joueurs"] -= c[cote]["gardien"]
        compos[match] = c
    return compos
