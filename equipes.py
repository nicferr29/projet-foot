"""
Identité visuelle des équipes pour la page : code de 3 lettres (comme sur un tableau d'affichage télé) et
couleurs du maillot (principale, secondaire). Sert seulement à l'affichage.
"""
import hashlib

import donnees

#  nom (après donnees.ALIAS) : (code, couleur principale, couleur secondaire)
EQUIPES = {
    # Bundesliga
    "1. FSV Mainz 05": ("M05", "#C3141E", "#FFFFFF"), "FC Augsburg": ("FCA", "#BA3733", "#FFFFFF"),
    "B. Monchengladbach": ("BMG", "#FFFFFF", "#00A651"), "Bayer Leverkusen": ("B04", "#E32221", "#1A1A1A"),
    "Bayern Munich": ("FCB", "#DC052D", "#FFFFFF"), "Darmstadt": ("D98", "#004E9F", "#FFFFFF"),
    "Dortmund": ("BVB", "#FDE100", "#1A1A1A"), "Eintracht Frankfurt": ("SGE", "#1A1A1A", "#E1000F"),
    "FC Koln": ("KOE", "#FFFFFF", "#ED1C24"), "Hamburger SV": ("HSV", "#FFFFFF", "#0A3F86"),
    "Hertha Berlin": ("BSC", "#005CA9", "#FFFFFF"), "Hoffenheim": ("TSG", "#1C63B7", "#FFFFFF"),
    "Ingolstadt": ("FCI", "#C8102E", "#1A1A1A"), "RB Leipzig": ("RBL", "#FFFFFF", "#DD0741"),
    "Paderborn": ("SCP", "#00519E", "#1A1A1A"), "SC Freiburg": ("SCF", "#E2001A", "#1A1A1A"),
    "SV Werder Bremen": ("SVW", "#1D9053", "#FFFFFF"), "Schalke": ("S04", "#004D9D", "#FFFFFF"),
    "VfB Stuttgart": ("VFB", "#FFFFFF", "#E32219"), "Wolfsburg": ("WOB", "#65B32E", "#FFFFFF"),
    # LaLiga
    "Alaves": ("ALA", "#0761AF", "#FFFFFF"), "Almeria": ("ALM", "#EE1119", "#FFFFFF"),
    "Ath Bilbao": ("ATH", "#EE2523", "#FFFFFF"), "Atl. Madrid": ("ATM", "#CB3524", "#272E61"),
    "Barcelona": ("BAR", "#A50044", "#004D98"), "Betis": ("BET", "#00954C", "#FFFFFF"),
    "Celta Vigo": ("CEL", "#8AC3EE", "#FFFFFF"), "Cordoba": ("COR", "#006A3B", "#FFFFFF"),
    "Dep. La Coruna": ("DEP", "#0067B1", "#FFFFFF"), "Eibar": ("EIB", "#A6192E", "#0C2A5B"),
    "Elche": ("ELC", "#FFFFFF", "#05642C"), "Espanyol": ("ESP", "#007FC8", "#FFFFFF"),
    "Getafe": ("GET", "#005999", "#FFFFFF"), "Gijon": ("SPO", "#E33030", "#FFFFFF"),
    "Granada CF": ("GRA", "#C31632", "#FFFFFF"), "Las Palmas": ("LPA", "#FFE667", "#0055A4"),
    "Leganes": ("LEG", "#0F4C9C", "#FFFFFF"), "Levante": ("LEV", "#B4053F", "#004F9F"),
    "Malaga": ("MAL", "#0091D5", "#FFFFFF"), "Osasuna": ("OSA", "#D91A21", "#0A346F"),
    "Rayo Vallecano": ("RAY", "#FFFFFF", "#E53027"), "Real Madrid": ("RMA", "#FFFFFF", "#00529F"),
    "Real Sociedad": ("RSO", "#0067B1", "#FFFFFF"), "Sevilla": ("SEV", "#FFFFFF", "#D81E05"),
    "Valencia": ("VAL", "#FFFFFF", "#EE3524"), "Villarreal": ("VIL", "#FFE667", "#005187"),
    # Ligue 1
    "Angers": ("ANG", "#1A1A1A", "#FFFFFF"), "Bastia": ("BAS", "#0055A4", "#FFFFFF"),
    "Bordeaux": ("BOR", "#001C50", "#FFFFFF"), "Caen": ("CAE", "#1B3B8C", "#D6213A"),
    "Dijon": ("DIJ", "#D2122E", "#FFFFFF"), "Guingamp": ("GUI", "#E30613", "#1A1A1A"),
    "Lille": ("LIL", "#E01E13", "#20325F"), "Lorient": ("LOR", "#F58113", "#1A1A1A"),
    "Lyon": ("LYO", "#FFFFFF", "#1D4E9E"), "Marseille": ("MAR", "#FFFFFF", "#2FAEE0"),
    "Metz": ("MET", "#6F1D46", "#FFFFFF"), "Monaco": ("MON", "#E51B22", "#FFFFFF"),
    "Montpellier": ("MTP", "#003B71", "#F37021"), "Nancy": ("NCY", "#FFFFFF", "#E2001A"),
    "Nantes": ("NAN", "#FCD405", "#00703C"), "Nice": ("NIC", "#C6161D", "#1A1A1A"),
    "Paris SG": ("PSG", "#004170", "#DA291C"), "Reims": ("REI", "#EE1C25", "#FFFFFF"),
    "Rennes": ("REN", "#E13327", "#1A1A1A"), "St Etienne": ("STE", "#009639", "#FFFFFF"),
    "Toulouse": ("TOU", "#593B8B", "#FFFFFF"),
    # Serie A
    "AC Milan": ("MIL", "#FB090B", "#1A1A1A"), "AS Roma": ("ROM", "#8E1F2F", "#F0BC42"),
    "Atalanta": ("ATA", "#1E71B8", "#1A1A1A"), "Bologna": ("BOL", "#A21C26", "#1A2F48"),
    "Cagliari": ("CAG", "#A00E1E", "#002350"), "Chievo": ("CHI", "#FFDD00", "#003D7C"),
    "Crotone": ("CRO", "#E10E1C", "#0E3386"), "Empoli": ("EMP", "#00579C", "#FFFFFF"),
    "Fiorentina": ("FIO", "#482E92", "#FFFFFF"), "Frosinone": ("FRO", "#FEDD00", "#0047AB"),
    "Genoa": ("GEN", "#A41E28", "#002147"), "Internazionale": ("INT", "#0068A8", "#1A1A1A"),
    "Juventus": ("JUV", "#1A1A1A", "#FFFFFF"), "Lazio": ("LAZ", "#87D8F7", "#FFFFFF"),
    "Napoli": ("NAP", "#12A0D7", "#FFFFFF"), "Palermo": ("PAL", "#F0B4C8", "#1A1A1A"),
    "Pescara": ("PES", "#3398D6", "#FFFFFF"), "Sampdoria": ("SAM", "#1B5497", "#FFFFFF"),
    "Sassuolo": ("SAS", "#00A752", "#1A1A1A"), "Torino": ("TOR", "#8A1E03", "#FFFFFF"),
    "Udinese": ("UDI", "#FFFFFF", "#1A1A1A"), "Verona": ("VER", "#FFD700", "#003F8C"),
    # Premier League
    "Arsenal": ("ARS", "#EF0107", "#FFFFFF"), "Aston Villa": ("AVL", "#670E36", "#95BFE5"),
    "Bournemouth": ("BOU", "#DA291C", "#1A1A1A"), "Burnley": ("BUR", "#6C1D45", "#99D6EA"),
    "Chelsea": ("CHE", "#034694", "#FFFFFF"), "Crystal Palace": ("CRY", "#1B458F", "#C4122E"),
    "Everton": ("EVE", "#003399", "#FFFFFF"), "Hull City": ("HUL", "#F5A12D", "#1A1A1A"),
    "Leicester": ("LEI", "#003090", "#FFFFFF"), "Liverpool": ("LIV", "#C8102E", "#FFFFFF"),
    "Manchester City": ("MCI", "#6CABDD", "#FFFFFF"), "Manchester United": ("MUN", "#DA291C", "#FBE122"),
    "Middlesbrough": ("MID", "#E11B22", "#FFFFFF"), "Newcastle Utd": ("NEW", "#241F20", "#FFFFFF"),
    "Norwich": ("NOR", "#FFF200", "#00A650"), "Southampton": ("SOU", "#D71920", "#FFFFFF"),
    "Stoke City": ("STK", "#E03A3E", "#FFFFFF"), "Sunderland": ("SUN", "#EB172B", "#FFFFFF"),
    "Swansea": ("SWA", "#FFFFFF", "#1A1A1A"), "Tottenham": ("TOT", "#FFFFFF", "#132257"),
    "Watford": ("WAT", "#FBEE23", "#1A1A1A"), "West Brom": ("WBA", "#122F67", "#FFFFFF"),
    "West Ham": ("WHU", "#7A263A", "#1BB1E7"),
    # Coupes d'Europe
    "Ajax": ("AJA", "#FFFFFF", "#D2122E"), "Anderlecht": ("AND", "#4E2A84", "#FFFFFF"),
    "BATE": ("BAT", "#FFD700", "#1E4E9D"), "Basel": ("BSL", "#E0001B", "#003D7C"),
    "Benfica": ("BEN", "#E83030", "#FFFFFF"), "Besiktas": ("BJK", "#1A1A1A", "#FFFFFF"),
    "CSKA Moscow": ("CSK", "#E30613", "#003F87"), "Celtic": ("CLT", "#018749", "#FFFFFF"),
    "D. Zagreb": ("DIN", "#0A50A1", "#FFFFFF"), "Dyn. Kiev": ("DYN", "#FFFFFF", "#00529F"),
    "FC Astana": ("AST", "#00AEEF", "#FFD200"), "FC Porto": ("POR", "#003893", "#FFFFFF"),
    "Galatasaray": ("GAL", "#A90432", "#FDB912"), "Gent": ("GNT", "#0055A4", "#FFFFFF"),
    "Ludogorets Razgrad": ("LUD", "#00843D", "#FFFFFF"), "Maccabi Tel Aviv": ("MTA", "#FFE600", "#003C82"),
    "Malmo FF": ("MFF", "#87C3E8", "#FFFFFF"), "Olympiakos Piraeus": ("OLY", "#E30613", "#FFFFFF"),
    "PSV": ("PSV", "#ED1C24", "#FFFFFF"), "Shakhtar Donetsk": ("SHA", "#F47920", "#1A1A1A"),
    "Zenit Petersburg": ("ZEN", "#009BDE", "#FFFFFF"),
}

# Nom affiché quand celui des dossiers SoccerNet est peu lisible
NOM_AFFICHE = {"Internazionale": "Inter", "B. Monchengladbach": "M'gladbach", "Atl. Madrid": "Atlético Madrid",
               "Dep. La Coruna": "La Corogne", "Dyn. Kiev": "Dynamo Kiev", "D. Zagreb": "Dinamo Zagreb",
               "St Etienne": "Saint-Étienne", "Ath Bilbao": "Athletic Bilbao", "Gijon": "Sporting Gijón"}

_PALETTE = ["#D7263D", "#1B998B", "#2E86AB", "#F46036", "#5B2A86", "#E2C044", "#3A7D44", "#C2185B"]


def identite(nom):
    """{"nom", "code", "c1", "c2"} pour une équipe ; une équipe inconnue reçoit un code et une couleur stables."""
    nom = donnees.ALIAS.get(nom, nom)
    if nom in EQUIPES:
        code, c1, c2 = EQUIPES[nom]
    else:
        mots = sorted(nom.replace(".", " ").split(), key=len, reverse=True)
        code = donnees.normaliser(mots[0])[:3].upper() if mots else "???"
        h = int(hashlib.md5(nom.encode()).hexdigest(), 16)
        c1, c2 = _PALETTE[h % len(_PALETTE)], "#FFFFFF"
    return {"nom": NOM_AFFICHE.get(nom, nom), "code": code, "c1": c1, "c2": c2}
