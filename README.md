# Lire un match de foot dans ses commentaires

Projet de NLP sur les commentaires de 367 matchs ([SoccerNet-Echoes](https://huggingface.co/datasets/SoccerNet/SN-echoes),
transcriptions Whisper traduites en anglais) : détecter les buts, puis prédire le match « en direct », minute
après minute, comme si on le regardait à la télé.

**Le piège** : « goal » veut dire à la fois *but marqué* (« Goal! Goal for Real Madrid! ») et *cage*
(« the ball goes wide of Courtois's goal », « shot on goal », « goal kick »). Le mot apparaît 43 fois par match
pour 3,4 vrais buts : prendre chaque « goal » pour un but donne 16 faux buts par match.

## Lancer

```bash
conda activate nlp-foot
pip install -r requirements.txt
python detection_buts.py      # détection des buts et évaluation (~10 min la 1re fois, modèle de langue)
python direct.py              # prédictions minute par minute + page match_en_direct.html (~5 min)
open match_en_direct.html
```

`python direct.py --page` reconstruit seulement la page (après une retouche de `gabarit_page.html`).

## La page `match_en_direct.html`

- **L'habillage** : un soir de match télévisé. Stade de nuit et projecteurs, typographie condensée façon
  habillage télé, couleurs des deux équipes. En haut, trois gros onglets : compétition, match, « Match au hasard ».
- **Le stade** : un terrain vu d'en haut où 22 petits joueurs rejouent le match. Ils suivent les actions annotées
  par SoccerNet au bon moment (buts, tirs, corners, touches, penalties, cartons). « BUT ! » s'affiche quand le
  ballon entre, l'arbitre sort les cartons, un joueur exclu quitte le terrain. Les positions sont inventées :
  c'est une illustration. Autour du terrain :
  - derrière chaque but, un tifo, une mosaïque de cartons aux couleurs du club avec son nom (domicile à gauche,
    visiteurs à droite) ;
  - le long des lignes de touche, la foule en lumières floues aux couleurs des clubs, avec des flashs de téléphones ;
  - des panneaux LED qui défilent le long des lignes de touche.

  À chaque but, chez l'équipe qui marque : une vague traverse le tifo, les fumigènes s'allument, la foule
  s'illumine, les flashs crépitent et les panneaux affichent « BUT ! ».
- **Le tableau d'affichage**, façon télé : codes des équipes, couleurs des maillots, chrono qui tourne, temps
  additionnel, cartons rouges. Le gros score est **celui que le modèle lit** dans les commentaires (il clignote
  quand le modèle change d'avis) ; le vrai score est en petit dessous. Les événements (but, carton, changement)
  s'affichent quelques secondes.
- **Le bandeau** du bas de l'écran : à chaque point de contrôle où le modèle change d'avis, le score qu'il lit et
  la phrase qui l'a convaincu.
- **Sous l'écran**, un panneau LED : la frise 0' · 10' · … · 90' · Fin, ▶ pour regarder le match (vitesse ×1, ×2,
  ×4) et les flèches du clavier. Pour un match sans annotation, le terrain rejoue le match tel que le modèle le lit.
- **Le panneau d'analyse** :
  - l'évolution des chances de victoire, avec les vrais buts marqués sur la courbe ;
  - le pronostic du score final contre le vrai score ;
  - deux onglets. « Lu dans les commentaires » donne, pour chaque but compté par le modèle, la phrase de
    l'annonce et le score cité qui l'a confirmé (« Bayern are leading 1-0 »). « Le vrai match » donne les vrais
    buts, cartons et changements.

- **La page 2** (le ballon « 2 » sur le côté de l'écran, ou la touche 2) : le projet expliqué. On y trouve la
  problématique, les données, ce que fait le programme, les modèles de langue (BERT), les **méthodes** (pourquoi
  chacune, et comment elle marche), les résultats et les limites.

Fichier autonome : il s'ouvre sans connexion. On peut viser un match précis :
`match_en_direct.html#<dossier du match>`, ou la page 2 avec `match_en_direct.html#projet`.

## Données

- `echoes_brut.csv` : les commentaires (un segment par ligne, minute depuis le coup d'envoi de la mi-temps)
- `SoccerNet/*/Labels-v2.json` : buts officiels (minute, équipe) = vérité terrain. Les autres actions annotées
  (tirs, corners, cartons, changements) ne servent qu'à animer la page, jamais à prédire.
- Les équipes qui portent deux noms selon les dossiers (« Bayern » / « Bayern Munich », « Inter » /
  « Internazionale »…) sont regroupées (`donnees.ALIAS`). Sans cela, l'Elo et les effectifs étaient coupés en deux.
- `SoccerNet/*/Labels-caption.json` : compositions officielles (joueurs, gardien titulaire) ; le champ `facts`
  (buts, cartons) n'est jamais lu. Sans composition, on prend l'effectif de l'équipe vu dans ses autres matchs
  de la saison.

## Méthode

`detection_buts.py` :

1. **Mentions** : chaque mot qui peut annoncer un but (goal, scores, nets, equaliser, score cité « 1-0 »), et
   des tournures qui trahissent un but sans ces mots : « the ball is in », « in off the post », « into the far
   corner », « scorer », « takes the lead », « goes crazy »… Exemple : « And the ball is in. Olivier Giroud.
   77th minute. Now [the stadium] is going crazy. » n'a aucun mot-clé. Chaque tournure n'a été retenue que si
   elle tombe bien plus souvent que le hasard dans les 3 min qui suivent un vrai but (fête 48 %, contre 11 %
   au hasard).
2. **Entités** : les noms propres deviennent `<gardien>`, `<joueur>`, `<equipe>` ou `<autre_equipe>`.
   « Courtois's goal » devient « `<gardien>` 's goal » : la cage d'un gardien, quel que soit le match.
3. **Étape 1, sens de la mention** : régression logistique sur le contexte (mots typés à gauche et à droite,
   ponctuation, densité de mentions) **et** sur la lecture du passage par un modèle de langue pré-entraîné
   (DistilRoBERTa, famille BERT), à qui l'on donne le texte avec les rôles en clair (« goalkeeper 's goal »).
   Supervision à distance : une mention juste après un vrai but est une annonce, une mention loin de tout but
   n'en est pas une.
4. **Étape 2, vérification** : les annonces proches forment un but candidat ; un second modèle juge la rafale
   entière (annonces répétées, exclamations, score cité nouveau et cohérent avec les buts déjà comptés).
5. **Fiabilité des scores cités** : 47 % des « 2-1 » prononcés ne sont pas le score du moment (schémas
   tactiques, autres matchs, souvenirs) ; un classifieur juge chaque citation d'après ses mots voisins.
6. **Score du match** : buts candidats + scores cités, lus dans l'ordre.
7. **Explication** : le chemin le plus probable (Viterbi) qui mène de 0-0 au score lu dit quels candidats sont
   des buts. Chaque but reçoit sa probabilité a posteriori : ce qui est dit après compte aussi, par exemple un
   « 3-0 » prononcé après une annonce confuse. On garde aussi le score cité qui le confirme. C'est ce
   qu'affiche la page.

`direct.py` : à chaque point de contrôle, on ne garde que les commentaires déjà prononcés, on estime le score
courant (étapes 1 à 6), puis on projette la fin du match : buts restants ~ Poisson(buts attendus × part du temps
restant). Les buts attendus viennent de la force des équipes avant le match, estimée par un **filtre de Kalman**
sur les seuls matchs antérieurs : chaque équipe a une force (en buts par match) et une incertitude ; après chaque
match, elle corrige sa force d'une part de la surprise (écart de buts observé − attendu), d'autant plus grande
qu'on la connaît mal. Un Elo classique fait la même correction avec un gain fixe pour tout le monde.

Tout est évalué en validation croisée par match : aucun match n'est prédit par un modèle qui l'a vu.

## Résultats (330 matchs annotés, validation croisée)

**Détection des buts**

| Approche | Faux buts / match | Précision | Rappel |
|---|---|---|---|
| Naïve (tout « goal » est un but) | 15,8 | 16 % | 88 % |
| Règles écrites à la main | 14,0 | 18 % | 87 % |
| Modèle sans modèle de langue¹ | 0,48 | 79 % | 53 % |
| Modèle avec DistilRoBERTa | 0,45 | 81 % | 57 % |
| **Modèle avec DistilRoBERTa et tournures** | **0,45** | **82 %** | **58 %** |

¹ mesuré avant le regroupement des noms d'équipes.

- Sens des mentions : AUC 0,86 → 0,89 avec le modèle de langue (règles à la main : 0,55). Les 1 900 mentions
  de tournures sont triées avec une AUC de 0,88.
- Score final lu dans tout le commentaire : exact 62 %, à un but près 88 %, bon vainqueur 78 %. Sur les
  37 matchs jamais annotés : exact 46 %, bon vainqueur 70 %.
- « `<gardien>`'s goal » : probabilité d'annonce moyenne 0,05 (5 % sont de vrais buts) ; « `<joueur>`'s goal » :
  0,26 (28 %).

**Prédiction en direct** (`direct.py`)

| Moment | Bon vainqueur | Log loss | Score final exact |
|---|---|---|---|
| Coup d'envoi (force des équipes seule) | 68,5 % | 0,81 | 9 % |
| Mi-temps | 68,8 % | 0,71 | 16 % |
| 70' | 67,6 % | 0,68 | 22 % |
| 90' | 76,4 % | 0,59 | 51 % |
| Fin | 80,3 % | 0,59 | 62 % |

Repère : prédire toujours « victoire à domicile » donne 49 %. La log loss (justesse des probabilités) baisse
à chaque étape ; le pourcentage de bons vainqueurs oscille de ± 3 points d'un moment à l'autre (bruit dû aux
330 matchs), mais progresse. Sur les 37 matchs jamais annotés (test à part) : 73 % au coup d'envoi, 76 % à la fin.

**Force des équipes au coup d'envoi : filtre de Kalman contre Elo**

| Force | Bon vainqueur (330 annotés) | Log loss | Bon vainqueur (37 inédits) | Log loss |
|---|---|---|---|---|
| Elo classique (K = 20) | 65,5 % | 0,822 | 67,6 % | 0,892 |
| Elo mieux réglé (K = 40 à 60) | 68,2 % | 0,816 | 73,0 % | 0,830 |
| **Filtre de Kalman** (réglage fixé d'avance) | **68,5 %** | **0,813** | **73,0 %** | **0,824** |

Les équipes ont peu de matchs dans l'historique (5 en médiane, 57 équipes sur 133 en ont 3 ou moins) : il faut
apprendre vite au début, puis moins. Kalman le fait seul, sans chercher le bon K sur les matchs d'évaluation, et
donne une incertitude par équipe (affichée sur la page). Ses 27 réglages voisins donnent tous de 67 à 69 %. La
log loss baisse à chaque point de contrôle ; plus tard dans le match, le commentaire prend le dessus et l'écart
s'efface.

**Limites** : un but sur deux n'est pas annoncé assez clairement dans la transcription pour être détecté
seul (traduction bruitée, commentaire manquant) ; les scores cités en rattrapent une partie. Exemple :
Bayern – Francfort (2015), commenté en allemand puis traduit, où personne ne crie « Goal! ». Les trois buts
sont comptés grâce à « Bayern are leading 1-0 », « the 2-0 » et « 3-0 Muller », souvent plusieurs minutes après
le but.

Où sont les buts ratés : 62 % des vrais buts ont un candidat sûr à plus de 50 %, 9 % entre 30 et 50 %, 11 %
entre 10 et 30 %, et 17 % moins de 10 % ou aucun (souvent une traduction illisible). Un second avis parfait,
par exemple une IA qui relirait les candidats à plus de 10 % (5,5 par match), retrouverait au mieux 78 % des
buts. Les indices simples (applaudissements notés par Whisper, « replay », débit de parole) ne changent pas
autour des buts ratés.

Le modèle de langue améliore la détection des buts mais pas sensiblement le score final (écarts dans le bruit). Il n'est
pas ré-entraîné (trop lourd sur un processeur M1 sans accélération graphique avec cette version de macOS).

## Fichiers

- `donnees.py` : lecture des commentaires, buts et compositions
- `detection_buts.py` : détection des buts et évaluation ; écrit `resultats/` (relire les erreurs dans
  `buts_detectes.csv`, colonne `verdict`)
- `direct.py` : prédictions minute par minute et page ; `gabarit_page.html` : le modèle de la page ;
  `equipes.py` : codes et couleurs des équipes (affichage)
- `cache/` : vecteurs du modèle de langue (recalculés si les mentions changent)
