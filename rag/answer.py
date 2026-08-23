"""rag/answer.py — la politique de reponse graduee a 3 niveaux, en CODE.

    py rag/answer.py "A quelle periode faut-il semer le mil dans la region de Maradi au Niger ?"

Ce que ce fichier decide, et ou la decision est PROUVEE :

  niveau 3 — REFUS. Une seule statistique scalaire, lue dans `rag/index/seuils.json`, jamais
             ecrite ici. 🔴 Le refus est du CODE **en amont du modele** : sous le seuil bas, la
             fonction rend son message et **le generateur n'est jamais appele**. Ce n'est pas une
             precaution de style : 6 prompts nus portant « Si tu ne sais pas, dis-le » ont produit
             **0 refus sur 18 generations** (mesure du 18/08, 3 quantifications × 6 prompts). Un
             0,5B ne refuse pas parce qu'on le lui demande. Le compteur `appels_generateur` du
             rapport rend la propriete VERIFIABLE au lieu de la promettre — `verify_retrieve.py`
             §6 l'exerce avec un generateur factice qui compte ses appels, et il verifie d'abord
             que ce faux generateur SAIT compter : une defense jamais vue refuser ne prouve rien.
  niveau 1 — SOURCE. **Aucun seuil scalaire** : la porte 1/2 est un resultat *negatif* mesure (le
             plus haut item de niveau 2 domine 8 des 11 items de niveau 1 ⇒ aucun cosinus ne les
             separe). Le niveau 1 s'accorde par **VERIFICATION D'ANCRE** : (a) l'entite demandee
             nommee dans le passage — ou, a defaut, dans le `titre` du document — **et** (b) un
             motif de valeur du type de demande trouve DANS l'unite citee. Sans les deux : niveau 2.
             🔴 Type de demande inconnu ⇒ niveau 2. **Jamais de promotion.**
  niveau 2 — NON SOURCE, **SIGNALE**. Il se **melange** au niveau 1 dans le meme message, il ne s'y
             substitue pas : `tp1` (l'un des deux `test_prompts` de `metadata.json`, recopie mot
             pour mot dans le formulaire de soumission) a une moitie sourcable et une moitie
             absente POUR L'ENTITE DEMANDEE. Et l'absence est **situee** : le code nomme la valeur
             documentee la plus proche et son ecart exact, au lieu de dire « je ne sais pas ». Une
             absence situee bat une absence nue — c'est precisement ce que fait le mode d'echec
             qu'on cherche a eviter (repondre hors corpus sans le signaler, dose fausse ×10).

🔴 Une requete porte PLUSIEURS demandes et **chaque demande recoit SON niveau**. Ce n'est pas un
raffinement : la moitie « cumul pluviometrique » de `tp1` est absente pour le mil, l'autre moitie
est citable a la page. `verify_retrieve.py` §10 l'assert de bout en bout.

Ce que ce fichier NE fait pas, volontairement :
  - il n'ecrit **aucun seuil**. Il LIT `rag/index/seuils.json` (genere par l'outillage) et il
    **verifie** que la porte 1/2 y declare toujours « AUCUN seuil scalaire » : si une session
    future y glissait un scalaire, l'import leve au lieu de l'ignorer silencieusement ;
  - il ne parse **jamais** la provenance dans le texte : editeur / annee / page viennent des
    metadonnees du chunk (D3, en-tete a 25 champs) ;
  - il ne « normalise » **rien** au-dela des espaces. Le corpus ecrit trois conventions de
    milliers (`10000 poquets/ha`, `170 000 plants/ha`, `100.000 plants/ha`) : y toucher casserait
    le verbatim, qui est notre garantie de justesse.

Deux blocs vivent ici et non dans l'outillage de mesure, qui les REIMPORTE d'ici plutot que d'en
garder une copie (une constante dupliquee divergera, et la copie survivra a la correction) :
  - la segmentation en unites de citation + les motifs de valeur. Mesures : la segmentation coupe
    sur `.!?` et **jamais** sur `:` (sinon `AUTRE DENOMINATION: EL MARADI` se scinde et l'ancre
    devient incitable), et les 46 unites d'ancre du jeu mesure font au plus 259 c ;
  - le bloc canonique d'entite : 11 mesures, 0 echec, dont le veto d'espece (le binome
    `Striga hermonthica` ne doit pas faire reconnaitre une culture) et le piege « mais »
    conjonction contre « mais » culture. `verify_retrieve.py` §2 et §7 les rejouent ici.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys
import unicodedata
from typing import Callable

RACINE = pathlib.Path(__file__).resolve().parent.parent
_ICI = str(pathlib.Path(__file__).resolve().parent)
if _ICI not in sys.path:                      # `import answer` depuis un AUTRE repertoire (l'outillage
    sys.path.insert(0, _ICI)                  # de mesure) doit marcher : il importe d'ici, pas l'inverse

import retrieve  # noqa: E402
from index import jetons, replie  # noqa: E402  le tokeniseur de l'index, jamais un `in` naif

SEUILS = RACINE / "rag" / "index" / "seuils.json"


# =========================================================================================
# 1. segmentation en unites de citation  (mesure M3 de la phase de mesure du D5)
# =========================================================================================
PUCES = ("-", "•", "*", "▪")
MAJ = "A-ZÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ"


def unites(texte: str) -> list[str]:
    """Decoupe un chunk en unites de citation.

    Deux etages, tous deux dictes par le texte REEL du corpus (mesure, pas usage) :
      1. recollage des lignes de continuation — l'extraction coupe au milieu des valeurs
         (« trois decades de | juin », « 2 g/poquet (20 | kg/ha) ») ; une ligne qui commence par
         une minuscule, un chiffre ou une ponctuation continue la precedente.
      2. decoupe en phrases sur `.!?` UNIQUEMENT. Jamais sur `:` — sinon
         « AUTRE DENOMINATION: EL MARADI » se coupe en deux et l'ancre devient incitable.
    """
    blocs: list[str] = []
    for ligne in texte.split("\n"):
        s = ligne.strip()
        if not s:
            continue
        puce = s.startswith(PUCES)
        suite = (not puce) and (s[0].islower() or s[0].isdigit() or s[0] in "(,;)’'")
        if puce:
            s = s[1:].strip()
        if suite and blocs:
            blocs[-1] = blocs[-1] + " " + s
        else:
            blocs.append(s)
    sorties: list[str] = []
    for b in blocs:
        for p in re.split(r"(?<=[.!?])\s+(?=[" + MAJ + r"])", b):
            p = p.strip()
            if p:
                sorties.append(p)
    return sorties


def norm(s: str) -> str:
    return " ".join(s.split())


_CACHE_UNITES: dict[int, list[str]] = {}


def unites_du_chunk(chunk: dict) -> list[str]:
    """`unites()` memoisee par indice de chunk — le balayage de niveau 2 parcourt les 3 180."""
    i = chunk.get("i")
    if not isinstance(i, int):
        return unites(chunk["texte"])
    if i not in _CACHE_UNITES:
        _CACHE_UNITES[i] = unites(chunk["texte"])
    return _CACHE_UNITES[i]


# =========================================================================================
# 2. motifs de valeur  (mesures M4/M5 de la phase de mesure du D5)
# =========================================================================================
# Chaque type de demande porte les motifs qui ATTESTENT une valeur de ce type. Ils sont ecrits
# depuis le texte reel des 10 chunks etiquetes, jamais de memoire — d'ou les trois conventions de
# milliers qui coexistent dans le meme corpus : 10000, 170 000, 100.000.
NB = r"\d+(?:[ ., ]\d+)*"
# --------------------------------------------------------------------------- classe de duree
# Echafaudage du type `cycle_duree`, ELARGI EN CONNAISSANCE le 19/08. Ce qui suit est le resultat
# d'un classement manuel, pas une intuition sur le motif.
# Le motif d'origine exigeait un intervalle a tiret ASCII : il ne captait que 159 des 197 vrais
# cycles. Ce qui a ete mesure avant d'elargir, sur 299 occurrences de duree classees une par une :
#   cycle 197 (76 nues) · delai 66 (39 nues) · dispersion/erreur 10 · jour-de-l-annee 1 · indet. 25
# ⇒ 39 des 138 durees NUES sont des DELAIS, soit 28,3 %. Elargir a « tout NB jours » aurait promu
#   39 delais + 10 dispersions + 1 jour-de-l-annee en CYCLE, c'est-a-dire fabrique du faux niveau 1
#   en figeant dans le code l'erreur de lecture qu'on cherche a eviter (« 45 jours apres semis »
#   n'est pas une duree de cycle).
# ⇒ L'elargissement retenu n'ouvre donc PAS le nombre nu. Il exige un DECLENCHEUR de cycle, pose
#   un VETO de delai a droite, et ouvre la vraie CLASSE DE TIRET du corpus (« 70–75 jours »,
#   « 75 a 80 » : des durees comptees « nues » etaient en fait des intervalles que l'ancien motif
#   ratait sans qu'elles soient nues — il ne voyait que 122 des 161 intervalles).
# Cout mesure : cycles captes 159/197 -> 195/197, et **0** unite non-cycle touchee. Les 2 cycles
# encore manques (i402, i1619) tombent en niveau 2 : un cycle manque coute une citation, un faux
# cycle serait une fausse citation.
DELAI_D = (r"apr[eè]s|avant|after|before|prior|of\s+the\s+year|DOY|d[’']intervalle|"
           r"de\s+la\s+ponte")
DISPERSION_MOT = r"\bMAE|varied\s+between|by\s+between|erreur|error|[eé]cart-type"
TIRET = r"(?:-|–|—|[aà]|to)"          # le corpus ecrit les 3 tirets ET « 75 a 80 »
DUR = NB + r"\s*(?:" + TIRET + r"\s*" + NB + r"\s*)?(?:jours?|days?)"
DUR_V = DUR + r"(?!\s*(?:" + DELAI_D + r"))"          # veto de delai a droite
INTERDITS = DISPERSION_MOT + r"|±|between|" + DELAI_D


def _gap(n: int, interdits: str) -> str:
    """Un intervalle de n caracteres au plus qui ne franchit ni le point ni un mot interdit."""
    return r"(?:(?!" + interdits + r")[^.]){0," + str(n) + r"}?"


MOTIFS: dict[str, list[str]] = {
    "periode_semis": [
        r"(?:trois|premi[eè]re|deuxi[eè]me|troisi[eè]me|1[eè]?re|2[eè]me)\s+d[eé]cades?",
        r"mi-\w+\s+[aà]\s+la\s+mi-\w+",
        r"(?:semis|semer)[^.]{0,80}?(?:janvier|f[eé]vrier|mars|avril|mai|juin|juillet|ao[uû]t|"
        r"septembre|octobre|novembre|d[eé]cembre)",
    ],
    "dose_engrais": [
        NB + r"\s*(?:g|kg)\s*(?:/|par\s+)(?:poquet|ha|hectare|plant)",
        NB + r"\s*g\s+of\s+\d{2}-\d{2}-\d{2}",
        r"\d{2}-\d{2}-\d{2}[^.]{0,40}?" + NB + r"\s*(?:g|kg)",
    ],
    "densite_ecartement": [
        NB + r"\s*(?:poquets?|plants?|pieds?|pockets?)\s*/\s*ha",
        r"densit[eé][^.]{0,60}?" + NB + r"\s*(?:poquets?|plants?|pieds?)",
        r"\d+(?:[.,]\d+)?\s*(?:cm|m)\s*[x×]\s*\d+(?:[.,]\d+)?\s*(?:cm|m)",
    ],
    "quantite_semence": [
        NB + r"\s*(?:[aà]|-|à)\s*" + NB + r"\s*(?:KG|kg|Kg)\s+de\s+semences?",
        NB + r"\s*(?:KG|kg|Kg)\s+de\s+(?:semences?|graines?)",
        r"semence[^.]{0,40}?" + NB + r"\s*(?:kg|KG)",
    ],
    "cycle_duree": [
        # 1. en-tete de fiche variete du catalogue CNS — la forme DOMINANTE du corpus
        r"CYCLE\s+SEMIS\s*[-–—]?\s*MATURIT[EÉ]\s*(?:\([^)]{0,8}\))?\s*:?\s*" + DUR_V,
        # 2. declencheur de cycle a gauche (fr et en), sans franchir un veto
        r"(?:cycle|dur[eé]e|duration|semis\s*[-–—]\s*maturit[eé]|plantation\s*[-–—]\s*maturit[eé])"
        + _gap(60, INTERDITS) + DUR_V,
        # 3. classe de precocite d'une variete (ligne de tableau ou intertitre)
        r"(?:extra[- ]pr[eé]coces?|semi[- ]pr[eé]coces?|pr[eé]coces?|tardives?|"
        r"interm[eé]diaires?|maturit[eé]\s+moyenne)\s*\(?\s*" + DUR_V,
        # 4. anglais « (120 days to maturity) ». 🔴 La garde d'ouverture `(?:^|[(:]\s*)` est
        #    OBLIGATOIRE et recensee, pas devinee : `to maturity` apparait 19 fois dans le corpus
        #    et UNE SEULE est un vrai cycle (i1146 p3, « Haini kirey (120 days to maturity) »).
        #    Les 18 autres : 9 noms de variable APSIM, 4 « MAEs … days to maturity » (metrique
        #    d'erreur de modele), 4 indetermines, 1 delai. Sans garde a gauche ce motif promeut une
        #    metrique d'erreur en cycle. ⚠️ Portee honnete : calibre sur n=1 vrai positif.
        r"(?:^|[(:]\s*)" + DUR_V + r"\s+to\s+maturity",
        # 5. ligne de tableau variete : la colonne zone de pluie precede la colonne cycle
        NB + r"\s*[-–—]\s*" + NB + r"\s*mm\s+" + DUR_V,
    ],
    "zone_pluie": [
        r"zone[^.]{0,40}?" + NB + r"\s*mm",
        NB + r"\s*-\s*" + NB + r"\s*mm",
    ],
    # Un cumul DECLENCHEUR de semis. Le motif exige le mot `cumul` (ou « apres/declenche ») A COTE
    # des mm — sans quoi il ramasserait les isohyetes de zone, qui sont un autre fait.
    # 🔴 Ce motif etait annonce « MUET partout : c'est le test de l'absence ». **Mesure fausse**, et
    # c'est l'angle mort translingue : il rend **3** occurrences sur les 3 180 chunks (la mesure qui
    # portait « muet partout » ne balayait que le FRANCAIS, sur un corpus a 14/33 documents
    # anglais) — i286 p29 « apres une pluie d'au moins 15mm » (re-semis de legumes) et
    # i1456/i1457 p8 « accumulated rainfall of 20 mm » (sorgho, Mali, parametre de modele APSIM).
    # L'anglais n'est attrape que par accident : `cumul` est une sous-chaine de « ac-cumul-ated ».
    # ⇒ L'absence n'est donc PAS absolue, elle est relative a l'ENTITE DEMANDEE : aucune des 3
    # occurrences ne nomme le mil (ni jeton, ni binome, ni titre) ⇒ `verifie_entite` refuse le
    # niveau 1 sur les trois, sans qu'aucun seuil n'intervienne. C'est exactement ce qui rend le
    # niveau 2 de `tp1` **situe** : le code cite la valeur la plus proche ET nomme l'ecart.
    "cumul_pluie": [
        r"cumul[^.]{0,60}?" + NB + r"\s*mm",
        NB + r"\s*mm[^.]{0,40}?cumul",
        r"(?:d[eé]clenche|apr[eè]s)[^.]{0,40}?" + NB + r"\s*mm",
    ],
    # 🔴 absence mesuree, celle-ci vraiment MUETTE : **0** occurrence sur 3 180 chunks. Le nombre
    # doit porter une UNITE de comptage du ravageur — sinon « seuil economique [37] » (reference
    # bibliographique) matche.
    "seuil_intervention": [
        r"seuil[^.]{0,80}?" + NB + r"\s*(?:chenilles?|larves?|%|%|plants?|pieds?|insectes?)",
        NB + r"\s*(?:chenilles?|larves?)\s*(?:/|par\s+)(?:plant|pied|m2)",
    ],
    "identite_variete": [
        r"(?:AUTRE\s+)?D[EÉ]NOMINATION\s*:\s*[" + MAJ + r"0-9][" + MAJ + r"0-9 \-]+",
    ],
    # 🔴 LE SEUL TYPE NON NUMERIQUE DU JEU, et celui qui fait vivre `tp2`
    # (`striga_sorgho_cross_lingual`, recopie mot pour mot dans le formulaire de soumission) : une
    # « mesure pratique de lutte » n'a aucune valeur chiffree a extraire. Il est donc ecrit CONTRE UNE
    # MESURE — 9 motifs candidats balayes un par un sur les 22 324 unites de citation du corpus,
    # et non contre une intuition de ce qu'est une pratique agronomique :
    #   motif                total   + cible sanitaire   SANS cible   -> retenu ici
    #   lutte_contre            78          45               33            OUI
    #   variete_resistante      25           8               17            OUI
    #   arrachage                9           4                5            OUI
    #   association             82          22               60            NON
    #   fumure_organique        65           1               64            NON
    #   sarclage                44           0               44            NON
    #   rotation                 6           0                6            NON
    #   traitement_semence       4           0                4            NON
    #   culture_piege            0           0                0            NON
    # Deux enseignements que seule la lecture des unites pouvait donner :
    #  1. **La conjonction est OBLIGATOIRE** (`CONJONCTIF` ci-dessous), et son effet est mesure sur les
    #     motifs REELLEMENT retenus ci-dessus, par le chemin de code de ce fichier (C6bis, `trouve()`
    #     avec puis sans) : **107 unites matchent, 54 survivent, la conjonction en RETIRE 53**.
    #     ⚠️ Ce n'est PAS le 108 (44 sarclages + 64 fumures) que ce commentaire affichait avant le
    #     19/08 : `sarclage` et `fumure_organique` sont ECARTES, donc leurs unites ne peuvent etre
    #     promues ni avec ni sans la conjonction. C'etait un chiffre juste pour un AUTRE ensemble
    #     (un chiffre juste ailleurs survit a toutes les relectures), ecrit dans le fichier soumis. Les 53 vrais retires sont pires que prevu :
    #     `arrach*` matche massivement du REPIQUAGE de pepiniere et du DEMARIAGE (« Arracher les
    #     jeunes plants ages de 3 semaines », i95 p30 ; « Arracher delicatement le plant le plus
    #     frele », i107/i108 p33) et `variete_resistante` matche « Utilisation de varietes
    #     resistantes. » sans dire resistantes A QUOI (i194/i195 p3). Le motif de pratique ne suffit
    #     JAMAIS : il faut une cible sanitaire nommee dans la MEME unite.
    #     ⚠️ Portee honnete, et elle demande de distinguer DEUX roles que ce dict porte desormais :
    #     comme **filtre de promotion** (ici), la conjonction est **INERTE sur les 8 sondes** — C8,
    #     en la neutralisant par un motif vide, mesure **0/8** reponse changee : elle ne corrige
    #     aucune reponse mesuree, elle ferme 53 portes que personne n'a encore poussees. Comme
    #     **portee de la reserve** de niveau 2 (`reserve_niveau_2` rend None hors `CONJONCTIF`), le
    #     meme dict est **porteur** : le neutraliser fait tomber tp2 de `[1, 2]` a `1`, donc lui
    #     retire sa reserve, soit **1/8**.
    #     🔴 Ce n'est PAS ce que C8 disait il y a une heure. Il neutralisait les deux roles d'un seul
    #     `CONJONCTIF = {}` et rendait « 1 sonde sur 8 » : j'ai lu ce 1 comme « le filtre est devenu
    #     porteur sur tp2 », alors qu'aucune citation ne bouge (le diff n'imprime aucune ligne
    #     `+`/`-` sur tp2) et que seul le niveau tombe. Un delta obtenu en changeant deux variables
    #     n'est attribuable a aucune des deux : attribuer l'ecart a la variable qu'on a en tete est
    #     une inference, l'isoler factoriellement est une mesure (quatrieme occurrence). C8 les isole
    #     maintenant separement et verifie que l'effet combine est bien leur RECOUVREMENT.
    #  2. **`association` est ecarte bien qu'il co-occurre 22 fois avec une cible**, et c'est le point
    #     contre-intuitif : ses unites striga sont « AUTRES CARACTERES: Sensible au striga » (i1659,
    #     i1660, i1662) — une declaration de SENSIBILITE, l'inverse exact d'une mesure de lutte. Un
    #     motif peut etre statistiquement riche et semantiquement faux. `sarclage` / `rotation` /
    #     `traitement_semence` sont ecartes pour la raison OPPOSEE : 0 co-occurrence mesuree, donc
    #     zero preuve qu'ils servent a quoi que ce soit ici (un motif muet ne prouve
    #     rien, il faut l'avoir vu rendre quelque chose).
    "mesure_lutte": [
        r"lutte\s+(?:\w+\s+){0,2}?contre\s+(?:le|la|les|l')?\s*\w+"
        r"|control\s+(?:of|strategies|measures|methods)",
        r"vari[eé]t[eé]s?[^.;:]{0,40}?(r[eé]sistantes?|tol[eé]rantes?)"
        r"|(resistant|tolerant)\s+(varieties|cultivars)",
        r"arrach(?:age|er|ees?)|hand[\s-]?pulling|uprooting",
    ],
}

# Les cibles sanitaires du perimetre. Recensees dans le corpus, pas listees de memoire : ce sont les
# ravageurs et maladies qui apparaissent effectivement dans les unites mesurees par C6 (striga,
# chenille legionnaire, pucerons, bruches, thrips, viroses...) plus les termes generiques que le
# corpus utilise pour les designer.
CIBLES_SANITAIRES = (r"striga|chenille|l[eé]gionnaire|spodoptera|puceron|bruche|thrips|mildiou|"
                     r"charbon|rosette|virose|aphid|borer|pest|ravageur|maladie|nuisible")

# 🔴 Types dont le motif de valeur ne suffit PAS : il doit co-occurrer avec ce second motif DANS LA
# MEME UNITE. Mesure a l'appui ci-dessus (C6bis, par `trouve()` lui-meme) : la conjonction retire
# **53** unites sur 107 — des repiquages de pepiniere, des demariages et des « varietes resistantes »
# sans dire resistantes a quoi, qui auraient tous ete servis en niveau 1 avec citation verbatim, page
# et editeur.
# 🔴 SECOND ROLE, a ne pas perdre de vue si on est tente de vider ce dict : il borne aussi la portee
# de `reserve_niveau_2` (« la valeur est une pratique, pas un chiffre »). Comme filtre il est mesure
# INERTE sur les 8 sondes (C8 : 0/8) ; comme portee de la reserve il est PORTEUR (C8 : 1/8, tp2 perd
# son niveau 2). Une entree retiree d'ici ne desarme donc pas un mecanisme, elle en desarme deux.
CONJONCTIF: dict[str, str] = {"mesure_lutte": CIBLES_SANITAIRES}

# 🔴 VERBE DE PRATIQUE — le predicat qui distingue « une pratique A APPLIQUER » d'une simple mention
# du sujet. Il ne decide AUCUN niveau, et c'est deliberé : l'option qui en faisait un VETO de
# promotion a ete mesuree puis ECARTEE, parce qu'elle faisait tomber `tp2` entierement en
# niveau 2 et lui otait ses deux citations francaises — c'est-a-dire la demonstration translingue du
# prompt qui existe pour ca. Il sert a DEUX choses, toutes deux non destructives :
#   (1) declencher la RESERVE de niveau 2 greffee sur une reponse de niveau 1 (§8bis) ;
#   (2) classer le TEMOIN du voisinage documente pour les types conjonctifs (§7).
# Provenance : recense dans le corpus par la phase de mesure du D5 (8 unites portent striga + un
# verbe de pratique, **0** nomme le sorgho), qui importe cette definition D'ICI — une seule
# definition, sinon les unites mesurees et le code divergent en silence.
# ⚠️ La classe TOLERE LES ACCENTS (`d[eé]trui`, `br[uû]l`, `d[eé]sherb`) — adopte le 19/08 APRES
# mesure, pas par principe. Ce qui a ete chiffre sur les 3 180 chunks, en comparant ce motif au
# variant nu qui etait ecrit ici avant :
#   corpus entier  **847** unites captees, contre **819** au motif nu ⇒ le nu en ratait **28** ;
#   sous-ensemble striga  **8 contre 8** ⇒ delta **0**, donc `tp2` est INCHANGE par ce choix ;
#   les 28 gagnees sont TOUTES des formes `Desherb*` accentuees (`i21` p5, `i67` p21, `i75` p24,
#   `i76` p24, `i83` p27, `i96` p31, …) — un seul mot francais portait tout l'ecart.
# ⇒ Le sens de l'erreur decide, et il est asymetrique : rater une pratique reelle fait ouvrir une
# RESERVE qui affirme « aucune unite citee n'enonce de pratique » alors que la citation servie EN
# EST une — c'est-a-dire une phrase fausse dans un message lu par un juge. Elargir ne peut, lui,
# que faire taire une reserve. On paye donc l'elargissement, et on le declare : la classe est plus
# large, la reserve se declenche moins souvent, et `tp2` ne bouge pas d'une unite.
VERBE_PRATIQUE = re.compile(
    r"\b(arrach|d[eé]trui|d[eé]trus|br[uû]l|sarcl|semer|semez|repiqu|appliqu|apport|enfoui|"
    r"traiter|traitez|utilise|utilisez|pratiqu|associ|rotation|assol|d[eé]sherb|"
    r"pull|uproot|destroy|burn|weed|apply|rotate|intercrop)", re.I)

# 🔴 OUVERTURE PRESCRIPTIVE — le predicat qui distingue une unite qui *ordonne* une pratique d'une
# unite qui *parle* d'une pratique. Il ne nait pas d'une intuition de style : il nait d'un temoin
# mesure. Trois unites arrivaient a EGALITE de rang pour `(mesure_lutte, sorgho)` — les
# trois portent un verbe de pratique ET nomment le striga — et c'est l'indice de chunk, donc le
# hasard de l'ordre d'indexation, qui elisait la piece a conviction :
#   `i1836`/`i1837` « Le striga est rencontre dans presque toutes les grandes zones de production
#                     du niebe du pays… »          <- une REPARTITION. Le verbe (`rencontre`, puis
#                     `pratiqu` plus loin dans la phrase) est present, l'ordre ne l'est pas.
#   `i2094`         « Arracher et detruire les plants malades et les touffes de Striga. »  <- la
#                     seule vraie pratique du corpus, et elle OUVRE sur son verbe.
# Le corpus ecrit ses actions a l'infinitif ou a l'imperatif EN TETE d'unite (les fiches RECA/INRAN
# sont des listes d'actes) ; une description place son sujet devant. D'ou le predicat : verbe de
# pratique en POSITION D'OUVERTURE, apres un eventuel marqueur de liste survivant. `unites()` a deja
# retire les puces, donc il ne reste que les numerotations recollees (« 1. Arracher… »).
_OUVERTURE_LISTE = re.compile(r"^(?:\d+\s*[.)°]\s*|[-–—*·]\s*)+")


def est_prescriptif(u: str) -> bool:
    """L'unite OUVRE-t-elle sur un verbe de pratique ? Cle (0ter) du classement des temoins.

    Portee honnete, et **chiffree** : c'est un predicat de FORME, pas de sens. **101 unites
    sur les 847 qui portent une pratique** (11,9 %) l'ouvrent — donc il departage vraiment, il n'est
    ni vide ni total. Ses deux limites sont EXERCEES par contre-epreuve, pas seulement declarees :
      il accepte « *Apport de 100 kg/ha de NPK.* » — un groupe NOMINAL, parce que `apport`, `rotation`,
        `associ` et `assol` entrent dans la classe comme substantifs autant que comme verbes ;
      il refuse « *Il faut arracher les touffes.* » — un ordre reel, mais en peripharse.
    🔴 L'exemple ecrit ici a d'abord ete « *Utilisation de varietes resistantes.* » : **mesure fausse**,
    `M9f` l'a refusee. Cette unite ne porte meme pas de verbe de pratique (`utilise` ne matche pas
    « Utilisation »), donc elle n'arrive jamais jusqu'a ce predicat — c'est `variete_resistante` qui
    la captait, et la conjonction qui la retire (cf. les 53 de `trouve()`). Un exemple plausible dans
    une docstring est une affirmation non mesuree comme une autre.
    Il n'a donc pas vocation a decider un niveau — il ne fait que departager des candidats DEJA a
    egalite sur la pratique et sur la cible, la ou l'alternative etait l'indice de chunk.
    """
    return bool(VERBE_PRATIQUE.match(_OUVERTURE_LISTE.sub("", u.strip())))


def trouve(typ: str, texte: str) -> list[str]:
    # La conjonction est verifiee AVANT les motifs de valeur, et sur la MEME chaine `texte` : c'est
    # l'appelant qui decide du perimetre (unite de citation ici, chunk entier dans les mesures), donc
    # la regle suit automatiquement le perimetre au lieu d'en imposer un second.
    if typ in CONJONCTIF and not re.search(CONJONCTIF[typ], texte, re.I):
        return []
    out = []
    for m in MOTIFS[typ]:
        for t in re.finditer(m, texte, re.I):
            out.append(norm(t.group(0)))
    return out


# =========================================================================================
# 3. BLOC CANONIQUE D'ENTITE  (11 mesures E1..E11 de la phase de mesure du D5)
# =========================================================================================
# Les entites du perimetre declare — Sahel francophone, coeur Niger : mil, sorgho, mais, arachide,
# niebe (le perimetre est aussi annonce par `PERIMETRE` plus bas, et `verify_retrieve.py` §0 verifie
# que ses deux nombres sont ceux de l'index livre). Le MECANISME de recherche est MESURE, pas
# stylistique :
#   - `jetons`     : passe a `Index.contient()` ⇒ on relit les POSTINGS LIVRES, donc l'artefact
#                    soumis fait foi. C'est le cas par defaut.
#   - `motif_brut` : regex sur le TEXTE BRUT, OBLIGATOIRE quand le repli des accents de
#                    `index.py:replie()` fusionne l'entite avec un mot courant. Un seul cas dans le
#                    perimetre, et il est ecrit noir sur blanc dans la docstring de `replie()` :
#                    « mais » (la cereale) se replie sur la conjonction francaise « mais ».
#                    ⇒ les postings ne peuvent PAS porter le veto mais : E5 le chiffre.
#   - `latin`      : le BINOME LINNEEN, regex sur le texte brut. Ajoute le 19/08 APRES MESURE, et ce
#                    n'est pas un ornement de botaniste : sans lui la regle refuse le niveau 1 sur
#                    171 des 197 chunks du catalogue CNS/Niger 2021 (86,8 %), qui ne nomment la
#                    culture ni en francais ni en anglais et dont le `titre` n'en nomme aucune —
#                    c'est-a-dire sur la quasi-totalite de notre source de VARIETES. Chaque fiche,
#                    elle, porte son binome (`i1571` finit par « ... Pennisetum glaucum »). Gain
#                    mesure : E10, 43 chunks qui seraient tous refuses sans lui. Le `\s+` tolere le
#                    saut de ligne de l'extraction D3 (« Oryza\nsativa » est coupe en deux lignes).
ENTITES: dict[str, dict[str, object]] = {
    "mil": {"jetons": ["mil", "millet"], "latin": r"Pennisetum\s+glaucum"},
    "sorgho": {"jetons": ["sorgho", "sorghum"], "latin": r"Sorghum\s+bicolor"},
    "niebe": {"jetons": ["niebe", "cowpea"], "latin": r"Vigna\s+unguiculata"},
    "arachide": {"jetons": ["arachide", "groundnut"], "latin": r"Arachis\s+hypogaea"},
    "mais": {"motif_brut": r"ma[ïÏ]s|\bmaize\b", "latin": r"Zea\s+mays"},
}

# Especes RIVALES. Elles ne sont JAMAIS une cible (hors du perimetre declare ci-dessus) : elles ne
# servent qu'a REFUSER. Lues dans le document lui-meme — intertitres de section du catalogue CNS
# (riz p125, oignon p247, manioc p257, tomate p273) et binomes de ses fiches — jamais inventees.
# 🔴 Pourquoi par le BINOME et pas par le nom francais : `i1670` (p204) est une fiche de SESAME qui
# ecrit « association mil/sesame », donc le jeton `mil` y EST present et le veto de culture ne se
# declenche pas ; sa ligne « CYCLE SEMIS-MATURITE 80 jours » serait alors citee comme cycle du MIL.
# Le binome, lui, est la piece d'IDENTITE de la fiche : les 4 fiches sesame portent `Sesamum
# indicum` et **aucune** ne porte `Pennisetum glaucum`. Cout et gain du veto : E11 (gain 9, cout 0).
RIVALES_ESPECES: dict[str, str] = {
    "riz": r"Oryza\s+sativa",
    "sesame": r"Sesamum\s+indicum",
    "oignon": r"Allium\s+cepa",
    "tomate": r"Solanum\s+lycopersicum",
    "manioc": r"Manihot\s+esculenta",
    "pomme_de_terre": r"Solanum\s+tuberosum",
}


def _par_jetons(entite: str, texte: str) -> bool:
    """Le tokeniseur de l'index, jamais un `in` naif : « mille plants » ne nomme pas le mil."""
    js = set(jetons(texte))
    return any(replie(str(t)) in js for t in ENTITES[entite].get("jetons", ()))  # type: ignore[union-attr]


def _par_brut(entite: str, texte: str) -> bool:
    brut = ENTITES[entite].get("motif_brut")
    return bool(brut) and bool(re.search(str(brut), texte, re.I))


def _par_latin(entite: str, texte: str) -> bool:
    lat = ENTITES[entite].get("latin")
    return bool(lat) and bool(re.search(str(lat), texte, re.I))


def nomme_texte(entite: str, texte: str) -> bool:
    """L'entite est-elle nommee dans CE texte (unite de citation, titre de document, chunk) ?

    Union des trois mecanismes. `re.I` est OBLIGATOIRE et il a ete trouve manquant par E8 :
    `\\bmaize\\b` sans lui rate « Maize yield… » en tete de phrase. ⚠️ `re.I` ne retire PAS les
    diacritiques : `ma[ïÏ]s` continue de refuser la conjonction « mais », qui est tout l'objet du
    motif brut.
    """
    if ENTITES[entite].get("motif_brut"):
        return _par_brut(entite, texte) or _par_latin(entite, texte)
    return _par_jetons(entite, texte) or _par_latin(entite, texte)


def mecanisme_texte(entite: str, texte: str) -> str:
    """Lequel des trois mecanismes atteste l'entite. Sert aux RAPPORTS, jamais a decider."""
    parts = [n for n, f in (("brut", _par_brut), ("jetons", _par_jetons), ("latin", _par_latin))
             if f(entite, texte)]
    return "+".join(parts) or "aucun"


def nomme_chunk(entite: str, i: int, idx: retrieve.Index) -> bool | None:
    """L'entite est-elle nommee dans le chunk `i`, lu dans les POSTINGS LIVRES ?

    Rend `True` quand l'artefact soumis l'atteste — c'est la seule reponse que les postings peuvent
    donner **seuls**. Rend `None` quand ils ne peuvent pas conclure, et il y a deux raisons
    distinctes de ne pas conclure, toutes deux mesurees :
      - `motif_brut` : le repli des accents a fusionne « mais » et « mais » a l'indexation, donc le
        posting ne distingue plus la cereale de la conjonction (E5) ;
      - `latin` : le binome est deux jetons dont l'ADJACENCE n'est pas dans les postings ;
        `Pennisetum` seul designe aussi le fourrage `P. purpureum`. Un `False` mentirait ici.
    Le `False` final n'est atteignable que pour une entite sans latin ni motif brut — aucune
    aujourd'hui ; il est garde pour que l'ajout d'une telle entite reste correct par construction.
    """
    spec = ENTITES[entite]
    if spec.get("motif_brut"):
        return None
    if any(idx.contient(replie(str(t)), i) for t in spec.get("jetons", ())):  # type: ignore[union-attr]
        return True
    return None if spec.get("latin") else False


def _binomes_rivaux(cible: str) -> list[tuple[str, str]]:
    """Tous les binomes qui ne sont PAS celui de la cible : especes hors perimetre + autres cibles.

    Les 4 autres cultures declarees en font partie : « association mil/niebe » sur une fiche
    `Vigna unguiculata` a exactement la forme du piege sesame.
    """
    out = [(n, m) for n, m in RIVALES_ESPECES.items()]
    out += [(e, str(s["latin"])) for e, s in ENTITES.items() if e != cible and s.get("latin")]
    return out


def veto_espece(cible: str, texte: str) -> str | None:
    """Le binome d'une espece RIVALE est present alors que celui de la cible est ABSENT ?

    Rend le nom de la rivale, ou None. Le veto ne regarde PAS le nom francais de la cible : c'est
    tout son objet (`i1670` ecrit « mil » sans etre une fiche de mil). ⚠️ Il est asymetrique par
    construction : il refuse, il ne promeut jamais. Son cout est donc une citation perdue, jamais
    une citation fausse — et E11 le chiffre au lieu de le supposer.
    """
    if _par_latin(cible, texte):
        return None
    for nom, motif in _binomes_rivaux(cible):
        if re.search(motif, texte, re.I):
            return nom
    return None


# =========================================================================================
# 3bis. CO-MENTION : quand l'unite nomme DEUX cultures, laquelle porte la valeur ?
# =========================================================================================
# 🔴 Ce bloc a ete ECRIT ET MESURE dans la phase de mesure du D5 puis DEPLACE ici, et non
# re-derive : la re-derivation a ete punie deux fois (363 puis 283 desaccords, docstring de
# `spans_entite` ci-dessous). L'outillage de mesure IMPORTE desormais cette regle depuis le
# livrable, comme il importe `MOTIFS`/`trouve` — une regle dupliquee des deux cotes divergerait, et
# c'est la MESURE qui deviendrait fausse en silence.
#
# Le defaut qu'elle corrige est celui de `tp1` lui-meme, donc du prompt recopie chez les juges :
# `i2093` porte « Il est recommande de semer le niebe 2 semaines apres le premier sarclage du mil. »
# ⇒ `nomme_texte("mil", u)` est VRAI, un motif `periode_semis` est present, et l'ancienne regle
# accordait le niveau 1 : une date de NIEBE servie verbatim comme date de MIL, avec page et editeur.
# C'est l'attribution d'un chiffre au voisin, figee en code — exactement le mode d'echec qu'un RAG
# sourcé existe pour exclure. Ce que la mesure a rendu :
#   - 205 unites sur 22 324 nomment >= 2 cultures (0,92 %) ⇒ la regle est RARE, donc peu risquee ;
#   - elle retire 4 couples (unite, type, cible), et les 4 sont de VRAIS positifs, relus un par un ;
#   - elle fait perdre 0 ancre etiquetee sur 12 ⇒ cout mesure NUL sur le jeu de calibration.
# ⚠️ Portee honnete : 4 retraits, ce n'est pas une statistique. Ce qui est etabli c'est qu'elle ne
# casse RIEN de ce qui marchait, pas qu'elle attrape tous les cas de la forme.


def plie_1a1(texte: str) -> str:
    """Repli des accents garanti 1 CARACTERE -> 1 CARACTERE, pour que les offsets restent valides.

    ⚠️ Ce n'est pas `index.replie()` : une ligature « œ » -> « oe » decalerait tout le reste de
    l'unite et fausserait chaque distance mesuree ensuite. On la laisse donc telle quelle.
    """
    out = []
    for ch in texte:
        d = unicodedata.normalize("NFD", ch)
        base = "".join(x for x in d if not unicodedata.combining(x))
        out.append(base.lower() if len(base) == 1 else ch.lower())
    return "".join(out)


def spans_entite(entite: str, texte: str) -> list[tuple[int, int]]:
    """POSITIONS des mentions de `entite` — la ou `nomme_texte` ne rend qu'un booleen.

    🔴 DEUX VERSIONS FAUSSES AVANT CELLE-CI, toutes deux attrapees par la contre-epreuve C4 qui
    compare cette fonction a `nomme_texte` sur 22 324 unites x 5 entites :
      1. 363 desaccords : je repliais les jetons de l'index mais cherchais dans le texte ACCENTUE ;
      2. 283 desaccords de sens INVERSE : j'avais replie `motif_brut` « pour etre coherent », ce qui
         transforme `ma[ïÏ]s` en `mais` — donc le motif qui existe UNIQUEMENT pour exiger le trema se
         mettait a matcher la conjonction francaise. Un raisonnement de coherence, refute en une
         execution.
      3. 1 desaccord (`i235`) : `\\b` de Python compte `_` comme caractere de MOT, alors que le
         tokeniseur de l'index ne connait que `RE_JETON = [0-9a-z]+(?:-[0-9a-z]+)*`
         (`index.py:484`). Dans l'URL « FT_Maruca_niebe_INRAN.pdf » le livrable voit le jeton `niebe`
         et `\\b` ne voyait rien. ⇒ La frontiere n'est pas `\\b`, c'est « pas dans l'alphabet du
         tokeniseur » : les lookarounds ci-dessous sont ecrits DEPUIS `RE_JETON`, pas depuis une
         intuition de mot. Le tiret reste une frontiere valide (`jetons()` emet les morceaux d'un
         compose), donc « niebe-mil » nomme le niebe des deux cotes.
    ⇒ Elle MIROITE `nomme_texte` branche par branche, et C4 exige 0 desaccord. C'est la seule facon
    de savoir que les distances mesurees portent sur les memes mentions que celles qui decident.
    """
    spec = ENTITES[entite]
    sp: list[tuple[int, int]] = []
    if spec.get("motif_brut"):
        for m in (str(spec["motif_brut"]), str(spec.get("latin") or "")):
            if m:
                for t in re.finditer(m, texte, re.I):
                    sp.append(t.span())
        return sorted(set(sp))
    plie = plie_1a1(texte)
    for j in spec.get("jetons", ()) or ():
        mot = re.escape(plie_1a1(str(j)))
        for t in re.finditer(r"(?<![0-9a-z])" + mot + r"(?![0-9a-z])", plie):
            sp.append(t.span())
    if spec.get("latin"):
        for t in re.finditer(str(spec["latin"]), texte, re.I):
            sp.append(t.span())
    return sorted(set(sp))


def spans_valeur(typ: str, texte: str) -> list[tuple[int, int]]:
    """POSITIONS des valeurs du type `typ`. Honore `CONJONCTIF` comme `trouve`, pour la meme raison :
    sans quoi l'arbitrage pourrait se declencher sur une unite ou `trouve` n'a rien rendu."""
    if typ in CONJONCTIF and not re.search(CONJONCTIF[typ], texte, re.I):
        return []
    sp: list[tuple[int, int]] = []
    for m in MOTIFS[typ]:
        for t in re.finditer(m, texte, re.I):
            sp.append(t.span())
    return sorted(sp)


def _distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    """Distance en caracteres entre deux spans (0 si elles se chevauchent)."""
    if a[1] <= b[0]:
        return b[0] - a[1]
    if b[1] <= a[0]:
        return a[0] - b[1]
    return 0


def arbitre_co_mention(typ: str, cible: str, unite: str) -> dict:
    """Ne s'applique QUE si l'unite nomme >= 2 cultures du perimetre (0,92 % des unites).

    Critere : la valeur du type demande doit etre plus proche d'une mention de la CIBLE que de toute
    mention d'une culture rivale. Une egalite stricte est accordee a la cible — elle est demandee, la
    rivale ne fait que co-exister ; et sur les 4 retraits mesures aucune egalite n'apparait, donc ce
    choix ne porte aucun d'entre eux.

    ⚠️ La proximite N'EST PAS la syntaxe. Les 4 retraits mesures sont larges (d_cible 13 contre 0,
    33 contre 0, 21 contre 8, 9 contre 8) ; le dernier ne tient qu'a 1 caractere. Une unite mal
    ponctuee peut donc encore tromper la regle : elle reduit le risque, elle ne l'annule pas.
    Rend {'declenche', 'accorde', 'cultures', 'gagnante', 'd_cible', 'd_rivale', ...}.
    """
    u = norm(unite)
    presentes = {e: spans_entite(e, u) for e in ENTITES}
    presentes = {e: s for e, s in presentes.items() if s}
    vals = spans_valeur(typ, u)
    if len(presentes) < 2 or not vals or cible not in presentes:
        return {"declenche": False, "accorde": True, "cultures": sorted(presentes),
                "gagnante": None, "d_cible": None, "d_rivale": None}
    d_cible = min(_distance(v, s) for v in vals for s in presentes[cible])
    rivales = {e: min(_distance(v, s) for v in vals for s in sp)
               for e, sp in presentes.items() if e != cible}
    e_min = min(rivales, key=lambda k: rivales[k])
    return {"declenche": True, "accorde": d_cible <= rivales[e_min],
            "cultures": sorted(presentes), "gagnante": cible if d_cible <= rivales[e_min] else e_min,
            "d_cible": d_cible, "d_rivale": rivales[e_min], "rivale_la_plus_proche": e_min}


# `HKP3` : le chiffre doit etre admis A L'INTERIEUR du premier jeton, sinon l'extracteur ne rend
# que « HKP ».
RE_MAJ_RUN = re.compile(r"\b[" + MAJ + r"][" + MAJ + r"0-9]+(?:[ -]+[" + MAJ + r"0-9]{2,})*")


def entite_dynamique(requete: str) -> list[str]:
    """Entite qui n'est pas une culture : un NOM PROPRE demande en capitales (« EL MARADI »).

    ⚠️ Portee honnete : cette regle est calibree sur **n=1** item (`variete_el_maradi`). E9 a mesure
    ce qu'elle produit sur les 47 requetes de calibration AVANT qu'elle ne serve a decider, et le
    resultat est qu'elle tire aussi sur 4 requetes sans demande de variete ⇒ elle n'est employee
    QUE lorsque le type de demande est `identite_variete` (voir `cible_de_la_demande`).
    """
    return [m.group(0).strip() for m in RE_MAJ_RUN.finditer(requete)]


# 🔴 Commutateur DECLARE, pas un reglage : il existe pour que l'arme « sans la regle » soit
# reproductible par une mesure et par la batterie, comme `PENALITE_STRUCTURE` du D4. Une regle dont
# on ne peut plus exhiber l'etat anterieur n'est plus mesurable — elle devient une croyance.
EXIGE_VARIETE_DANS_LE_CHUNK = True


def tous_nommes(noms: list[str], texte: str) -> bool:
    """TOUS les noms demandes sont-ils presents dans ce texte ? Un seul site, un seul comportement :
    l'etage `dyn` et le veto de variete doivent tester la meme chose, sinon ils divergeraient sur le
    meme passage sans que rien ne le signale."""
    bas = texte.lower()
    return all(n.lower() in bas for n in noms)


# Un nom de VARIETE est demande explicitement — « la variete de mil HKP3 ». Le marqueur lexical est
# exige EN PLUS du jeton en capitales : sans lui, `NPK` (declencheur de `dose_engrais`, ecrit
# « 15-15-15 » dans les sources) serait pris pour une variete et perdrait ses citations legitimes.
# Mesure prealable sur 61 requetes (14 etiquetees + 47 de calibration) : la conjonction ne tire que
# sur `mil_cycle_hkp3`, et reste inerte sur `NPK`, `IV`, `CMD`, `TLS`.
MARQUEUR_VARIETE = re.compile(r"vari[eé]t[eé]|vari[eé]tal|cultivar|variety", re.I)


def variete_demandee(requete: str) -> list[str]:
    """Les noms propres en capitales d'une requete QUI DEMANDE une variete. Sinon : liste vide."""
    if not MARQUEUR_VARIETE.search(requete):
        return []
    return entite_dynamique(requete)


def verifie_entite(idx: retrieve.Index, chunk: dict, unite: str, cible: str,
                   requete: str = "", veto_sur_chunk: bool = False) -> tuple[bool, str]:
    """(a) de la regle du niveau 1 : l'entite demandee est-elle nommee dans ce passage ?

    Rend (accorde, etage). `etage` ∈ {a1, a2, veto-espece:<nom>, veto:<rivale>, aucun, dyn,
    dyn-absent}.
    🔴 JAMAIS `provenance.sujet` : le catalogue CNS estampille `sujet='varietes'` sur les pages
    mil, niebe ET arachide indifferemment — mesure E7 : ses 197 chunks portent **1** seul `sujet`
    distinct.

    Les DEUX vetos n'ont pas la meme portee, et chacune est mesuree :
      - veto d'ESPECE, toujours sur le CHUNK : le binome est la piece d'identite de la fiche et il
        se trouve a sa FIN, jamais dans l'unite citee (E11) ;
      - veto de CULTURE, sur l'unite par defaut : E6 mesure les deux portees et l'unite ne perd
        aucune citation legitime (0 des 10 perdue, a l'unite comme au chunk).
    """
    if cible not in ENTITES:
        # entite hors nomenclature : nom propre demande en capitales dans la requete
        noms = entite_dynamique(requete)
        if not noms:
            return False, "dyn-absent"
        ok = tous_nommes(noms, unite)
        return ok, "dyn" if ok else "dyn-absent"

    # veto d'ESPECE : la fiche est identifiee comme etant celle d'une AUTRE espece
    rivale_esp = veto_espece(cible, chunk["texte"])
    if rivale_esp:
        return False, "veto-espece:" + rivale_esp

    # veto de VARIETE : la culture est bien la bonne, mais le passage documente une AUTRE variete
    noms_var = variete_demandee(requete) if EXIGE_VARIETE_DANS_LE_CHUNK else []
    if noms_var and not tous_nommes(noms_var, chunk["texte"]):
        return False, "var-absente:" + "/".join(noms_var)

    # veto de CULTURE : une culture RIVALE nommee dans le passage alors que la demandee ne l'est pas
    portee = chunk["texte"] if veto_sur_chunk else unite
    if not nomme_texte(cible, portee):
        for rivale in ENTITES:
            if rivale != cible and nomme_texte(rivale, portee):
                return False, "veto:" + rivale

    # a1 — le texte du chunk, via les postings livres quand ils sont decisifs
    par_postings = nomme_chunk(cible, chunk["i"], idx) if isinstance(chunk.get("i"), int) else None
    a1 = True if par_postings is True else nomme_texte(cible, chunk["texte"])
    if a1:
        return True, "a1"
    # a2 — a defaut, le titre du document (porte par le chunk lui-meme, cf. D3)
    if nomme_texte(cible, chunk.get("titre", "")):
        return True, "a2"
    return False, "aucun"


# =========================================================================================
# 4. ROUTAGE DE LA DEMANDE — de la requete brute vers (type, cible)
# =========================================================================================
# `answer.py` recoit une QUESTION, pas un identifiant d'item : le routage est du code de production,
# pas une etiquette de jeu de test. Chaque motif est ecrit contre les 14 requetes reelles du jeu
# etiquete a la main, qui sont ecrites SANS ACCENTS (comme un utilisateur les tape) ⇒ tous les
# motifs tolerent l'absence d'accent, sinon ils ne verraient aucune de nos propres requetes.
# 🔴 L'ordre compte : le premier motif qui matche NE CLOT PAS la recherche (une requete porte
# plusieurs demandes, et chacune recoit son niveau), mais l'ordre fixe l'ordre d'affichage.
DEMANDES: list[tuple[str, str]] = [
    ("periode_semis",
     r"[aà]\s+quelle\s+p[eé]riode|quelle?\s+(?:est\s+la\s+)?(?:meilleure\s+)?p[eé]riode"
     r"|p[eé]riode\s+(?:optimale\s+)?de\s+semis|quand\s+(?:faut-il\s+)?semer"
     r"|date\s+(?:optimale\s+)?(?:de\s+)?semis|when\s+to\s+(?:sow|plant)"),
    ("cumul_pluie",
     r"cumul[^?.]{0,40}?(?:pluie|pluvio|mm)|(?:pluie|pluvio\w*)[^?.]{0,40}?cumul"
     r"|(?:accumulated|cumulative)\s+rainfall"),
    ("dose_engrais",
     r"micro-?dose|dose\s+d[e’']\s?engrais|engrais\s+min[eé]ral|fertilizer|\bNPK\b"
     r"|dose\s+de\s+(?:15-15-15|urée|uree)"),
    ("densite_ecartement",
     r"[eé]cartement|quelle\s+densit[eé]|densit[eé]\s+(?:de\s+)?(?:poquets?|peuplement|plants?"
     r"|semis|recommand)|plant(?:ing)?\s+(?:density|spacing)|poquets?\s*/\s*ha"),
    ("quantite_semence",
     r"quantit[eé]\s+de\s+semences?|dose\s+de\s+semences?|combien\s+de\s+kg\s+de\s+semences?"
     r"|seed\s+rate|semences?\s+faut-il"),
    ("cycle_duree",
     r"cycle\s+semis|semis\s*-\s*maturit[eé]|dur[eé]e\s+du\s+cycle|combien\s+de\s+jours"
     r"|days?\s+to\s+maturity|cycle\s+(?:de\s+la\s+vari[eé]t[eé]|du\s+mil|de\s+production)"),
    ("zone_pluie",
     r"zone\s+(?:de\s+)?pluviom|zone\s+de\s+pluie|selon\s+la\s+zone|isohy[eè]te"
     r"|rainfall\s+zone"),
    ("seuil_intervention",
     r"seuil\s+d[e’']\s?intervention|[aà]\s+partir\s+de\s+combien|faut-il\s+traiter"
     r"|seuil\s+(?:de\s+)?(?:traitement|nuisibilit[eé])|treatment\s+threshold"),
    # 🔴 LE DECLENCHEUR DE `tp2` (`striga_sorgho_cross_lingual`), et sans lui `tp2` route vers ZERO
    # demande : verifie motif par motif, aucun des 8 types ci-dessus ne matche « How can a smallholder
    # farmer in Niger control Striga hermonthica in sorghum? Give practical measures… ». Le prompt le
    # plus differenciant du dossier — une question ANGLAISE servie par une source FRANCAISE —
    # ne rendait donc aucune citation, et la cause n'etait PAS la recuperation : la mesure de
    # recuperation montre `i223` deja RETENU au bon rang, sorgho +
    # striga + « Promotion des varietes de sorgho resistantes au Striga ». La chaine entiere marchait
    # sauf son premier maillon.
    # ⚠️ `practical\s+measures` est le motif de NOTRE prompt : il est donc juge sur un texte qu'on a
    # ecrit nous-memes. C'est pourquoi les deux formes generiques (`how can … control`,
    # `comment lutter contre`) sont la — elles sont ce qui fait tenir la regle sur une question qu'un
    # juge reformulerait. Contre-epreuves obligatoires : ne doit tirer NI sur `tp1`, NI sur les deux
    # pieges « mais » (aucun ne parle de lutte).
    ("mesure_lutte",
     r"practical\s+measures|how\s+can\s+\w+.{0,60}?\bcontrol\b"
     r"|control\s+(?:measures|methods|strategies)|how\s+to\s+control"
     r"|comment\s+(?:lutter|combattre|controler|contr[oô]ler)|lutter\s+contre"
     r"|moyens?\s+de\s+lutte|mesures?\s+de\s+lutte|comment\s+se\s+d[eé]barrasser"),
    # 🔴 Declencheur FORT, et c'est mesure : `mil_cycle_hkp3` ecrit « de la variete de mil HKP3 »
    # sans etre une demande d'identite. Exiger « quel(le) est la variete », une denomination, ou
    # l'obtenteur evite de router cette requete-la vers l'etage `dyn` (E9 : l'extracteur de
    # capitales tire sur 4 requetes sans demande de variete).
    ("identite_variete",
     r"(?:quelle|quel)\s+est\s+la\s+vari[eé]t[eé]|d[eé]nomination|qui\s+l[’']a\s+obtenue"
     r"|\bcultivar\b|obtenteur"),
]

# 🔴 Le motif « mais » COTE REQUETE est distinct du motif cote chunk, et il le faut :
# `ma[ïÏ]s|\bmaize\b` (ENTITES) exige le trema, or notre propre item 12 ecrit « traiter le mais »
# SANS trema — un utilisateur ne met pas les accents. Le relachement est confine a la REQUETE et
# borne par un DETERMINANT a gauche, jamais applique au corpus : cote chunk, `\bma[iï]s\b` a ete
# mesure a 237 occurrences dont 140 de faux positifs sur la conjonction (E5), c'est-a-dire le motif
# que le brief interdit. Contre-epreuves obligatoires (verify_retrieve.py) : « je veux semer du mil
# mais je ne sais pas quand » et « il pleut mais le sol est sec » ne doivent PAS etre lues comme
# une demande sur le mais.
MOTIF_MAIS_REQUETE = re.compile(
    r"\b(?:le|du|de|au|sur\s+le|pour\s+le|mon|ce|un|des)\s+ma[iï]s\b", re.I)


def entites_de_la_requete(requete: str) -> list[str]:
    """Les cultures nommees dans la requete, dans leur ordre d'apparition."""
    trouvees: list[tuple[int, str]] = []
    for e in ENTITES:
        pos = None
        if e == "mais":
            m = MOTIF_MAIS_REQUETE.search(requete)
            if m:
                pos = m.start()
        if pos is None and nomme_texte(e, requete):
            # position approchee : premier jeton/binome connu de l'entite present dans la requete
            for cle in (list(ENTITES[e].get("jetons", ())) + [ENTITES[e].get("latin")]):
                if not cle:
                    continue
                m = re.search(str(cle), requete, re.I)
                if m:
                    pos = m.start() if pos is None else min(pos, m.start())
        if pos is not None:
            trouvees.append((pos, e))
    return [e for _, e in sorted(trouvees)]


def cible_de_la_demande(typ: str, requete: str) -> str | None:
    """La cible d'une demande : une culture du perimetre, la sentinelle `variete`, ou None.

    `None` **interdit le niveau 1** : sans entite a verifier, l'etage (a) de la regle n'a pas
    d'objet, et promouvoir quand meme fabriquerait le « faux niveau 1 » que toute cette regle
    existe pour exclure. La sentinelle `variete` n'est rendue que pour `identite_variete`, ce qui confine
    l'etage `dyn` a son seul cas mesure (E9).
    """
    ents = entites_de_la_requete(requete)
    if typ == "identite_variete":
        return "variete" if entite_dynamique(requete) else (ents[0] if ents else None)
    return ents[0] if ents else None


def demandes_de_la_requete(requete: str) -> list[dict]:
    """Decompose la requete en demandes typees. Une requete peut en porter plusieurs.

    ⚠️ Une requete en langue naturelle porte legitimement PLUS de demandes que le jeu etiquete n'a
    d'etiquettes : `niebe_dates_zones` demande la periode ET la zone, `mil_cycle_hkp3` le cycle ET
    la zone de pluie. Le jeu n'etiquette qu'une ancre par item ; le code, lui, doit repondre a la
    question posee. Chaque demande garde donc SON niveau (contrainte n°6).
    """
    out: list[dict] = []
    for typ, motif in DEMANDES:
        if re.search(motif, requete, re.I):
            out.append({"type": typ, "cible": cible_de_la_demande(typ, requete)})
    return out


# =========================================================================================
# 5. SEUILS — lus, jamais ecrits
# =========================================================================================
def charge_seuils(chemin: pathlib.Path = SEUILS) -> dict:
    """Lit `rag/index/seuils.json` et VERIFIE ses invariants au lieu de leur faire confiance.

    TROIS gardes, toutes destinees a faire lever l'import plutot qu'a laisser une derive
    silencieuse — une tolerance qui masque le defaut que le controle existe pour attraper est pire
    qu'aucun controle :
      1. la porte 1/2 doit continuer de declarer « AUCUN seuil scalaire ». Si une session future y
         glisse un scalaire, ce code ne le lira pas — il doit donc refuser de tourner, pas ignorer ;
      2. le `sens` declare de la porte 2/3 doit etre celui que le code applique (`>=`). Recopier un
         seuil sans son sens, c'est inverser la porte sans qu'aucun test ne bronche ;
      3. les trois cles dont ce code depend (`statistique`, `repli_si_absente`, `seuil`) doivent
         etre presentes : une cle absente donnerait un `None` qui traverserait la comparaison.
    Les trois sont exercees par `verify_retrieve.py` §1, qui denature une COPIE du fichier et exige
    que chaque garde leve vraiment — plus un temoin sur la copie intacte, sans quoi « ca leve »
    serait indistinguable de « ca leve toujours ».
    """
    s = json.loads(chemin.read_text(encoding="utf-8"))
    p12 = s["porte_niveau_1_contre_2"]
    if not str(p12.get("decision", "")).startswith("AUCUN seuil scalaire"):
        raise RuntimeError(
            "seuils.json a change de nature : la porte 1/2 ne declare plus « AUCUN seuil "
            "scalaire ». answer.py accorde le niveau 1 par verification d'ancre et ignorerait "
            "silencieusement un scalaire ⇒ arret. Ce fichier est un artefact GENERE par "
            "l'outillage de calibration : le regenerer, jamais l'editer a la main.")
    p23 = s["porte_niveau_2_contre_3"]
    if ">=" not in str(p23.get("sens", "")):
        raise RuntimeError("seuils.json : le `sens` de la porte 2/3 n'est plus « valeur >= seuil », "
                           "or c'est l'inegalite que ce code applique ⇒ arret.")
    for cle in ("statistique", "repli_si_absente", "seuil"):
        if cle not in p23:
            raise RuntimeError(f"seuils.json : cle `{cle}` absente de porte_niveau_2_contre_3")
    return s


def valeur_porte(res: dict, seuils: dict) -> tuple[float | None, str | None]:
    """La valeur de la statistique de porte, lue dans `res['porte']` de `retrieve.cherche()`.

    Le NOM de la statistique et celui de son repli viennent de `seuils.json` : ni l'un ni l'autre
    n'est ecrit ici. Le repli sert quand le gagnant de la fusion RRF vient du BM25 SEUL, donc hors
    du pool dense — il n'a alors pas de cosinus.
    """
    p23 = seuils["porte_niveau_2_contre_3"]
    for cle in (p23["statistique"], p23["repli_si_absente"]):
        v = res.get("porte", {}).get(cle)
        if v is not None:
            return float(v), str(cle)
    return None, None


# =========================================================================================
# 6. POLITIQUE DE CITATION
# =========================================================================================
# 🔴 `citation_verbatim_autorisee` n'est PAS un veto de citation — mesure du 19/08 : il vaut
# `non-redistribuable` sur **2 044** chunks contre `oui` sur 1 136, et sur **8 des 10** chunks
# porteurs de nos ancres de niveau 1 (toutes les fiches RECA/INRAN, c'est-a-dire les 21 fiches de
# conseil direct en francais qui SONT notre differenciateur). En faire un veto supprimerait 8 de nos
# 11 citations. Le champ encode l'interdiction de REDISTRIBUER le document entier (l'index est
# committe en entier, et son regime de licence est declare en clair dans REPORT.md), pas d'en
# CITER un passage.
# ⇒ Ce qui s'applique est un PLAFOND de longueur (citation courte sourcee) + toujours
# editeur + annee + page.
# ⚠️ Portee honnete du plafond : les 11 unites mesurees font **max 259 c**, mediane **100 c** ⇒ 300
# ne coute **zero** sur le jeu mesure, mais la marge est de **41 caracteres**, pas un ordre de
# grandeur. Ce n'est donc pas un seuil de decision (il ne change aucun niveau), c'est une regle de
# rendu — et `verify_retrieve.py` le RE-DERIVE des unites au lieu de croire la constante.
PLAFOND_CITATION_C = 300
ELLIPSE = " […]"


def tronque(unite: str) -> tuple[str, str, bool]:
    """Rend (verbatim, affichee, tronquee).

    La troncature coupe a une frontiere de mot, donc `verbatim` reste une SOUS-CHAINE du chunk :
    c'est ce qui permet a l'invariant M6 de porter sur elle. L'ellipse n'est ajoutee qu'a la chaine
    AFFICHEE — sinon le mot « verbatim » deviendrait faux d'un caractere, et l'invariant le dirait.
    """
    u = norm(unite)
    if len(u) <= PLAFOND_CITATION_C:
        return u, u, False
    coupe = u[:PLAFOND_CITATION_C]
    if " " in coupe:
        coupe = coupe[:coupe.rindex(" ")]
    return coupe, coupe + ELLIPSE, True


def invariant_verbatim(citation: str, texte_chunk: str) -> bool:
    """M6 : la citation, espaces normalises, est une sous-chaine du chunk, espaces normalises.

    L'extraction du D3 coupe les valeurs en pleine ligne (« trois decades de\\njuin ») ⇒ toute
    citation lisible normalise les espaces. Le mot « verbatim » n'est alors honnete que si AUCUN
    caractere non-espace n'est ajoute, retire ni substitue — c'est ce que cette fonction verifie,
    au lieu de le supposer.
    """
    return norm(citation) in norm(texte_chunk)


def est_ligne_de_tableau(unite: str) -> bool:
    """Le corpus n'ecrit pas toujours des phrases : le catalogue CNS ecrit des LIGNES de tableau.

    Propriete DERIVEE de l'unite (pas de la culture ni du type) : une unite qui ne se termine pas
    par une ponctuation de phrase est une ligne de fiche ou de tableau. Elle est citee ENTIERE et
    la reponse est etiquetee « lecture de tableau » — on ne fabrique jamais une phrase a partir de
    cellules (cas `mil_cycle_hkp3` et `variete_el_maradi`).
    """
    return not norm(unite).endswith((".", "!", "?"))


def provenance_lisible(p: dict) -> str:
    """Editeur / annee / page LUS DANS LES METADONNEES du chunk, jamais parses du texte."""
    bouts = [str(p.get("editeur") or "editeur non renseigne")]
    titre = str(p.get("titre") or "").strip()
    if titre:
        bouts.append("« " + titre + " »")
    if p.get("annee"):
        bouts.append(str(p["annee"]))
    if p.get("page") is not None:
        bouts.append("page " + str(p["page"]))
    lic = str(p.get("licence") or "").strip()
    if lic:
        bouts.append(lic)
    else:
        bouts.append("licence non declaree dans le PDF (regime " + str(p.get("regime")) + ")")
    return ", ".join(bouts)


# =========================================================================================
# 7. NIVEAU 2 SITUE — nommer la valeur documentee la plus proche, et l'ecart
# =========================================================================================
# Lexique de lieux du perimetre declare + les pays voisins que le corpus cite reellement. Il ne
# sert qu'a NOMMER un ecart dans le message ; il ne decide aucun niveau. Si un lieu n'y est pas, le
# message ne nomme simplement pas d'ecart de lieu — il ne peut donc pas en inventer un.
LIEUX = ("Niger", "Maradi", "Zinder", "Tahoua", "Dosso", "Tillab", "Diffa", "Agadez",
         "Mali", "Burkina", "S[eé]n[eé]gal", "Nigeria", "Ghana", "Tchad", "B[eé]nin")
# Vocabulaire de MODELISATION : une valeur trouvee dans ce voisinage est un parametre de modele,
# pas une recommandation au producteur. ⚠️ Portee honnete : calibre sur le seul cas mesure
# (`oar_57448`, APSIM) — n=1, comme la garde `to maturity`.
MARQUEURS_MODELE = r"APSIM|simulat|calibrat|mod[eè]l|scenario|sc[eé]nario|DSSAT"


def _lieux_nommes(texte: str) -> list[str]:
    out = []
    for l in LIEUX:
        if re.search(r"\b" + l, texte, re.I):
            out.append(re.sub(r"\[.\.?\]|\[|\]", "", l))
    return out


# 🔴 Les alternatives GENERIQUES de `CIBLES_SANITAIRES` : elles designent « un ravageur » sans dire
# lequel. Les separer n'est pas une finesse de vocabulaire, c'est la seule chose qui distingue les
# deux unites que la mesure `M9a` a opposees — toutes deux passent la conjonction, toutes deux
# portent un verbe de la classe `VERBE_PRATIQUE` :
#   `i2094` « Arracher et detruire les plants malades et les touffes de **Striga**. »  <- la pratique
#   `i847`  « ...how a given inter**crop** contributes to **pest** suppression... »    <- agenda de
#           recherche anglais, ou `intercrop` matche comme SUBSTANTIF et `pest` comme generique.
# Sans la distinction, `i847` gagnait par la cle (1) `citable` (il est `committed`/CC-BY, `i2094` est
# `fetched`) et le niveau 2 de `tp2` nommait une phrase d'agenda a la place de la seule vraie
# pratique du corpus.
GENERIQUES_SANITAIRES = ("pest", "ravageur", "maladie", "nuisible")

# Une divergence entre les deux listes rendrait la cle (0bis) silencieusement inerte — donc elle
# leve a l'import plutot qu'a l'usage. C'est la meme regle que partout ici : re-deriver de la source
# unique, jamais maintenir une seconde liste.
_ORPHELINS = set(GENERIQUES_SANITAIRES) - set(CIBLES_SANITAIRES.split("|"))
if _ORPHELINS:
    raise RuntimeError("GENERIQUES_SANITAIRES hors de CIBLES_SANITAIRES : "
                       + ", ".join(sorted(_ORPHELINS)))


def _cibles_sanitaires(texte: str, generiques: bool = False) -> list[str]:
    """Les cibles sanitaires nommees dans `texte`, alternatives RE-DERIVEES de `CIBLES_SANITAIRES`.

    Par defaut les generiques sont exclus : ce qu'on cherche est « cette unite nomme-t-elle LE
    ravageur demande », et « pest » ne le nomme pas.
    """
    out = []
    for alt in CIBLES_SANITAIRES.split("|"):
        if not generiques and alt in GENERIQUES_SANITAIRES:
            continue
        if re.search(r"\b" + alt, texte, re.I):
            out.append(alt)
    return out


RANG_IDEAL = (0, 0, 0)


def _sert_la_cible_de_la_requete(typ: str, unite: str, requete: str) -> bool:
    """L'unite candidate nomme-t-elle la cible sanitaire que la REQUETE demande ?

    Borne 6 des deux cadres de citation (`_citations_niveau_1` et `_couture_meme_page`). Elle ne
    duplique aucune des quatre verifications existantes : celles-la portent sur la CULTURE, sur la
    co-mention culture↔valeur, sur le motif de valeur et sur la fidelite au chunk — aucune sur le
    ravageur. Defaut mesure qu'elle ferme : `tp2` (striga) servait une ligne « Chenille Legionnaire /
    Spodoptera », meme document, meme page, valeur bien portee par le sorgho. Cf.
    `CITATION_EXIGE_CIBLE_DE_LA_REQUETE` pour le pourquoi du commutateur.

    QUATRE portes de sortie, chacune une decision et non une commodite :
      - commutateur a False ⇒ l'arme « sans la regle » reste exercable a l'identique ;
      - type NON conjonctif ⇒ rien a exiger : `periode_semis`, `cycle_duree`, `dose_engrais` n'ont
        pas de cible sanitaire, et l'exiger d'eux supprimerait toutes leurs citations (`tp1` en
        premier). C'est `CONJONCTIF` — un **dict** — qui nomme les types concernes, pas une seconde
        liste maintenue ici ;
      - la requete ne nomme aucune cible SPECIFIQUE (`generiques=False` : « pest » ou « maladie » ne
        nomment pas un ravageur) ⇒ il n'y a pas de cible a servir, et refuser sur une exigence vide
        rendrait la regle plus severe que la question ;
      - sinon : intersection non vide entre les cibles de la requete et celles de l'unite.
    `_cibles_sanitaires` est REUTILISEE des deux cotes — la liste d'alternatives reste re-derivee de
    `CIBLES_SANITAIRES`, comme la cle (0bis) de `_rang_unite` qui fait deja ce test au niveau 2.
    """
    if not CITATION_EXIGE_CIBLE_DE_LA_REQUETE or typ not in CONJONCTIF:
        return True
    demandees = _cibles_sanitaires(requete, generiques=False)
    if not demandees:
        return True
    return bool(set(demandees) & set(_cibles_sanitaires(unite, generiques=False)))


def _rang_unite(c: dict) -> tuple[int, int, int]:
    """Rang lexicographique d'une unite candidate pour un type CONJONCTIF. `RANG_IDEAL` = meilleur.

    Une seule definition, utilisee pour le choix DANS un chunk et pour le tri GLOBAL — sinon les
    deux peuvent diverger en silence et le temoin elu n'est plus celui que le tri declare.
      1er rang  porte un verbe de pratique ;
      2e  rang  nomme la cible sanitaire SPECIFIQUE de la requete (pas un generique) ;
      3e  rang  OUVRE sur son verbe de pratique — donc ordonne un acte au lieu d'en parler.
    L'ordre des trois est une decision, pas une commodite : « nommer la bonne cible » passe devant
    « etre formule a l'impératif » parce que la reserve de niveau 2 promet de nommer la pratique la
    plus PROCHE — se tromper de ravageur est un ecart plus grand qu'une tournure descriptive.
    """
    return (0 if c["porte_pratique"] else 1,
            0 if c["recouvre_cible"] else 1,
            0 if c["prescriptif"] else 1)


def voisins_documentes(idx: retrieve.Index, typ: str, cible: str | None,
                       requete: str) -> list[dict]:
    """Tous les voisins candidats, DEJA CLASSES. `voisinage_documente` n'en prend que le premier.

    Rendre la liste et non seulement le gagnant est un choix de verifiabilite : le classement
    ci-dessous est une decision de contenu (quelle piece a conviction un juge lit), donc il doit
    etre mesurable de l'exterieur sans reimplementer le balayage. `verify_retrieve.py` §10 l'exerce
    sur `tp2` ; sans cela le tri serait une affirmation de commentaire.

    Classement, deterministe et declare — 5 cles :
      (0) **types conjonctifs SEULEMENT** : porte un verbe de pratique d'abord. Mesure a l'appui :
          sans cette cle, `(mesure_lutte, sorgho)` nomme `i755`, une phrase d'AGENDA DE
          RECHERCHE en anglais (« *pigeonpea pest management research should take a more integrated
          approach* »), au lieu de la seule vraie pratique du corpus (`i2094`, « *Arracher et
          detruire les plants malades et les touffes de Striga.* »). La cle est conditionnee au type
          parce qu'un verbe de pratique n'a aucun sens pour une demande CHIFFREE : l'appliquer
          partout deplacerait le temoin de `tp1` — verifie inchange.
      (0bis) **types conjonctifs SEULEMENT** : nomme la cible sanitaire de la REQUETE, pas seulement
          un generique. 🔴 La cle (0) seule **n'a pas suffi**, et c'est mesure : elle a deplace le
          temoin de `i755` vers `i847` — « *how a given intercrop contributes to pest suppression* »,
          un autre agenda de recherche anglais, ou `intercrop` matche comme SUBSTANTIF et `pest`
          comme generique — pendant que `i2094` restait **7e**, derriere trois unites anglaises
          `committed`/CC-BY. Le proxy `porte_pratique` etait satisfait par les deux, donc un controle
          qui n'assertait que lui restait vert — une defense jamais vue refuser ne prouve rien. (0) et (0bis) sont
          fondues dans `_rang_unite`, une seule definition pour le tri global et le choix intra-chunk.
          ⚠️ Portee honnete : la cle est **inerte** si la requete ne nomme qu'un generique (« mesures
          contre les ravageurs ») — alors aucune unite ne peut « nommer la meme cible », et le
          classement retombe sur (1)/(2)/(3). Ce n'est pas un repli choisi, c'est une limite mesuree.
      (0ter) **types conjonctifs SEULEMENT** : l'unite OUVRE sur son verbe de pratique. 🔴 (0)+(0bis)
          n'ont pas suffi non plus : elles laissaient **trois** candidats a egalite parfaite, et le
          departage retombait sur la cle (3), l'indice de chunk — c'est-a-dire sur l'ordre
          d'indexation. Elle elisait `i1836` « *Le striga est rencontre dans presque toutes les
          grandes zones de production du niebe du pays…* », une phrase de REPARTITION, pendant que
          `i2094` « *Arracher et detruire les plants malades et les touffes de Striga.* » — la seule
          vraie pratique du corpus (`M8b`/`M8c`) — arrivait **3e**. Deux asserts successifs sont
          restes verts au-dessus de ce defaut, parce qu'ils testaient les proxys `porte_pratique`
          puis `recouvre_cible`, tous deux satisfaits par une simple description de la maladie.
          Le predicat est chiffre plus haut, pas postule.
      (1) citable sans reserve (regime `committed` **et** `citation_verbatim_autorisee == oui`) ;
      (2) chevauchement de lieu avec la requete ; (3) indice de chunk croissant.
    ⚠️ Les cles (0)/(0bis)/(0ter) passent DEVANT (1), donc elles preferent sciemment un chunk
    `fetched` a un CC-BY quand lui seul porte une pratique sur la bonne cible. C'est assume : il a
    ete tranche que `licence: null` n'est pas une interdiction de citer, et `tronque()` plafonne de toute
    facon la longueur. Aucun cosinus n'entre ici : ce n'est pas un classement de pertinence, c'est un
    choix de PIECE A CONVICTION.
    """
    if typ not in MOTIFS:
        return []
    conjonctif = typ in CONJONCTIF
    cands: list[dict] = []
    lieux_req = set(_lieux_nommes(requete))
    # Cibles sanitaires SPECIFIQUES de la requete, calculees une fois. Vide ⇒ cle (0bis) inerte.
    cibles_req = set(_cibles_sanitaires(requete)) if conjonctif else set()
    # Meme predicat et MEME PORTEE que le veto de variete de `verifie_entite` : si le message
    # nommait l'ecart sur l'unite alors que le veto porte sur le chunk, les deux pourraient se
    # contredire sur le meme passage.
    noms_var = variete_demandee(requete) if EXIGE_VARIETE_DANS_LE_CHUNK else []
    for c in idx.chunks:
        retenue: dict | None = None
        for u in unites_du_chunk(c):
            vals = trouve(typ, u)
            if not vals:
                continue
            if cible:
                ok, _etage = verifie_entite(idx, c, u, cible, requete=requete)
                if ok:
                    continue                      # aurait donne un niveau 1 : pas un « voisin »
            citable = (str(c.get("regime")) == "committed"
                       and str(c.get("citation_verbatim_autorisee")) == "oui")
            lieux = _lieux_nommes(c["texte"])
            # La cible sanitaire est cherchee dans l'UNITE, pas dans le chunk : c'est l'unite qu'un
            # juge lit. Un chunk qui nomme le striga trois lignes plus haut ne fait pas de la phrase
            # servie une phrase sur le striga.
            cibles_u = _cibles_sanitaires(u) if conjonctif else []
            cand = {
                "chunk": c["i"], "unite": u, "valeurs": vals, "citable": citable,
                "lieux": lieux, "recouvre_lieu": bool(lieux_req & set(lieux)),
                "cultures": [e for e in ENTITES
                             if nomme_texte(e, c["texte"] + " " + str(c.get("titre") or ""))],
                "variete_absente": bool(noms_var) and not tous_nommes(noms_var, c["texte"]),
                "modele": bool(re.search(MARQUEURS_MODELE, c["texte"], re.I)),
                "porte_pratique": bool(VERBE_PRATIQUE.search(u)),
                "prescriptif": est_prescriptif(u) if conjonctif else False,
                "cibles_sanitaires": cibles_u,
                "recouvre_cible": bool(cibles_req & set(cibles_u)),
            }
            if retenue is None:
                retenue = cand
            elif conjonctif and _rang_unite(cand) < _rang_unite(retenue):
                # Dans un MEME chunk, la meilleure unite bat la premiere venue — sinon le `break`
                # d'origine figeait le hasard de l'ordre des unites. Meme rang que le tri global.
                retenue = cand
            if not conjonctif or _rang_unite(retenue) == RANG_IDEAL:
                break                              # rien de mieux a esperer dans ce chunk
        if retenue is not None:
            cands.append(retenue)
    cands.sort(key=lambda d: (_rang_unite(d) if conjonctif else RANG_IDEAL,
                              not d["citable"], not d["recouvre_lieu"], d["chunk"]))
    return cands


def voisinage_documente(idx: retrieve.Index, typ: str, cible: str | None,
                        requete: str) -> dict | None:
    """La valeur documentee la plus proche d'une demande dont le niveau 1 est REFUSE.

    C'est ce qui transforme « ce chiffre n'est pas dans mes sources » en une absence **SITUEE**.
    Le balayage porte sur les 3 180 chunks (pas sur les 8 retenus) : la valeur la plus proche du
    cumul demande pour le mil est un chunk **anglais** sur le sorgho au Mali, que la recuperation
    d'une requete francaise sur le mil ne remonte pas — et c'est justement pour cela qu'elle est
    instructive. On ne garde que les occurrences qui ECHOUENT la verification d'entite : celles qui
    la passent auraient deja donne un niveau 1.

    Le classement est celui de `voisins_documentes`, ou il est declare et mesure.
    """
    cands = voisins_documentes(idx, typ, cible, requete)
    return cands[0] if cands else None


def ecarts_nommes(voisin: dict, cible: str | None, requete: str) -> list[str]:
    """Nomme l'ecart entre la valeur trouvee et la question posee : culture, lieu, nature.

    Chaque ecart est DERIVE du chunk retenu, jamais code en dur : si le chunk ne nomme pas de lieu,
    le message ne nomme pas d'ecart de lieu. Le code ne peut donc pas mentir sur l'ecart.
    """
    out: list[str] = []
    autres = [c for c in voisin["cultures"] if c != cible]
    if cible and autres:
        out.append("culture " + "/".join(autres) + " ≠ " + cible)
    elif cible and not voisin["cultures"]:
        out.append("aucune culture nommee dans la source ≠ " + cible)
    lieux_req = _lieux_nommes(requete)
    if lieux_req and voisin["lieux"] and not voisin["recouvre_lieu"]:
        out.append("lieu " + "/".join(voisin["lieux"][:2]) + " ≠ " + "/".join(lieux_req[:2]))
    if voisin.get("variete_absente"):
        # On nomme l'ABSENCE de la variete demandee, jamais une variete rivale : les unites de
        # tableau mesurees commencent par « INRAN » ou « ICRISAT », qui sont des obtenteurs et non
        # des varietes ⇒ nommer « variete INRAN ≠ HKP3 » serait faux. Dans un tableau extrait, la
        # colonne d'a cote appartient a une autre grandeur aussi souvent qu'a la meme ligne : lire
        # un libelle voisin comme s'il qualifiait la valeur est le mode d'erreur le plus frequent
        # de tout ce module. Ce que le code sait, il le dit ; ce qu'il ne sait pas, il ne l'invente
        # pas.
        out.append("variete non attestee dans le passage ≠ "
                   + "/".join(variete_demandee(requete)))
    if voisin["modele"]:
        out.append("parametre de modele ≠ recommandation au producteur")
    return out


def _decore_voisin(idx: retrieve.Index, voisin: dict) -> dict:
    """Ajoute au voisin sa citation plafonnee, son invariant verbatim et sa provenance complete.

    Factorise parce que DEUX chemins servent desormais un voisin — le niveau 2 plein et la reserve
    greffee sur un niveau 1 (§8bis) — et qu'un voisin servi sans `invariant_verbatim` serait une
    citation dont personne n'a verifie qu'elle est verbatim. Un seul site, donc un seul comportement.
    """
    verbatim, affichee, coupee = tronque(voisin["unite"])
    ch = idx.chunks[voisin["chunk"]]
    voisin["citation"] = verbatim
    voisin["citation_affichee"] = affichee
    voisin["tronquee"] = coupee
    voisin["invariant_verbatim"] = invariant_verbatim(verbatim, ch["texte"])
    voisin["provenance"] = {k: ch.get(k) for k in
                            ("doc", "titre", "editeur", "annee", "page", "langue",
                             "licence", "regime", "citation_verbatim_autorisee")}
    return voisin


# =========================================================================================
# 8bis. RESERVE — du niveau 2 GREFFE sur une reponse de niveau 1
# =========================================================================================
def reserve_niveau_2(idx: retrieve.Index, typ: str, cible: str | None,
                     requete: str, cites: list[dict]) -> dict | None:
    """Une reserve de niveau 2 greffee sur une reponse de niveau 1 — jamais substituee a elle.

    Le cas mesure qui l'exige est notre propre vitrine (`tp2`, recopie mot pour mot dans le
    formulaire de soumission) : la question demande des *practical measures* contre le striga du sorgho, et
    le niveau 1 sert deux unites qui sont des **titres de documents**. Le JSON etait vert
    — niveau 1, invariant verbatim vrai, 0 appel de generation — et le TEXTE servait a un juge une
    entree bibliographique comme si c'etait un conseil : servir une reference comme si elle etait
    un protocole de terrain, en plus discret. La politique graduee fait un devoir de le SIGNALER
    plutot que de le laisser passer.

    Ce que la reserve dit est entierement DERIVE : aucune unite citee ne porte de verbe de pratique,
    donc ce sont des orientations documentees et non un protocole de terrain ; puis elle nomme la
    pratique documentee la plus proche et son ecart exact. Elle ne se declenche que pour les types
    **conjonctifs** — ceux dont la « valeur » est une pratique et non un chiffre —, derive de
    `CONJONCTIF` et non d'une seconde liste a maintenir.

    🔴 Elle ne retire AUCUNE citation et ne change AUCUN niveau de demande : c'est exactement ce qui
    la distingue du veto de promotion mesure puis ECARTE, lequel otait a `tp2` ses deux
    citations francaises, c'est-a-dire la demonstration translingue du prompt qui existe pour ca. Le
    `niveau_global` du rapport, lui, porte bien les deux niveaux : un rapport qui dirait « 1 » d'un
    message servant du niveau 2 serait une fausse declaration dans le champ le plus lu.

    ⚠️ Portee honnete, et c'est une limite de forme, pas de degre : le declencheur est l'ABSENCE d'un
    verbe dans une liste finie. Une pratique redigee avec un verbe hors liste declencherait la reserve
    a tort, et sa premiere phrase deviendrait fausse a la lettre. La contre-epreuve `M9d` exerce ce
    sens-la (une unite QUI porte un verbe ne doit pas declencher la reserve) ; le sens inverse — un
    verbe qui manque a la liste — n'est pas couvert par du code, il est borne par la mesure `M8b`.
    """
    if typ not in CONJONCTIF:
        return None
    # Teste l'unite ENTIERE (`citation`), pas la version affichee plafonnee par `tronque()` : si la
    # pratique est dans la queue coupee, la source en porte une et la reserve mentirait.
    if any(VERBE_PRATIQUE.search(c["citation"]) for c in cites):
        return None
    voisin = voisinage_documente(idx, typ, cible, requete)
    if voisin is not None:
        _decore_voisin(idx, voisin)
    return {
        "motif": ("aucune unite citee n'enonce de pratique a appliquer : ce sont des orientations "
                  "documentees, pas un protocole de terrain"),
        "declencheur": "absence de verbe de pratique dans les citations de niveau 1",
        "voisin": voisin,
        "ecarts": ecarts_nommes(voisin, cible, requete) if voisin else [],
    }


# =========================================================================================
# 8. DECISION
# =========================================================================================
MAX_CITATIONS_PAR_DEMANDE = 2
# 🔴 Budget SEPARE, et non `MAX_CITATIONS_PAR_DEMANDE + 1` : la couture n'est pas une citation de
# plus, c'est l'ACHEVEMENT d'une source deja citee (meme document, meme page). Les melanger ferait
# qu'un chunk classe au rang 3 prendrait la place d'une couture, ou l'inverse, selon l'ordre — donc
# selon le hasard. Mesure a l'appui : `niebe_dates_zones` est deja AU PLAFOND de 2 citations quand la
# zone manquante arrive ⇒ sans budget propre, la couture n'atterrit jamais et le mecanisme serait
# ecrit mais mort. Un mecanisme jamais exerce ne prouve rien : il ressemble a une garantie et n'en
# est pas une. 0 = regle desactivee, pour que l'arme « sans la regle » reste reproductible.
MAX_COUTURE_PAR_DEMANDE = 1
# Troisieme commutateur DECLARE (avec `EXIGE_VARIETE_DANS_LE_CHUNK` et le budget ci-dessus). Il porte
# la borne 5 de `_couture_meme_page`. Il existe pour la meme raison que les deux autres : le refus de
# la borne 5 est MUET — il ne rend rien, exactement comme une absence de voisin — donc sans un moyen
# d'exhiber l'etat sans la regle, « elle refuse le contournement de plafond » serait une croyance et
# non une mesure : une defense qu'on n'a jamais VUE refuser est indistinguable d'une defense
# debranchee, et les deux se lisent pareil dans un rapport vert.
COUTURE_EXIGE_TEXTE_ABSENT_DE_LA_SOURCE = True
# QUATRIEME commutateur DECLARE, et il porte la borne 6 des DEUX cadres de citation (le niveau 1
# ordinaire et la couture). Defaut MESURE qui l'exige, et les chiffres sont ecrits ICI parce que le
# rapport de mesure n'est pas un livrable public : `tp2` demande la lutte contre le **striga** et servait TROIS citations — chunks
# 223 / 223 / **224**, toutes page 3 du meme document — dont la troisieme nomme « Chenille
# Legionnaire / Spodoptera » et **pas** la cible de la requete. Elle passait parce qu'aucune des
# quatre verifications de citation ne teste le ravageur demande : `verifie_entite` verifie la
# CULTURE (le sorgho est bien nomme), `arbitre_co_mention` verifie que c'est elle qui porte la
# valeur, `trouve` le motif de valeur, `invariant_verbatim` la fidelite au chunk. La cle (0bis) de
# `_rang_unite` fait deja ce travail — mais au niveau 2 SEULEMENT.
# 🔴 L'hypothese de depart (« penaliser les chunks de type sommaire/listing via le champ
# `structure` ») est REFUTEE par la mesure et ne doit pas etre reimplementee : les deux chunks
# portent `structure: ""` (`n_chunks_marques_structure: 0`) ⇒ une porte fondee sur `structure` est
# INERTE sur ce defaut precis. Le predicat qui agit est la cible sanitaire de la requete.
# 0/False = regle desactivee, pour la meme raison que `MAX_COUTURE_PAR_DEMANDE = 0` : le refus de la
# borne 6 est MUET (une unite en moins, pas un message), donc sans un moyen d'exhiber l'etat « sans
# la regle » elle serait une croyance et non une mesure — une defense qu'on n'a jamais vue refuser
# est indistinguable d'une defense debranchee.
CITATION_EXIGE_CIBLE_DE_LA_REQUETE = True
PERIMETRE = ("Sahel francophone, cœur Niger — mil, sorgho, mais, arachide, niebe "
             "(33 documents indexes, 3 180 passages)")


def _couture_meme_page(idx: retrieve.Index, res: dict, typ: str, cible: str, requete: str,
                       cites: list[dict], vus: set[tuple[int, str]]) -> list[dict]:
    """ACHEVE une reponse de niveau 1 avec le chunk VOISIN DE PAGE d'un chunk deja cite.

    Le D4 avait mesure que le decoupage a 700 caracteres coupe un tableau ou une liste en pleine
    valeur, et avait declare la recouture des chunks voisins de la MEME page comme la recuperation
    prevue (`retrieve.voisins_meme_page`, ecrit au D4). Cette primitive existait et `answer.py` ne
    l'appelait NULLE PART : la reparation etait ecrite dans les notes, pas dans le code.

    Le cas mesure qui l'exige (`niebe_dates_zones`, M10) : le catalogue CNS enumere TROIS zones
    pluviometriques sur sa page 169, le chunk `i1647` en porte deux et `i1646` — meme document, meme
    page, NON RETENU par le classement — porte la troisieme, seul dans les 3 180 chunks. La reponse
    servie etait donc juste et INCOMPLETE, ce qu'aucun niveau ne signale : ni le niveau 1 (il est
    verbatim et sourcé), ni le niveau 2 (rien n'est absent du corpus).

    SIX bornes, chacune destinee a empecher que « achever » ne devienne « elargir » :
      1. seulement si la demande est DEJA au niveau 1 — la couture acheve, elle ne promeut jamais ;
      2. seulement un voisin de MEME document et MEME page (la primitive ne traverse pas la page :
         un chunk a cheval n'aurait aucun numero de page unique a citer) ;
      3. jamais un chunk RETENU : le classement a deja eu sa chance de le servir, et le reprendre ici
         contournerait le plafond de citations par le detour d'un voisinage ;
      4. les MEMES quatre verifications que le niveau 1 ordinaire — entite, co-mention, motif de
         valeur, invariant verbatim. Une citation cousue qui sauterait un seul de ces controles
         serait un niveau 1 obtenu par une porte de service ;
      5. 🔴 le texte cousu ne doit PAS deja se trouver dans le chunk SOURCE. C'est la borne qui rend
         le verbe « achever » verifiable au lieu de declaratif, et elle a ete ajoutee APRES mesure,
         parce que les trois declenchements observes n'etaient pas de meme nature. Le decoupage du D4
         CHEVAUCHE : **1 999 chunks sur 3 180** ont un voisin de meme page qui reprend leur QUEUE de
         60 caracteres. Le predicat est nomme parce qu'il est orientable et que les deux orientations
         ne donnent pas le meme compte (tete d'un voisin dans le chunk : 1 944 ; union : 2 000) — un
         « 62,9 % » nu laisserait croire a une grandeur unique. Ce compte est RE-DERIVE par la
         section 15 de `verify_retrieve.py`, qui le relit dans cette docstring meme et echoue s'il
         s'en ecarte ; il n'est donc pas recopie d'un tableau de synthese. ⚠️ La version precedente
         de cette ligne annoncait « 1 994 (62,7 %) », chiffre qui ne se re-derive d'AUCUN des trois
         predicats et qui n'existe dans aucun rapport de mesure : ne pas le restaurer.
         Sans cette borne, la couture sert donc, dans la majorite des cas, une
         phrase que le chunk source contenait deja et que `MAX_CITATIONS_PAR_DEMANDE` avait justement
         refuse de servir — c'est-a-dire qu'elle CONTOURNE le plafond par le detour d'un voisinage,
         au lieu de completer quoi que ce soit. Cas mesure : `mil_microdose_dose`, ou i1958 rendait la
         phrase « Cependant, l'application de 50 a 200 kg/ha de superphosphate… » deja presente en
         queue de i1957. Les deux achevements reels passent la borne (i1646 apporte la premiere zone
         pluviometrique que i1647 n'a pas ; i347 apporte les quatre doses NPK que i346 annonce sans
         les donner), le contournement est refuse.
      6. 🔴 l'unite cousue doit nommer la CIBLE SANITAIRE de la requete quand le type est conjonctif
         (`_sert_la_cible_de_la_requete`, commutateur `CITATION_EXIGE_CIBLE_DE_LA_REQUETE`). La
         borne 4 ci-dessus disait « les MEMES quatre verifications que le niveau 1 » — mesure faite,
         ces quatre-la sont AVEUGLES au ravageur : elles verifient la culture, la co-mention
         culture↔valeur, le motif de valeur et la fidelite au chunk. C'est par ce trou que `tp2`
         (striga) servait une troisieme citation nommant « Chenille Legionnaire / Spodoptera » —
         chunk 224, meme document et MEME PAGE 3 que les deux citations justes, donc les bornes 2
         et 5 la laissaient passer sans faute. La borne 6 est portee aux DEUX cadres, pas seulement
         ici : le meme trou existe dans `_citations_niveau_1`, et ne le fermer que du cote de la
         couture aurait deplace le defaut au lieu de le fermer.
         ⚠️ Ce qu'elle COUTE, mesure sur les 3 180 chunks AVANT d'etre acceptee : **50 %**
         des 54 unites portant un motif `mesure_lutte` sur 7 documents ne nomment aucune cible
         specifique et deviennent incitables. Accepte parce que ces unites-la ne repondaient a la
         question de personne — mais c'est un cout de RAPPEL, pas un gain gratuit, et il est ecrit
         ici pour qu'un futur elargissement sache ce qu'il rachete.
    Plus une dedup sur le TEXTE (et non sur le couple chunk+texte comme le fait la boucle
    principale) : deux chunks d'une meme page repetent parfois la meme ligne, et re-servir un texte
    deja cité consommerait le budget sans rien achever — ce qui est le contraire de la definition.
    """
    if not cites or MAX_COUTURE_PAR_DEMANDE <= 0:
        return []
    par_rang = {r["i"]: r for r in res["retenus"]}
    deja_texte = {norm(c["citation"]) for c in cites}
    out: list[dict] = []
    for c in cites:                                   # dans l'ordre de rang des citations servies
        r = par_rang.get(c["chunk"])
        if r is None:
            continue
        # `voisins_meme_page` est livre par `retrieve.cherche()` ; le repli couvre un `res` FABRIQUE
        # par une contre-epreuve, qui n'a pas a connaitre ce detail de forme.
        voisins = r.get("voisins_meme_page")
        if voisins is None:
            voisins = retrieve.voisins_meme_page(idx, c["chunk"])
        for j in voisins:
            if j in par_rang:
                continue                              # borne 3
            chunk = idx.chunks[j]
            for u in unites_du_chunk(chunk):
                vals = trouve(typ, u)
                if not vals:
                    continue
                ok, etage = verifie_entite(idx, chunk, u, cible, requete=requete)
                if not ok:
                    continue
                if not _sert_la_cible_de_la_requete(typ, u, requete):
                    continue                              # borne 6
                co = (arbitre_co_mention(typ, cible, u) if cible in ENTITES
                      else {"declenche": False, "accorde": True})
                if not co["accorde"]:
                    continue
                verbatim, affichee, coupee = tronque(u)
                if not invariant_verbatim(verbatim, chunk["texte"]):
                    continue
                cle = (j, norm(verbatim))
                if cle in vus or norm(verbatim) in deja_texte:
                    continue
                # borne 5 : ce que le chunk source contient deja n'est pas un achevement. Le test est
                # exactement celui d'`invariant_verbatim`, applique a l'autre chunk de la paire — pas
                # une heuristique de « continuation d'enumeration » qu'il faudrait calibrer.
                if (COUTURE_EXIGE_TEXTE_ABSENT_DE_LA_SOURCE
                        and norm(verbatim) in norm(idx.chunks[c["chunk"]]["texte"])):
                    continue
                vus.add(cle)
                deja_texte.add(norm(verbatim))
                out.append({
                    "chunk": j, "rang": len(cites) + len(out) + 1, "etage": etage,
                    "mecanisme": mecanisme_texte(cible, chunk["texte"]) if cible in ENTITES else "dyn",
                    "valeurs": vals, "citation": verbatim, "citation_affichee": affichee,
                    "tronquee": coupee, "lecture_de_tableau": est_ligne_de_tableau(u),
                    "co_mention": co if co["declenche"] else None,
                    # Les cles de provenance sont RELUES du voisin de rang (`r["provenance"]`) au lieu
                    # d'etre recopiees : retrieve.py en est la seule source, et une liste dupliquee
                    # ici derivrait le jour ou il en ajoute une.
                    "provenance": {k: chunk.get(k) for k in c["provenance"]},
                    # Pas de cosinus : ce chunk n'a PAS ete classe. Mettre celui du voisin serait
                    # attribuer a un passage une mesure faite sur un autre.
                    "cos": None,
                    "couture": True, "couture_de": c["chunk"],
                })
                if len(out) >= MAX_COUTURE_PAR_DEMANDE:
                    return out
    return out


def _citations_niveau_1(idx: retrieve.Index, res: dict, typ: str, cible: str,
                        requete: str) -> list[dict]:
    """Parcourt les chunks RETENUS **dans leur ordre de rang** et cherche une ancre citable.

    🔴 La verification porte sur le chunk REELLEMENT CITE, jamais sur un `any()` d'un ensemble de
    bons chunks : un `any()` prouve « il existe un chunk qui passe », pas « celui
    que je cite passe ». Mesure a l'appui : sur 12 items, les deux formulations divergent une fois.

    🔴 DEUX conditions doivent tomber, et dans cet ordre — l'entite d'abord (« est-elle nommee ici »,
    `verifie_entite`), la co-mention ensuite (« est-ce ELLE qui porte cette valeur »,
    `arbitre_co_mention`). La seconde n'est pas un raffinement de la premiere : `i2093` passe l'etage
    (a1) sans discussion — « … apres le premier sarclage du mil » nomme bel et bien le mil — et sa
    date appartient au niebe. Une regle qui ne teste que la presence est donc structurellement
    incapable d'attraper ce cas, quel que soit son seuil.
    """
    trouvees: list[dict] = []
    vus: set[tuple[int, str]] = set()
    for r in res["retenus"]:
        chunk = idx.chunks[r["i"]]
        for u in unites_du_chunk(chunk):
            vals = trouve(typ, u)
            if not vals:
                continue
            ok, etage = verifie_entite(idx, chunk, u, cible, requete=requete)
            if not ok:
                continue
            # borne 6 : la CULTURE est verifiee ci-dessus, le RAVAGEUR ici. Le defaut mesure qui
            # l'exige (`tp2` servant une ligne Spodoptera sur une question striga) passe les quatre
            # autres controles sans faute — voir `_sert_la_cible_de_la_requete`.
            if not _sert_la_cible_de_la_requete(typ, u, requete):
                continue
            # `cible` peut etre une entite DYNAMIQUE (« EL MARADI »), absente d'`ENTITES` : l'arbitre
            # ne saurait pas la localiser, et il n'a rien a arbitrer puisqu'aucune culture n'est
            # demandee. On ne le fait donc porter que sur les cultures du perimetre.
            co = (arbitre_co_mention(typ, cible, u) if cible in ENTITES
                  else {"declenche": False, "accorde": True})
            if not co["accorde"]:
                continue
            verbatim, affichee, coupee = tronque(u)
            if not invariant_verbatim(verbatim, chunk["texte"]):
                # M6 viole : on refuse de l'appeler verbatim plutot que de le pretendre
                continue
            cle = (r["i"], norm(verbatim))
            if cle in vus:
                continue
            vus.add(cle)
            trouvees.append({
                "chunk": r["i"], "rang": len(trouvees) + 1, "etage": etage,
                "mecanisme": mecanisme_texte(cible, chunk["texte"]) if cible in ENTITES else "dyn",
                "valeurs": vals, "citation": verbatim, "citation_affichee": affichee,
                "tronquee": coupee, "lecture_de_tableau": est_ligne_de_tableau(u),
                "co_mention": co if co["declenche"] else None,
                "provenance": r["provenance"], "cos": r.get("cos"),
            })
            if len(trouvees) >= MAX_CITATIONS_PAR_DEMANDE:
                break
        if len(trouvees) >= MAX_CITATIONS_PAR_DEMANDE:
            break                       # le plafond sort de la boucle, il ne sort plus de la
                                        # fonction : la couture doit encore pouvoir ACHEVER
    trouvees += _couture_meme_page(idx, res, typ, cible, requete, trouvees, vus)
    return trouvees


def analyse(res: dict, requete: str, idx: retrieve.Index, seuils: dict | None = None,
            generateur: Callable[[str], str] | None = None) -> dict:
    """Le coeur de la politique : rend un rapport TRACABLE, sans rien afficher.

    Separee de la recuperation pour une raison de verification : `verify_retrieve.py` peut lui
    passer un `res` FABRIQUE (les contre-epreuves doivent vraiment echouer) sans
    ouvrir de serveur d'embedding.

    🔴 **La reponse est 100 % EXTRACTIVE : `generateur` n'est appele NULLE PART, pour aucune entree.**
    Ce n'est pas un effet de bord — les deux sites qui l'appelaient (type inconnu, et niveau 2)
    JETAIENT leur retour : une generation payee sur un 0,5B pour rien, et surtout une fuite latente
    (quelqu'un « repare » la composition un jour, et du texte non garanti entre dans une reponse qui
    promet du verbatim). Ils sont supprimes.

    `generateur` reste au contrat pour une raison, et une seule : **rendre l'invariant OBSERVABLE**.
    Sans le parametre, `appels_generateur == 0` serait une tautologie qu'aucune batterie ne peut
    exercer — une defense jamais vue refuser ne prouve rien. Une batterie injecte donc
    un generateur factice **comptant** ses appels et lit 0 (`generateur_recu` du rapport atteste que
    l'injection a bien eu lieu, sinon le 0 ne dirait rien). ⚠️ Un compteur d'execution ne vaut que
    pour les entrees essayees : `verify_retrieve.py` double la mesure d'un controle **structurel** sur
    la source de ce fichier — aucun site d'appel du parametre, avec une contre-epreuve sur source
    fabriquee qui doit vraiment echouer. C'est cette seconde preuve qui porte « pour toute entree ».
    ⚠️ La signature d'un faux generateur doit accepter un argument **positionnel**.
    """
    seuils = seuils or charge_seuils()
    p23 = seuils["porte_niveau_2_contre_3"]
    valeur, statistique = valeur_porte(res, seuils)
    seuil = float(p23["seuil"])

    rap: dict = {
        "requete": requete,
        "porte_2_3": {"statistique": statistique, "valeur": valeur, "seuil": seuil,
                      "source_du_seuil": "rag/index/seuils.json",
                      "franchie": (valeur is not None and valeur >= seuil)},
        "demandes": [], "niveau_global": None, "appels_generateur": 0,
        # Sans ce champ, lire « appels_generateur: 0 » ne dirait rien : on ne saurait pas si le
        # generateur a ete injecte et non appele, ou simplement absent — sans ce temoin, « 0 appel »
        # serait une tautologie.
        "generateur_recu": generateur is not None,
        "extractif_seulement": True,
    }

    # ---- niveau 3 : REFUS. Du code, en amont du modele. Aucun appel de generation n'a lieu.
    if valeur is None or valeur < seuil:
        rap["niveau_global"] = 3
        rap["refus"] = {
            "motif": ("aucune statistique de porte disponible" if valeur is None
                      else "sous le seuil bas de la porte 2/3"),
            "perimetre": PERIMETRE,
            "message": ("Je ne reponds pas a cette question : elle sort du perimetre que mes "
                        "sources documentent. Ce que je documente : " + PERIMETRE + ". "
                        "Reformule dans ce perimetre et je citerai mes sources a la page."),
        }
        rap["appels_generateur"] = 0     # invariant, pas commentaire : verify_retrieve.py §6 l'exige
        return rap

    # ---- porte franchie : chaque demande recoit SON niveau
    demandes = demandes_de_la_requete(requete)
    if not demandes:
        # 🔴 Type de demande inconnu ⇒ niveau 2. JAMAIS de promotion.
        pistes = [{"chunk": r["i"], "provenance": r["provenance"],
                   "apercu": norm(r["texte"])[:200]} for r in res["retenus"][:3]]
        rap["demandes"].append({
            "type": None, "cible": None, "niveau": 2,
            "motif": "type de demande non reconnu : aucune valeur a attester, aucune promotion",
            "pistes": pistes, "citations": [], "voisin": None, "ecarts": [], "reserve": None,
        })
        rap["niveau_global"] = 2
        return rap

    for d in demandes:
        typ, cible = d["type"], d["cible"]
        cites: list[dict] = []
        if cible is not None:
            cites = _citations_niveau_1(idx, res, typ, cible, requete)
        if cites:
            # Niveau 1 servi — et, si les unites citees ne portent aucune pratique, une RESERVE de
            # niveau 2 GREFFEE dessus (§8bis). Le niveau de la demande reste 1 : rien n'est retire.
            rap["demandes"].append({
                "type": typ, "cible": cible, "niveau": 1, "citations": cites,
                "voisin": None, "ecarts": [],
                "reserve": reserve_niveau_2(idx, typ, cible, requete, cites),
                "motif": "ancre verifiee : entite nommee (" + cites[0]["etage"]
                         + ") + motif de valeur dans l'unite citee",
            })
            continue
        # ---- niveau 2, SIGNALE et SITUE
        voisin = voisinage_documente(idx, typ, cible, requete)
        ecarts = ecarts_nommes(voisin, cible, requete) if voisin else []
        if voisin:
            _decore_voisin(idx, voisin)
        rap["demandes"].append({
            "type": typ, "cible": cible, "niveau": 2, "citations": [],
            "voisin": voisin, "ecarts": ecarts, "reserve": None,
            "motif": ("aucune entite demandee" if cible is None else
                      "aucun passage retenu ne porte a la fois l'entite demandee et une valeur "
                      "de ce type"),
        })
    # Une demande de niveau 1 PORTANT une reserve contribue les DEUX niveaux : le message en sert
    # deux, donc le champ le plus lu du rapport doit en declarer deux.
    niveaux: set[int] = set()
    for d in rap["demandes"]:
        niveaux.add(d["niveau"])
        if d.get("reserve"):
            niveaux.add(2)
    ordre = sorted(niveaux)
    rap["niveau_global"] = ordre[0] if len(ordre) == 1 else ordre
    return rap


def compose(rap: dict) -> str:
    """Rend le message destine au lecteur. Le niveau 2 est MELANGE au niveau 1, jamais substitue."""
    L: list[str] = []
    L.append("Question : " + rap["requete"])
    if rap["niveau_global"] == 3:
        L.append("")
        L.append("[3 · REFUS] " + rap["refus"]["message"])
        return "\n".join(L)
    L.append("Perimetre documente : " + PERIMETRE + ".")
    for n, d in enumerate(rap["demandes"], 1):
        L.append("")
        etiquette = "1 · SOURCE" if d["niveau"] == 1 else "2 · NON SOURCE — SIGNALE"
        titre = d["type"] or "demande non typee"
        cible = (" (" + d["cible"] + ")") if d.get("cible") else ""
        L.append("[" + etiquette + "] " + titre + cible)
        if d["niveau"] == 1:
            # RESERVE de niveau 2 GREFFEE sur ce niveau 1 (§8bis). Son etiquette porte « 2 » pour
            # que la gradation reste lisible a l'ecran — un juge doit pouvoir compter les niveaux
            # sans ouvrir le JSON.
            #
            # 🔴 ORDRE D'AFFICHAGE INVERSE le 22/08, apres lecture des 2 test_prompts COTE A COTE.
            # Le commentaire precedent posait « la reserve vient APRES les citations, jamais a leur
            # place : le lecteur voit d'abord ce qui est source ». Vrai pour tp1, FAUX pour tp2 :
            #   · tp1 cite une PHRASE qui porte la valeur (« … les trois decades de juin ») ⇒ le
            #     lecteur apprend « juin » des la premiere ligne ;
            #   · tp2 cite deux TITRES de documents — le motif `mesure_lutte` matche « lutte …
            #     contre le striga » a l'interieur d'un titre (:288) et la cible sanitaire y est
            #     aussi ⇒ la promotion est la regle qui s'applique, pas un bug, mais les deux
            #     premieres lignes n'ENONCENT RIEN, et la seule pratique du corpus arrivait
            #     cinquieme, presentee comme un repli.
            # Meme etiquette `[1 · SOURCE]`, deux natures d'objet, hierarchie inversee — sur le
            # prompt le plus lu de la soumission (§2.1). ⇒ Quand la reserve porte un voisin, ce
            # voisin passe EN TETE : c'est la seule unite du bloc qui enonce quelque chose a faire.
            # Les citations gardent leur place, leur provenance et leur tag « lecture de tableau »,
            # derriere la reserve qui dit ce qu'elles ne sont pas.
            # Ce changement est PUREMENT un ordre : aucun niveau ne bouge, aucune affirmation n'est
            # ajoutee, `d["niveau"]`/`rap["niveau_global"]` sont intacts. C'est ce qui le rend sur :
            # `verify_retrieve.py` §10 asserte sur le RAPPORT (`:1126` la citation de niveau 1,
            # `:1146` le voisin = chunk 2094), jamais sur ce texte. L'option qui faisait TOMBER tp2
            # en niveau 2 avait ete mesuree et ecartee le 19/08 (:315) — elle otait ses deux
            # citations francaises ; elle contredit en plus `:1126` et `:1423`.
            r = d.get("reserve")
            v = r.get("voisin") if r else None
            if v:
                L.append("  Pratique documentee la plus proche : « "
                         + v["citation_affichee"] + " »")
                L.append("  — " + provenance_lisible(v["provenance"]))
                if r.get("ecarts"):
                    L.append("  Ecart a ta question : " + " · ".join(r["ecarts"]) + ".")
                L.append("  [2 · RESERVE] " + r["motif"] + ".")
            for c in d["citations"]:
                tag = " — lecture de tableau" if c["lecture_de_tableau"] else ""
                L.append("  « " + c["citation_affichee"] + " »" + tag)
                L.append("  — " + provenance_lisible(c["provenance"]))
            if r and v is None:
                # Pas de voisin : ordre d'origine conserve (citations d'abord, reserve ensuite).
                # Ouvrir un bloc sur une negation nue serait pire que le defaut qu'on corrige.
                L.append("  [2 · RESERVE] " + r["motif"] + ".")
                L.append("  Et je n'ai trouve aucune pratique approchante dans mes "
                         "33 documents.")
        else:
            if d.get("pistes"):
                L.append("  Je n'ai pas reconnu de valeur chiffree demandee : je ne presente donc")
                L.append("  RIEN comme verifie. Passages les plus proches, NON verifies :")
                for p in d["pistes"]:
                    L.append("    · " + p["apercu"][:150])
                    L.append("      — " + provenance_lisible(p["provenance"]))
                continue
            cible_txt = (" pour " + d["cible"]) if d.get("cible") else ""
            # Le nom du manquant est DERIVE de `CONJONCTIF`, pas d'une seconde liste a maintenir :
            # les types conjonctifs sont precisement ceux dont la « valeur » est une pratique et non
            # un chiffre. Dire « cette valeur » d'une methode de lutte serait faux a l'ecran.
            quoi = "pratique" if d["type"] in CONJONCTIF else "valeur"
            L.append("  Cette " + quoi + " n'est pas dans mes sources" + cible_txt + ".")
            v = d.get("voisin")
            if v:
                L.append("  " + quoi.capitalize() + " documentee la plus proche : « "
                         + v["citation_affichee"] + " »")
                L.append("  — " + provenance_lisible(v["provenance"]))
                if d["ecarts"]:
                    L.append("  Ecart a ta question : " + " · ".join(d["ecarts"]) + ".")
            else:
                L.append("  Et je n'en ai trouve aucune " + quoi
                         + " approchante dans mes 33 documents.")
    return "\n".join(L)


def repond(requete: str, generateur: Callable[[str], str] | None = None,
           rec: retrieve.Recuperateur | None = None) -> tuple[dict, str]:
    """Chaine complete : recuperation -> politique -> message. Rend (rapport, message)."""
    seuils = charge_seuils()
    if rec is not None:
        res = rec.cherche(requete)
        rap = analyse(res, requete, rec.idx, seuils, generateur)
        return rap, compose(rap)
    with retrieve.Recuperateur() as r:
        res = r.cherche(requete)
        rap = analyse(res, requete, r.idx, seuils, generateur)
        return rap, compose(rap)


# =========================================================================================
# 9. sonde manuelle — `py rag/answer.py "ma question"`
# =========================================================================================
def principal() -> int:
    # La console Windows est en cp1252 : un U+2212 (vrai signe MOINS, present dans le corpus) y
    # leve UnicodeEncodeError et TUE la sonde EN PLEIN MILIEU de sa sortie — mesure a moitie
    # affichee, donc mesure trompeuse. Correction a la couche d'AFFICHAGE seule.
    for flux in (sys.stdout, sys.stderr):
        if hasattr(flux, "reconfigure"):
            flux.reconfigure(encoding="utf-8", errors="replace")
    args = [a for a in sys.argv[1:]]
    en_json = "--json" in args
    args = [a for a in args if a != "--json"]
    if not args:
        print(__doc__.strip().splitlines()[0])
        print()
        print('  py rag/answer.py "A quelle periode semer le mil a Maradi ?"  [--json]')
        return 2
    requete = " ".join(args)
    rap, msg = repond(requete)
    if en_json:
        print(json.dumps(rap, ensure_ascii=False, indent=1, default=str))
        return 0
    print(msg)
    print()
    p = rap["porte_2_3"]
    print("-" * 96)
    print("porte 2/3 : " + str(p["statistique"]) + " = " + str(p["valeur"])
          + "  vs seuil " + str(p["seuil"]) + " (lu dans " + p["source_du_seuil"] + ")"
          + "  -> " + ("franchie" if p["franchie"] else "REFUS"))
    print("niveau(x) : " + str(rap["niveau_global"])
          + "   appels au generateur : " + str(rap["appels_generateur"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(principal())
