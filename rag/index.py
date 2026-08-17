"""Bloc D4 — batit `rag/index/` : chunking + BM25 + embeddings BGE-M3, sur les 33 documents
de `corpus/txt/`.

Regles du projet qui contraignent ce fichier, et comment elles sont tenues ici :
  * regle 4 « llama.cpp only » -> embeddings par `llama-server --embedding` (embed_server.py),
    jamais sentence-transformers. numpy ne sert qu'a stocker des flottants.
  * regle 3 « 100 % offline » -> l'index est prebati et committe EN ENTIER (CLAUDE.md 7.11,
    tranche le 17/08 : 7 documents CC-BY-4.0 confirmee + 26 `licence: null`, provenance
    declaree dans REPORT.md). Aucun acces reseau ici hors 127.0.0.1.
  * le PERIMETRE se derive des CHAMPS de sources.yaml (`utilite_conseil`, `exclusion`), jamais
    d'un id code en dur — c'est ce qui a permis d'ecarter `oar_57215` (police cassee) sans
    toucher au code. Le compte derive est ensuite VERROUILLE a 33 : un ecart doit etre une
    decision, pas un effet de bord.
  * chaque chunk porte sa provenance de citation EN LIGNE (id, titre, editeur, annee, langue,
    licence, regime, page) : c'est exactement ce que ChukwumaUk ne fait pas (RAG committe,
    REPORT.md citant 0 source et 0 licence).

Deux proprietes tenues par construction, parce que la verification en depend :
  1. un chunk est un EXTRAIT CONTIGU du .txt source, aux offsets declares — aucune
     normalisation, aucun recollage. `texte_source[off0:off1] == chunk["texte"]` exactement.
     C'est ce qui rend la citation verbatim demontrable, et pas seulement plausible.
  2. un chunk ne TRAVERSE JAMAIS un marqueur `[[page N]]`. Un chunk a cheval sur deux pages
     n'aurait aucun numero de page unique a citer, et son texte ne serait plus contigu dans la
     source (le marqueur s'intercale). Les marqueurs eux-memes ne sont jamais indexes (bruit
     lexical dans BM25).

Aucun seuil cosinus n'est ecrit ici. Les DEUX seuils de la politique graduee (CLAUDE.md 7.9)
se calibrent au D5 sur une distribution mesuree : une valeur absolue de cosinus ne veut rien
dire hors de son pooling (cls 0,6578/0,3232 contre mean 0,8274/0,6476, PREUVES.md 17.5.1).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import pathlib
import re
import subprocess
import sys
import unicodedata

import numpy as np
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import embed_server  # noqa: E402
from embed_server import Serveur  # noqa: E402

RACINE = pathlib.Path(__file__).resolve().parent.parent
SOURCES = RACINE / "corpus" / "sources.yaml"
TXT = RACINE / "corpus" / "txt"
SORTIE = RACINE / "rag" / "index"
CACHE = RACINE / "tools_corpus" / "_embed_cache"      # hors index committe, git-ignore

SEPARATEUR_CORPS = "=== TEXTE ==="
RE_MARQUEUR = re.compile(r"^\[\[page (\d+)\]\][ \t]*$", re.M)

# ---- perimetre : derive des CHAMPS, puis verrouille -------------------------------------
N_SOURCES_ATTENDU = 39
N_METHODOLOGIQUE_ATTENDU = 5
N_EXCLUSION_ATTENDU = 1
N_DOCUMENTS_ATTENDU = 33

# ---- chunking ---------------------------------------------------------------------------
# Valeurs par defaut = celles RETENUES PAR LA MESURE, pas par l'usage (le brief interdit
# explicitement de « choisir 512 parce que c'est l'usage »). Balayage 500->1800 puis mesure
# de cosinus sur 5 requetes reelles, 6519 embeddings — PREUVES.md 18.
# Trois quantites ont decide, toutes dans le meme sens :
#   1. DILUTION — `marge_medianes` (pertinents moins hors-sujet) : 700 -> +0,0803 ;
#      1100 -> +0,0654 (-19 %) ; 1500 -> +0,0607 (-24 %). Decroissante sur 5/5 requetes.
#   2. INTEGRITE de la phrase de conseil : 0,00 % de phrases coupees des 700 (0,14 % a 500)
#      => rien n'est gagne a grossir au-dela.
#   3. BRUIT STRUCTUREL au rang 1 : 2 requetes sur 5 a 1100 (une bibliographie, une ligne de
#      mots-cles), 0 sur 5 a 700. Les petits chunks separent d'eux-memes la matiere du
#      paratexte.
# Contre-argument honnete, et pourquoi il ne l'emporte pas : `ancrage_cultural_pct` (le conseil
# chiffre est-il dans le meme chunk que le nom de la culture ?) monte avec la taille — 37,75 %
# a 700 contre 47,72 % a 1100. Mais il monte SANS GENOU jusqu'a 1800 : c'est en partie un
# artefact mecanique de longueur (un chunk plus long contient plus de mots quels qu'ils
# soient). Et le deficit est RECUPERABLE a la reponse, parce qu'un chunk est une tranche
# contigue a offsets absolus avec 1 phrase de chevauchement : le D5 recoud des chunks voisins
# de la MEME page depuis l'index. Une marge perdue, elle, ne se recoud pas.
CIBLE = 700
MAXI = 955           # 1,3636 x cible — rapport tenu constant sur tout le balayage
MINI = 220           # sous ce volume un chunk est un en-tete de page, pas un passage
CHEVAUCHEMENT = 1    # en unites (phrases), pas en caracteres
PLANCHER_PAGE = 40   # page dont le contenu utile est sous ce seuil = numero de page seul

# ---- format de stockage des vecteurs ----------------------------------------------------
# Decision n 2 du brief, MESUREE (probe_chunking.py --quant 700), pas supposee :
#   float32  13,03 Mo | err cos max 0        | top1 5/5 | recouvrement top10 1,0000
#   float16   6,51 Mo | err cos max 5,2e-05  | top1 5/5 | recouvrement top10 1,0000
#   int8      3,26 Mo | err cos max 8,7e-03  | top1 5/5 | recouvrement top10 0,9600
# A DECLARER franchement : la TAILLE N'EST PAS CONTRAIGNANTE (13 Mo contre la limite de
# 100 Mo de GitHub). Ce n'est donc pas une optimisation de poids, et le pretendre serait
# malhonnete. float16 divise le fichier par deux a COUT DE CLASSEMENT NUL mesure (5,2e-05,
# trois ordres de grandeur sous notre plus petite marge, +0,0477). int8 economiserait 3,25 Mo
# de plus mais perd 4 % du top-10 et injecte 8,7e-03 d'erreur, soit 9 a 17 % de nos marges :
# degrader la distribution AVANT que le D5 y calibre ses deux seuils corromprait la mesure
# dont il depend.
DTYPE = "float16"

# ---- BM25 -------------------------------------------------------------------------------
BM25_K1 = 1.2
BM25_B = 0.75

# Liste d'arret volontairement COURTE : l'IDF fait deja le travail, une liste longue
# supprime du signal. Aucun mot ci-dessous ne doit collisionner avec un terme metier apres
# repli des accents — c'est verifie par une assertion executable (voir `garde_liste_arret`),
# parce que « mais » est le repli de « mais » (la cereale) : la stopper effacerait une de nos
# cinq cultures du BM25.
LISTE_ARRET = {
    # francais
    "le", "la", "les", "un", "une", "des", "du", "de", "d", "au", "aux", "l",
    "et", "ou", "ni", "car", "donc", "or", "que", "qui", "quoi", "dont",
    "ce", "cet", "cette", "ces", "son", "sa", "ses", "leur", "leurs", "notre", "nos",
    "il", "elle", "ils", "elles", "on", "nous", "vous", "je", "tu", "se", "sy",
    "dans", "sur", "sous", "avec", "sans", "pour", "par", "vers", "chez", "entre",
    "est", "sont", "etait", "etaient", "sera", "seront", "soit", "avoir", "etre",
    "ont", "avait", "avaient", "aura", "auront", "fait", "faire", "peut", "peuvent",
    "plus", "moins", "tres", "trop", "aussi", "meme", "memes", "autre", "autres",
    "tout", "tous", "toute", "toutes", "chaque", "quelques",
    "en", "y", "ne", "pas", "ainsi", "afin", "lors", "apres", "avant", "depuis",
    "cela", "celui", "celle", "ceux", "comme", "alors", "encore", "deja",
    # anglais
    "a", "an", "of", "in", "on", "at", "to", "for", "with", "without", "from", "by",
    "and", "or", "but", "if", "then", "than", "that", "this", "these", "those",
    "is", "are", "was", "were", "be", "been", "being", "has", "have", "had",
    "it", "its", "as", "such", "which", "who", "whom", "whose", "not", "no",
    "can", "could", "may", "might", "will", "would", "should", "must",
    "there", "here", "also", "more", "most", "some", "any", "all", "both", "each",
    "we", "our", "they", "their", "he", "she",
}

# Termes que la liste d'arret n'a pas le droit de manger. Repli des accents applique.
TERMES_METIER = [
    "mais", "mil", "sorgho", "arachide", "niebe", "striga", "semis", "semer",
    "dose", "engrais", "urée", "variete", "cycle", "pluie", "sol", "ete", "hivernage",
    "chenille", "legionnaire", "ravageur", "fumure", "ecartement", "densite", "poquet",
    "millet", "cowpea", "groundnut", "sowing", "rate", "pest", "yield", "soil",
]


# ========================================================================================
# 1. PERIMETRE
# ========================================================================================
def charge_sources() -> list[dict]:
    with io.open(SOURCES, encoding="utf-8") as fh:
        return yaml.safe_load(fh)["sources"]


def documents_indexables(sources: list[dict]) -> tuple[list[dict], dict]:
    """Filtre sur les CHAMPS. Renvoie (retenus, comptes) pour que le manifeste declare
    l'arithmetique au lieu d'annoncer un total nu."""
    methodologique = [s for s in sources if s.get("utilite_conseil") == "methodologique"]
    exclus = [s for s in sources if s.get("exclusion")]
    retenus = [s for s in sources
               if s.get("utilite_conseil") != "methodologique" and not s.get("exclusion")]
    comptes = {
        "n_sources": len(sources),
        "exclus_utilite_methodologique": len(methodologique),
        "exclus_champ_exclusion": len(exclus),
        "n_documents": len(retenus),
        "ids_exclus_methodologique": sorted(s["id"] for s in methodologique),
        "ids_exclus_exclusion": sorted(f"{s['id']}:{s['exclusion']}" for s in exclus),
    }
    for nom, obtenu, attendu in (
            ("n_sources", len(sources), N_SOURCES_ATTENDU),
            ("methodologique", len(methodologique), N_METHODOLOGIQUE_ATTENDU),
            ("exclusion", len(exclus), N_EXCLUSION_ATTENDU),
            ("n_documents", len(retenus), N_DOCUMENTS_ATTENDU)):
        if obtenu != attendu:
            raise SystemExit(
                f"ECHEC perimetre : {nom} = {obtenu}, {attendu} attendu (verrou D4). "
                "Un ecart doit etre une decision : mettre a jour le verrou ET sources.yaml "
                "ET le pairing.description de metadata.json (tools_corpus/_check_metadata.py)."
            )
    return retenus, comptes


# ========================================================================================
# 2. LECTURE DES .txt : en-tete de provenance + corps
# ========================================================================================
def lit_document(source: dict) -> tuple[dict, str, int]:
    """(entete, texte_integral, offset_debut_corps). Les offsets manipules ensuite sont
    ABSOLUS dans `texte_integral` : la verification peut donc trancher par une egalite de
    tranche, sans rejouer le decoupage."""
    chemin = TXT / source["regime"] / f"{source['id']}.txt"
    if not chemin.is_file():
        raise SystemExit(f"ECHEC {chemin} absent — relancer tools_corpus/extract.py")
    texte = io.open(chemin, encoding="utf-8", newline="").read()
    coupe = texte.find(SEPARATEUR_CORPS)
    if coupe < 0:
        raise SystemExit(f"ECHEC {chemin} : ligne '{SEPARATEUR_CORPS}' absente")
    entete: dict[str, str] = {}
    for ligne in texte[:coupe].splitlines():
        if ligne.startswith("==="):
            continue
        if ": " in ligne or ligne.endswith(":"):
            cle, _, valeur = ligne.partition(":")
            valeur = valeur.strip()
            # L'en-tete est une SERIALISATION texte : le jeton `null` y signifie « absent »
            # (26 documents sur 33 n'ont aucune mention juridique dans leur PDF). Le rendre
            # a la chaine vide est un DECODAGE, pas un maquillage — sans quoi le chunk
            # annonce `"licence": "null"`, qu'un consommateur du D5 lirait comme un nom de
            # licence. Mesure : 2044 chunks sur 3180 etaient dans ce cas, et c'est
            # `verify_index.py` section 3 (croisement champ par champ contre sources.yaml)
            # qui l'a attrape — pas une relecture.
            entete[cle.strip()] = "" if valeur == "null" else valeur
    debut_corps = texte.index("\n", coupe + len(SEPARATEUR_CORPS) - 1) + 1
    return entete, texte, debut_corps


def segments_pages(texte: str, debut_corps: int) -> list[tuple[int, int, int]]:
    """[(page, debut, fin)] : le contenu de chaque page, marqueur EXCLU. Un chunk heritera
    du dernier marqueur vu avant lui parce qu'il ne peut pas sortir de son segment."""
    marqueurs = [(int(m.group(1)), m.start(), m.end())
                 for m in RE_MARQUEUR.finditer(texte, debut_corps)]
    if not marqueurs:
        raise SystemExit("ECHEC aucun marqueur [[page N]] : la citation perdrait sa page")
    avant = texte[debut_corps:marqueurs[0][1]]
    if avant.strip():
        raise SystemExit(
            f"ECHEC {len(avant.strip())} caracteres de corps AVANT le premier marqueur : "
            "ils n'auraient aucune page attribuable")
    segments = []
    for k, (page, _debut_m, fin_m) in enumerate(marqueurs):
        fin = marqueurs[k + 1][1] if k + 1 < len(marqueurs) else len(texte)
        segments.append((page, fin_m, fin))
    pages = [p for p, _, _ in segments]
    if pages != sorted(set(pages)):
        raise SystemExit(f"ECHEC marqueurs non croissants ou dupliques : {pages[:12]}…")
    return segments


# ========================================================================================
# 3. UNITES : phrases, repli lignes, repli dur
# ========================================================================================
RE_FIN_PHRASE = re.compile(r"([.!?…])([\"»’')\]]*)(\s)")
RE_LIGNE_VIDE = re.compile(r"\n[ \t]*\n")
ABREVIATIONS = {
    "etc", "ex", "cf", "fig", "figs", "tab", "tabl", "no", "nos", "p", "pp", "al", "vol",
    "ed", "eds", "dr", "pr", "m", "mme", "mm", "st", "env", "min", "max", "moy", "approx",
    "eq", "resp", "i.e", "e.g", "spp", "sp", "var", "cv", "subsp", "av", "ap", "art",
    "chap", "ref", "vs", "viz", "pl", "nb", "fao", "inc", "ltd", "dir", "coord", "trad",
}


def _est_coupe_valide(texte: str, m: re.Match, debut: int) -> bool:
    """Une fin de phrase plausible n'est pas une fin de phrase. Trois refus mesures sur ce
    corpus : abreviations, initiales, et continuation en minuscule (ligne PDF coupee)."""
    precedent = texte[max(debut, m.start(1) - 14):m.start(1)]
    mot = re.search(r"\S+$", precedent)
    if mot:
        brut = mot.group().lower().strip("(«\"'[")
        if brut in ABREVIATIONS or brut.rstrip(".") in ABREVIATIONS:
            return False
        if len(brut) == 1 and brut.isalpha():      # initiale : « B. Birhanu »
            return False
    if "\n\n" in m.group(3):
        return True
    suite = texte[m.end():m.end() + 40]
    apres = suite.lstrip()
    if apres and apres[0].islower():               # continuation, pas nouvelle phrase
        return False
    return True


def unites(texte: str, debut: int, fin: int, maxi: int) -> list[tuple[int, int]]:
    """Decoupe [debut, fin) en unites contigues et non vides."""
    coupes = {debut, fin}
    for m in RE_FIN_PHRASE.finditer(texte, debut, fin):
        if _est_coupe_valide(texte, m, debut):
            coupes.add(min(m.end(2), fin))
    for m in RE_LIGNE_VIDE.finditer(texte, debut, fin):
        coupes.add(min(m.start() + 1, fin))
    bornes = sorted(coupes)
    brutes = [(a, b) for a, b in zip(bornes, bornes[1:]) if texte[a:b].strip()]

    # Repli : une unite plus longue que maxi (tableau sans ponctuation, tres frequent dans
    # ces PDF) se recoupe aux fins de ligne, puis en dur. Sans ce repli, un tableau entier
    # deviendrait un chunk unique et le cosinus n'y discriminerait plus rien.
    sorties: list[tuple[int, int]] = []
    for a, b in brutes:
        if b - a <= maxi:
            sorties.append((a, b))
            continue
        coupes_l = sorted({a} | {a + m.start() + 1 for m in re.finditer(r"\n", texte[a:b])} | {b})
        courant = coupes_l[0]
        for borne in coupes_l[1:]:
            while borne - courant > maxi:
                sorties.append((courant, courant + maxi))
                courant += maxi
            if borne - courant > 0 and texte[courant:borne].strip():
                sorties.append((courant, borne))
                courant = borne
            else:
                courant = borne
    return [(a, b) for a, b in sorties if texte[a:b].strip()]


# ========================================================================================
# 4. EMPAQUETAGE EN CHUNKS
# ========================================================================================
def chunks_du_segment(texte: str, debut: int, fin: int, cible: int, maxi: int,
                      mini: int, chevauchement: int) -> list[tuple[int, int]]:
    """Empaquette les unites d'UNE page. Renvoie des tranches contigues [a, b)."""
    us = unites(texte, debut, fin, maxi)
    if not us:
        return []
    resultat: list[tuple[int, int]] = []
    i = 0
    while i < len(us):
        j = i
        while j + 1 < len(us) and (us[j + 1][1] - us[i][0]) <= cible:
            j += 1
        # Un chunk ne depasse jamais maxi : les unites sont deja bornees a maxi.
        a, b = us[i][0], us[j][1]
        if b - a > maxi and j > i:
            j -= 1
            a, b = us[i][0], us[j][1]
        resultat.append((a, b))
        if j + 1 >= len(us):
            break
        suivant = max(i + 1, j + 1 - chevauchement)
        i = suivant

    # Absorption des chunks sous `mini` : un chunk de 2 caracteres n'est ni citable ni
    # retrouvable, et BM25 le favorise mecaniquement (normalisation par la longueur). On
    # l'absorbe dans le voisin de la MEME page, en tolerant un debordement jusqu'a
    # maxi + mini : un chunk un peu trop long reste citable. On ne SUPPRIME jamais rien —
    # une troncature silencieuse est exactement ce qu'on reproche au champ.
    # Si la page entiere tient sous `mini`, le chunk reste tel quel : c'est le cas des
    # fiches varietales d'une page (« Dan Hadjia », 187 car.), notre matiere la plus citable.
    plafond = maxi + mini
    change = True
    while change and len(resultat) > 1:
        change = False
        for k, (a, b) in enumerate(resultat):
            if len(texte[a:b].strip()) >= mini:
                continue
            gauche = resultat[k - 1] if k > 0 else None
            droite = resultat[k + 1] if k + 1 < len(resultat) else None
            cands = []
            if gauche is not None and (b - gauche[0]) <= plafond:
                cands.append((b - gauche[0], k - 1, (gauche[0], b)))
            if droite is not None and (droite[1] - a) <= plafond:
                cands.append((droite[1] - a, k, (a, droite[1])))
            if not cands:
                continue
            _, base, fusion = min(cands)
            resultat[base:base + 2] = [fusion]
            change = True
            break
    return resultat


# ========================================================================================
# 4bis. Bruit structurel — MARQUE, jamais supprime
# ========================================================================================
# Defaut MESURE au D4 (PREUVES.md 18.3) : les meilleurs distracteurs du cosinus ne sont pas
# des chunks « un peu hors sujet », ce sont des bibliographies, des sommaires et des pages de
# titre — du texte lexicalement DENSE en termes agronomiques et VIDE de conseil.
# `oar_57448 p15` (« Gilbert RA, Heilman JL, Juo ASR (2003) Diurnal and seasonal light
# transmission to cowpea… ») sort RANG 1 sur « cowpea varieties for the Sahel ».
# Le phenomene est INDEPENDANT de la taille de chunk : on ne le corrige pas en changeant la
# cible, il faut le nommer.
#
# Trois signaux DISJOINTS, et non un seuil global — parce qu'un seuil global a ete mesure
# FAUX : la densite de citation ne separe pas une liste de references d'une prose
# scientifique densement citee (`oar_57372 p2` « One H. hebetor female can produce an
# offspring of 200-400 individuals (Yu et al. 1999… » est du VRAI conseil a 10,5 citations
# /1000 car.). C'est la FORME DES LIGNES qui separe, pas leur densite.
#
# On MARQUE (`structure`), on ne supprime pas : un chunk de bibliographie reste la seule
# trace de la provenance d'un fait, et supprimer du texte serait la troncature silencieuse
# qu'on reproche au champ. Le D5 s'en sert pour DEclasser, jamais pour cacher.
RE_LIGNE_REF = re.compile(
    r"^\s*(?:\[?\d{1,3}[\].)]\s*)?[A-Z][A-Za-zÀ-ſ'-]+,?\s+(?:[A-Z]\.?\s?){1,4}[,;(]")
RE_REF_INDICE = re.compile(r"(?:doi|https?://)", re.I)
RE_REF_ANNEE = re.compile(r"\((?:19|20)\d\d[a-z]?\)|\b(?:19|20)\d\d[;.]")
RE_POINTS_CONDUCTEURS = re.compile(r"\.{4,}")
RE_MOTS_CLES = re.compile(r"\b(?:key\s?words?|mots\s?cl[eé]s?)\b\s*[:.]", re.I)

FRAC_REF_SEUIL = 0.5          # calibre : 0,5 -> 69 chunks / 1905, echantillon 100 % references
POINTS_CONDUCTEURS_SEUIL = 3  # calibre : 54 chunks, tous des SOMMAIRE avec points de conduite


def structure(texte: str) -> str:
    """Nomme le bruit structurel d'un chunk. Renvoie "" pour du texte ordinaire.

    Aucun des trois tests ne regarde le SUJET : un chunk agronomique parfaitement pertinent
    ne peut pas etre marque par sa thematique, seulement par sa FORME.
    """
    if len(RE_POINTS_CONDUCTEURS.findall(texte)) >= POINTS_CONDUCTEURS_SEUIL:
        return "sommaire"
    lignes = [l for l in texte.split("\n") if l.strip()]
    if len(lignes) >= 3:
        n = sum(1 for l in lignes
                if RE_LIGNE_REF.match(l)
                or (len(l) < 130 and (RE_REF_INDICE.search(l) or RE_REF_ANNEE.search(l))))
        if n / len(lignes) >= FRAC_REF_SEUIL:
            return "bibliographie"
    if RE_MOTS_CLES.search(texte):
        return "mots_cles"
    return ""


def construis_chunks(sources: list[dict], cible: int, maxi: int, mini: int,
                     chevauchement: int) -> tuple[list[dict], list[dict], dict]:
    chunks: list[dict] = []
    documents: list[dict] = []
    stats = {"pages_totales": 0, "pages_sous_plancher": 0, "car_pages_sous_plancher": 0,
             "car_chunks": 0, "car_corps": 0}
    for source in sources:
        entete, texte, debut_corps = lit_document(source)
        documents.append({
            "id": source["id"],
            "entete": entete,
            "chemin_txt": f"corpus/txt/{source['regime']}/{source['id']}.txt",
            "sha256_txt": hashlib.sha256(texte.encode("utf-8")).hexdigest(),
            "car_txt": len(texte),
        })
        stats["car_corps"] += len(texte) - debut_corps
        n_avant = len(chunks)
        for page, a, b in segments_pages(texte, debut_corps):
            stats["pages_totales"] += 1
            if len(texte[a:b].strip()) < PLANCHER_PAGE:
                stats["pages_sous_plancher"] += 1
                stats["car_pages_sous_plancher"] += len(texte[a:b].strip())
                continue
            for ca, cb in chunks_du_segment(texte, a, b, cible, maxi, mini, chevauchement):
                extrait = texte[ca:cb]
                rogne_g = len(extrait) - len(extrait.lstrip())
                rogne_d = len(extrait) - len(extrait.rstrip())
                ca2, cb2 = ca + rogne_g, cb - rogne_d
                corps = texte[ca2:cb2]
                if not corps.strip():
                    continue
                chunks.append({
                    "i": len(chunks),
                    # --- provenance de citation, EN LIGNE sur chaque chunk ---
                    "doc": source["id"],
                    "titre": entete.get("titre", ""),
                    "editeur": entete.get("editeur", ""),
                    "annee": entete.get("annee", ""),
                    "langue": entete.get("langue", ""),
                    "licence": entete.get("licence", ""),
                    "regime": entete.get("regime", ""),
                    "page": page,
                    "citation_verbatim_autorisee": entete.get(
                        "citation_verbatim_autorisee", ""),
                    "sujet": entete.get("sujet", ""),
                    "portee": entete.get("portee", ""),
                    "utilite_conseil": entete.get("utilite_conseil", ""),
                    # --- tracabilite exacte : texte == txt[off0:off1] ---
                    "off": [ca2, cb2],
                    "car": cb2 - ca2,
                    # --- bruit structurel MARQUE, jamais supprime (section 4bis) ---
                    "structure": structure(corps),
                    "texte": corps,
                })
                stats["car_chunks"] += cb2 - ca2
                if chunks[-1]["structure"]:
                    stats.setdefault("structure", {})
                    stats["structure"][chunks[-1]["structure"]] = 1 + stats["structure"].get(
                        chunks[-1]["structure"], 0)
        if len(chunks) == n_avant:
            raise SystemExit(f"ECHEC {source['id']} n'a produit aucun chunk")
        documents[-1]["n_chunks"] = len(chunks) - n_avant
    return chunks, documents, stats


# ========================================================================================
# 5. BM25
# ========================================================================================
RE_JETON = re.compile(r"[0-9a-z]+(?:-[0-9a-z]+)*")


def replie(texte: str) -> str:
    """Minuscules + repli des accents. Le repli est un choix de RAPPEL : un juge peut taper
    « niebe » pour « niebe ». Il fusionne « mais » et « mais » (la cereale) — d'ou la garde
    `garde_liste_arret`, qui interdit de stopper le mot fusionne."""
    sans = unicodedata.normalize("NFKD", texte.lower().replace("’", "'"))
    return "".join(c for c in sans if not unicodedata.combining(c))


def jetons(texte: str) -> list[str]:
    """Un compose a tiret est emis ENTIER **et** en morceaux : « 15-15-15 » doit se retrouver
    tel quel (dose NPK citee par le test_prompt) et par « 15 » ; « septembre-octobre » doit
    se retrouver par « septembre »."""
    sortie: list[str] = []
    for brut in RE_JETON.findall(replie(texte).replace("'", " ")):
        morceaux = brut.split("-")
        if len(morceaux) > 1:
            sortie.append(brut)
        for m in morceaux:
            if len(m) >= 2 or m.isdigit():
                sortie.append(m)
    return [j for j in sortie if j not in LISTE_ARRET]


def garde_liste_arret() -> None:
    collisions = sorted({t for t in (replie(x) for x in TERMES_METIER)} & LISTE_ARRET)
    if collisions:
        raise SystemExit(
            f"ECHEC la liste d'arret mange des termes metier : {collisions}. "
            "Le repli des accents fusionne des mots ; un terme metier ne se stoppe jamais.")


def index_bm25(chunks: list[dict]) -> dict:
    postings: dict[str, dict[int, int]] = {}
    doclen: list[int] = []
    for c in chunks:
        js = jetons(c["texte"])
        doclen.append(len(js))
        vus: dict[str, int] = {}
        for j in js:
            vus[j] = vus.get(j, 0) + 1
        for terme, tf in vus.items():
            postings.setdefault(terme, {})[c["i"]] = tf
    return {
        "k1": BM25_K1,
        "b": BM25_B,
        "n_chunks": len(chunks),
        "avgdl": (sum(doclen) / len(doclen)) if doclen else 0.0,
        "doclen": doclen,
        # df se derive de len(postings[t]) : ne pas le stocker deux fois.
        "postings": {t: [[i, tf] for i, tf in sorted(p.items())]
                     for t, p in sorted(postings.items())},
    }


# ========================================================================================
# 6. EMBEDDINGS (avec cache de reprise, HORS index committe)
# ========================================================================================
def _cle(texte: str) -> str:
    empreinte = f"{embed_server.POOLING}|{embed_server.GGUF_OCTETS}|{texte}"
    return hashlib.sha256(empreinte.encode("utf-8")).hexdigest()


def charge_cache() -> tuple[dict[str, int], np.ndarray]:
    cles_p, vecs_p = CACHE.with_suffix(".json"), CACHE.with_suffix(".npy")
    if cles_p.is_file() and vecs_p.is_file():
        cles = json.loads(io.open(cles_p, encoding="utf-8").read())
        vecs = np.load(vecs_p)
        if len(cles) == len(vecs):
            return {c: i for i, c in enumerate(cles)}, vecs
    return {}, np.zeros((0, embed_server.DIM), dtype=np.float32)


def sauve_cache(index: dict[str, int], vecs: np.ndarray) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    ordre = [None] * len(index)
    for cle, i in index.items():
        ordre[i] = cle
    io.open(CACHE.with_suffix(".json"), "w", encoding="utf-8", newline="\n").write(
        json.dumps(ordre, ensure_ascii=False))
    np.save(CACHE.with_suffix(".npy"), vecs)


def embarque(chunks: list[dict], lot: int, sans_cache: bool = False) -> np.ndarray:
    idx, vecs = ({}, np.zeros((0, embed_server.DIM), dtype=np.float32)) if sans_cache \
        else charge_cache()
    cles = [_cle(c["texte"]) for c in chunks]
    manquants = [k for k, cle in enumerate(cles) if cle not in idx]
    print(f"       {len(chunks) - len(manquants)}/{len(chunks)} chunks deja en cache, "
          f"{len(manquants)} a embarquer")
    if manquants:
        with Serveur() as srv:
            faits = 0
            for depart in range(0, len(manquants), lot * 20):
                tranche = manquants[depart:depart + lot * 20]
                nouveaux = srv.plonge([chunks[k]["texte"] for k in tranche], lot=lot)
                base = len(vecs)
                vecs = np.concatenate([vecs, nouveaux], axis=0)
                for pos, k in enumerate(tranche):
                    idx[cles[k]] = base + pos
                faits += len(tranche)
                if not sans_cache:
                    sauve_cache(idx, vecs)   # reprise possible si le run meurt
                print(f"       lot cumule {faits}/{len(manquants)} (cache ecrit)")
    return np.stack([vecs[idx[cle]] for cle in cles]).astype(np.float32)


# ========================================================================================
# 7. STOCKAGE
# ========================================================================================
def encode_vecteurs(v: np.ndarray, dtype: str) -> tuple[np.ndarray, dict]:
    """int8 : quantification symetrique a echelle globale 127. Les vecteurs sont
    L2-normalises, donc chaque composante tient dans [-1, 1]."""
    if dtype == "float32":
        return v.astype(np.float32), {}
    if dtype == "float16":
        return v.astype(np.float16), {}
    if dtype == "int8":
        return np.clip(np.rint(v * 127.0), -127, 127).astype(np.int8), {"echelle": 1 / 127.0}
    raise SystemExit(f"dtype inconnu : {dtype}")


def sha256_fichier(chemin: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(chemin, "rb") as fh:
        for bloc in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloc)
    return h.hexdigest()


def version_llama_server() -> str:
    try:
        p = subprocess.run([str(embed_server.BINAIRE), "--version"],
                           capture_output=True, text=True, timeout=30)
        for ligne in (p.stderr + p.stdout).splitlines():
            if "version" in ligne.lower():
                return ligne.strip()
    except Exception as e:                                    # noqa: BLE001
        return f"illisible: {e}"
    return "illisible"


def ecris_jsonl(chemin: pathlib.Path, lignes: list[dict]) -> None:
    with io.open(chemin, "w", encoding="utf-8", newline="\n") as fh:
        for objet in lignes:
            fh.write(json.dumps(objet, ensure_ascii=False, sort_keys=True) + "\n")


def ecris_json(chemin: pathlib.Path, objet) -> None:
    with io.open(chemin, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(objet, fh, ensure_ascii=False, sort_keys=True, indent=1)
        fh.write("\n")


# ========================================================================================
# 8. PROGRAMME
# ========================================================================================
def bati(cible: int, maxi: int, mini: int, chevauchement: int, dtype: str,
         lot: int, sortie: pathlib.Path, sans_embeddings: bool) -> dict:
    garde_liste_arret()
    sources = charge_sources()
    retenus, comptes = documents_indexables(sources)
    print(f"ok     perimetre derive des CHAMPS : {comptes['n_sources']} sources "
          f"- {comptes['exclus_utilite_methodologique']} methodologique "
          f"- {comptes['exclus_champ_exclusion']} exclusion = {comptes['n_documents']} "
          f"(verrou {N_DOCUMENTS_ATTENDU})")

    chunks, documents, stats = construis_chunks(retenus, cible, maxi, mini, chevauchement)
    tailles = np.array([c["car"] for c in chunks])
    print(f"ok     {len(chunks)} chunks sur {stats['pages_totales']} pages "
          f"({stats['pages_sous_plancher']} pages sous plancher, "
          f"{stats['car_pages_sous_plancher']} car.) | "
          f"car/chunk min {tailles.min()} med {int(np.median(tailles))} max {tailles.max()}")

    bm25 = index_bm25(chunks)
    print(f"ok     BM25 : {len(bm25['postings'])} termes, avgdl {bm25['avgdl']:.1f}")

    vecteurs = None
    if not sans_embeddings:
        vecteurs = embarque(chunks, lot=lot)
        print(f"ok     embeddings {vecteurs.shape} dtype source float32")

    sortie.mkdir(parents=True, exist_ok=True)
    ecris_jsonl(sortie / "chunks.jsonl", chunks)
    ecris_json(sortie / "documents.json", documents)
    ecris_json(sortie / "bm25.json", bm25)
    extra = {}
    if vecteurs is not None:
        encodes, extra = encode_vecteurs(vecteurs, dtype)
        np.save(sortie / "vectors.npy", encodes)

    manifeste = {
        "bloc": "D4",
        "perimetre": comptes,
        "chunking": {
            "cible_car": cible, "maxi_car": maxi, "mini_car": mini,
            "chevauchement_unites": chevauchement, "plancher_page_car": PLANCHER_PAGE,
            "unite": "phrase, repli lignes puis coupe dure",
            "traverse_les_pages": False,
            "marqueurs_indexes": False,
            "texte_est_tranche_contigue_du_txt": True,
        },
        "chunks": {
            "n": len(chunks),
            "car_total": int(tailles.sum()),
            "car_min": int(tailles.min()), "car_median": int(np.median(tailles)),
            "car_moyen": round(float(tailles.mean()), 1), "car_max": int(tailles.max()),
            "pages_totales": stats["pages_totales"],
            "pages_sous_plancher": stats["pages_sous_plancher"],
            "car_pages_sous_plancher": stats["car_pages_sous_plancher"],
            # Declares, pas caches : chunks restes sous `mini` parce que la page entiere y
            # tient (fiches varietales d'une page). Aucun n'a ete supprime.
            "n_sous_mini": int((tailles < mini).sum()),
            "ids_sous_mini": sorted({f"{c['doc']}:p{c['page']}"
                                     for c in chunks if c["car"] < mini}),
            # Bruit structurel MARQUE, jamais supprime. Le D5 declasse sur ce champ.
            "structure_marquee": stats.get("structure", {}),
            "structure_effet_mesure": (
                "a cible 700 le marquage ne recupere AUCUNE marge (0,0000 sur 5 requetes) et "
                "gagne 3 rangs sur 1 requete : les distracteurs de rang 1 n'y sont pas "
                "marques. A cible 1100 il en marque 2 sur 5 et rend +0,0328 de marge. La "
                "taille de chunk fait donc le gros du travail ; le champ reste un levier "
                "residuel pour le D5, pas la correction qu'on croyait."),
        },
        "bm25": {"k1": BM25_K1, "b": BM25_B, "n_termes": len(bm25["postings"]),
                 "avgdl": round(bm25["avgdl"], 2),
                 "liste_arret_n": len(LISTE_ARRET),
                 "liste_arret_sha256": hashlib.sha256(
                     "|".join(sorted(LISTE_ARRET)).encode("utf-8")).hexdigest()[:16],
                 "jeton": "minuscules + repli accents ; compose a tiret emis entier ET en morceaux"},
        "embeddings": {
            "modele": "BGE-M3 Q8_0 (GGUF)",
            "chemin": "model/bge-m3/bge-m3-Q8_0.gguf",
            "octets": embed_server.GGUF_OCTETS,
            "moteur": "llama.cpp / llama-server --embedding (regle 4)",
            "llama_server": version_llama_server(),
            "pooling": embed_server.POOLING,
            "dim": embed_server.DIM,
            "normalisation": "faite par le serveur (norme 1,0 assertee, jamais reimposee)",
            "dtype_stocke": dtype if vecteurs is not None else None,
            **extra,
        },
        "seuils_cosinus": "AUCUN. Calibres au D5 sur distribution mesuree (PREUVES.md 17.5.1).",
    }
    if vecteurs is not None:
        manifeste["embeddings"]["sha256_gguf"] = sha256_fichier(embed_server.GGUF)
    ecris_json(sortie / "manifest.json", manifeste)

    for f in sorted(sortie.iterdir()):
        print(f"       {f.name:20} {f.stat().st_size:>12,} octets")
    return manifeste


def principal() -> None:
    ap = argparse.ArgumentParser(description="D4 : batit rag/index/")
    ap.add_argument("--cible", type=int, default=CIBLE)
    ap.add_argument("--max", dest="maxi", type=int, default=MAXI)
    ap.add_argument("--mini", type=int, default=MINI)
    ap.add_argument("--chevauchement", type=int, default=CHEVAUCHEMENT)
    ap.add_argument("--dtype", default=DTYPE,
                    choices=["float32", "float16", "int8"])
    ap.add_argument("--lot", type=int, default=8)
    ap.add_argument("--sortie", type=pathlib.Path, default=SORTIE)
    ap.add_argument("--sans-embeddings", action="store_true",
                    help="chunking + BM25 seuls (mesure rapide, aucun serveur lance)")
    a = ap.parse_args()
    bati(a.cible, a.maxi, a.mini, a.chevauchement, a.dtype, a.lot,
         a.sortie, a.sans_embeddings)


if __name__ == "__main__":
    principal()
