"""Verification COMPORTEMENTALE de rag/retrieve.py + rag/answer.py — chaque controle EXECUTE l'acte.

Meme doctrine que `rag/verify_index.py`, pour la meme raison : sur ce projet, quatre commandes de
verification ont deja rendu un resultat plausible AU LIEU d'un echec — `jq` jamais installe,
`grep -c $'\\r'` a motif vide qui compte 89 CR dans un fichier LF pur, `git check-ignore` qui cite
une ligne VIDE, et une sonde de detection de boucle qui a refuse un run SAIN la premiere fois
qu'elle a ete exercee. Une verification dont l'echec ressemble a un succes est pire qu'aucune.

D'ou la regle appliquee ici, sans exception : tout controle qui pourrait etre vert par accident
porte sa CONTRE-EPREUVE — on fabrique la faute que le controle doit attraper, et on exige qu'il
l'attrape. Un test qui ne peut pas echouer ne prouve rien.

Deux proprietes de ce fichier valent d'etre dites avant les sections :

1. Les sections 0 a 8 n'ouvrent AUCUN serveur d'embedding. Elles fabriquent le dictionnaire `res`
   que `retrieve.cherche()` aurait rendu, par les memes fonctions de fusion et de porte, puis le
   passent a `answer.analyse()`. C'est ce qui rend les contre-epreuves possibles : on peut poser un
   classement FAUX (le chunk niebe au rang 1 d'une question mil) et exiger que la politique le
   refuse, ce qu'aucun serveur ne produirait spontanement.
2. Les seuils ne sont jamais ecrits ici. Ils sont RE-DERIVES de `rag/index/seuils.json`, qui est un
   livrable GENERE : la section 1 verifie que le seuil est bien le milieu du creux mesure, que la
   statistique elue ne l'a pas ete sur l'AUC (une alternative fait mieux en AUC et pire en cout),
   et qu'aucun litteral de seuil ne s'est glisse dans le code des deux modules livres.

Sections (0 a 8 sans serveur, 9 a 15 avec) :
   0. parite des deux test_prompts contre metadata.json ; les 2 nombres du perimetre RE-DERIVES ;
      les 13 cles de provenance extraites de la SOURCE de retrieve.py                + 3 contre-ep.
   1. porte 2/3 re-derivee du creux ; election NON fondee sur l'AUC ; invariance dans le creux ;
      aucun litteral de seuil dans le code ; les 3 gardes de charge_seuils exercees  + 5 contre-ep.
   2. plafond de citation re-derive ; troncature a la frontiere de mot ; l'ellipse n'entre pas
      dans le verbatim                                                              + 3 contre-ep.
   3. parite du tokeniseur BM25 dans les DEUX sens contre les postings de l'index    + 1 contre-ep.
   4. le test negatif obligatoire passe pour DEUX raisons independantes, assertees separement
                                                                                    + 2 contre-ep.
   5. paire asymetrique : la meme phrase citable pour niebe est REFUSEE pour mil
   6. zero appel de generateur — structurellement (tokenize) ET par un compteur      + 3 contre-ep.
   7. la conjonction francaise « mais » n'est pas lue comme la culture « mais »      + 2 contre-ep.
   8. balayage permanent des renvois : aucun fichier committable ne cite un chemin absent du
      depot public                                                                   + 1 contre-ep.
   9. non-regression du test_prompt 1 sur le JEU DE CITATIONS (le niveau est aveugle)
  10. test_prompt 2 : niveau 1 + reserve de niveau 2, citations francaises pour une question anglaise
  11. refus reel : niveau 3, perimetre nomme, zero appel de generateur
  12. rejeu du jeu QCM etiquete a la main quand il est present ; sous-ensemble autoportant sinon
  13. PENALITE_STRUCTURE mesuree a 0 et a 5 avant d'etre figee (vecteur reutilise, reutilisation
      PROUVEE par le compteur d'appels d'embedding)
  14. degradation « socle committed seul » : que reste-t-il quand on retire les 26 documents non
      redistribues ?
  15. les QUATRE commutateurs de l'etape 4 dans leurs DEUX bras — veto de variete (portee CHUNK),
      couture de meme page, borne 5 « le texte cousu n'est pas deja dans sa source », borne 6 « la
      citation nomme la cible sanitaire de la requete ». La section 12 n'assert que le NIVEAU et
      aucune des regles n'en change un : elle est aveugle a ce que cette section mesure
                                                                                  + 1 contre-mesure

Usage : py -X utf8 rag/verify_retrieve.py [--rapide] [--echantillon N]
  --rapide  saute les sections 9 a 15 (les seules qui lancent un serveur) et imprime une note
            nommant ce qui reste NON prouve.
Sortie : exit 0 si 0 echec, exit 1 sinon. Les « notes » n'echouent pas mais s'impriment.
"""
from __future__ import annotations

import argparse
import ast
import io
import json
import pathlib
import random
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import tokenize

RACINE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE / "rag"))

import answer                            # noqa: E402
import retrieve                          # noqa: E402

INDEX = RACINE / "rag" / "index"
META = RACINE / "metadata.json"
MODULES_LIVRES = [RACINE / "rag" / "retrieve.py", RACINE / "rag" / "answer.py"]

# RENVOIS-DECLARES-DEBUT
# Region auto-exemptee du balayage de la section 8. Elle existe pour une raison precise : ce
# fichier-ci DOIT nommer les chemins que le balayage interdit, sinon il ne pourrait pas les
# chercher. L'exemption est donc declaree, delimitee, et la section 8 verifie elle-meme que chaque
# marqueur n'apparait qu'UNE fois dans le fichier (les marqueurs sont assembles a l'execution
# ailleurs, pour que le litteral complet ne vive que dans ces deux commentaires).
QCM = RACINE / "tools_corpus" / "_qcm.jsonl"
RENVOIS_INTERDITS = [
    # (a) chemins hors depot public
    "tools_corpus", "PREUVES.md", "RECON-D3.md", "CLAUDE.md", "spotcheck_nu",
    "corpus/txt/fetched", "check_submission.py", "_embed_server.log",
    # 🔴 `_qcm.jsonl` RETIRE de cette famille le 23/08 : le jeu etiquete a la main est
    #     devenu un LIVRABLE (exception `!tools_corpus/_qcm.jsonl`), donc un renvoi vers lui n'est
    #     plus pendant — un juge peut l'ouvrir. L'aiguille ne mesurait plus ce pour quoi elle
    #     existait. ⚠️ Son generateur `_build_qcm` reste dans la famille (b) ci-dessous, lui : il
    #     n'est pas publie. Et `tools_corpus` reste arme, donc un renvoi au REPERTOIRE echoue
    #     toujours : c'est le fichier qui est publie, pas l'outillage autour.
    #     Consequence a ne pas perdre : le jeu publie tombe desormais DANS le perimetre du
    #     balayage (`.jsonl` est un suffixe texte), il est donc lui-meme controle a chaque run.
    # (b) ARTEFACTS de mesure de cet outillage, cites par leur nom NU (donc invisibles a la
    #     famille (a), qui n'attrape que les mentions prefixees du repertoire) : mesure du 20/08,
    #     8 lignes des livrables renvoyaient a un rapport de mesure que le juge ne peut pas ouvrir.
    "_mesure_d5", "_mesure_chunking", "_mesure_motifs", "_mesure_entite", "_calibration_seuils",
    "_prompt_D5", "_ecris_seuils", "_build_qcm",
    # (c) le NUMERO d'une lecon du journal de projet. « erreur n°41 » ne se resout que dans un
    #     fichier hors depot : la lecon doit donc etre ECRITE sur place, jamais referencee par son
    #     numero. Mesure du 20/08 : 17 occurrences reecrites dans les trois modules du D5, puis
    #     16 de plus dans le reste des fichiers committables (voir ci-dessous).
    # 🔴 DEUX orthographes, et c'est la seconde qui portait le vrai gisement. L'aiguille n'a
    #     longtemps existe qu'avec le signe « ° » — donc le balayage annoncait « 0 renvoi pendant »
    #     pendant que 15 renvois ecrits « erreur n 23 » (sans le signe, convention ASCII des
    #     commentaires de ces fichiers) traversaient le filtre. Un detecteur dont l'aiguille est plus
    #     etroite que la faute qu'il existe pour attraper rend un vert qui ne veut rien dire, et le
    #     vert est plus dur a mettre en doute qu'un rouge. 0 faux positif mesure avant
    #     elargissement : la tournure « une erreur n'est pas » ne matche pas, l'aiguille exigeant
    #     un ESPACE ou un signe et non une apostrophe (contre-eprouve plus bas). Ne pas retirer
    #     l'une des deux formes en croyant l'autre suffisante.
    # 🔴 Et mon balayage A LA MAIN de cette meme faute en avait compte 15 : l'aiguille
    #     elargie en a leve 16 — `.gitattributes` etait hors du perimetre que j'avais regarde
    #     (les modules `rag/`). Une correction faite sur un perimetre plus etroit que
    #     l'affirmation qu'elle porte est une faute neuve, et elle herite de la credibilite du
    #     mot « corrige » : c'est le controle executable qui fixe le perimetre, jamais la
    #     relecture.
    "erreur n°", "erreurs n°", "erreur n ", "erreurs n ",
    # (d) RETIREE le 23/08. Une famille d'aiguilles « pseudos de forks concurrents » a existe ici :
    #     son travail etait d'empecher que la redaction ecrive ces noms dans un livrable, et elle
    #     l'a fait. Mais une fois la redaction finie, la LISTE ELLE-MEME etait le dernier endroit du
    #     depot a les porter — sept fois, trois lignes sous le commentaire qui l'interdisait, dans le
    #     seul fichier revelant qu'une reconnaissance concurrentielle a eu lieu. Un detecteur dont
    #     l'aiguille est le secret qu'il protege le publie a chaque commit. ⇒ Retirer la famille EST
    #     le correctif, pas un renoncement : les familles (a)/(b)/(c) designent des chemins, qui
    #     restent des chemins ; celle-ci citait la donnee. Mesure avant retrait, controle positif a
    #     l'appui : 0 occurrence dans les 47 fichiers committables hors cette liste — sauf
    #     `.gitignore`, hors perimetre declare (corrige separement le 23/08). Ne pas la reintroduire :
    #     le mode d'echec qu'on veut ecarter s'ecrit sans eux.
]
# Exclusions de PERIMETRE, declarees et imprimees plutot que tacites :
#  - `.gitignore` est exclu EN ENTIER : son role est precisement de nommer ce qui est exclu du
#    depot, donc l'y chercher rendrait un echec structurel a chaque run.
#  - `model/`, `.gguf` et `corpus/pdf` ne sont volontairement PAS des aiguilles : les livrables
#    doivent pouvoir decrire ou telecharger les poids (`download_model.sh` le fait par contrat).
RENVOIS_HORS_PERIMETRE = [".gitignore"]

# EXEMPTIONS NOMMEES, par couple (fichier, aiguille). La premiere execution du balayage a leve
# 7 echecs, et les lire un par un a montre que la liste d'aiguilles confondait TROIS choses :
#  (1) le RENVOI PENDANT — le texte invite le lecteur a consulter un fichier qu'il ne pourra pas
#      ouvrir dans le depot public. C'est une vraie faute : l'affirmation devient « crois-moi ».
#  (2) la DECLARATION DE REGIME — le chemin est nomme precisement pour dire qu'il n'est PAS
#      redistribue. Il n'y a rien a consulter, l'information est complete sur place.
#  (3) le CHEMIN D'EXECUTION PROPRE ou la PROVENANCE — le code ecrit la (cache, journal), ou
#      l'artefact enregistre d'ou vient un chunk. Ce dernier cas est EXIGE par la decision de
#      projet « la citation est tracable a la page » : le supprimer detruirait la tracabilite que
#      notre differenciateur revendique.
# Seul (1) est une faute. (2) et (3) sont exemptes ici, couple par couple, avec leur raison — et la
# section 8 NOTE toute exemption qui ne matche plus rien, pour qu'une exemption perimee ne survive
# pas en silence (meme mecanique que OUVERTS_DECLARES en §12).
RENVOIS_EXEMPTES: dict[tuple[str, str], str] = {
    ("corpus/fetch_corpus.sh", "corpus/txt/fetched"):
        "(2) declare le regime de l'outillage de collecte non redistribue",
    ("rag/retrieve.py", "corpus/txt/fetched"):
        "(2) declare que les requetes de sonde ne sont pas le jeu etiquete de calibration",
    ("rag/index/seuils.json", "corpus/txt/fetched"):
        "(2) champ `outillage_non_redistribue` : explique pourquoi les valeurs sont EN CLAIR ici",
    ("rag/index/documents.json", "corpus/txt/fetched"):
        "(3) provenance, « citation tracable a la page » : `chemin_txt` d'un document du regime "
        "fetched, 1 par document",
    ("rag/index.py", "tools_corpus"):
        "(3) chemin d'ECRITURE du cache d'embedding, hors index committe — pas un renvoi a lire",
    ("rag/embed_server.py", "_embed_server.log"):
        "(3) chemin de son propre journal, ecrit a l'execution",
}
# RENVOIS-DECLARES-FIN

ECHECS: list[str] = []
NOTES: list[str] = []


def ok(msg: str) -> None:
    print(f"  ok    {msg}")


def echec(msg: str) -> None:
    ECHECS.append(msg)
    print(f"  ECHEC {msg}")


def note(msg: str) -> None:
    NOTES.append(msg)
    print(f"  note  {msg}")


def exige(condition: bool, msg: str) -> bool:
    (ok if condition else echec)(msg)
    return bool(condition)


def titre(n: int, texte: str) -> None:
    print(f"\n--- {n}. {texte}")


# =========================================================================================
# Fabrication d'un `res` — la piece qui rend les contre-epreuves possibles sans serveur.
#
# `retrieve.cherche()` fait trois choses : plonger la requete (serveur), fusionner les deux
# classements (pur calcul), decorer les retenus de leur provenance (pur calcul). On garde les deux
# dernieres et on FOURNIT le classement a la main. Les 13 cles de provenance ne sont pas recopiees
# ici : elles sont extraites de la source de `retrieve.py` par la section 0, qui echoue si elles
# divergent. C'est la seule facon de garantir que ce `res` fabrique a la meme forme que le vrai.
# =========================================================================================
CLES_PROVENANCE: tuple[str, ...] = ()


def cles_provenance_de_la_source() -> tuple[str, ...]:
    """Extrait le tuple de cles de provenance de la SOURCE de retrieve.py, sans le recopier."""
    src = (RACINE / "rag" / "retrieve.py").read_text(encoding="utf-8")
    m = re.search(r"\[.provenance.\]\s*=\s*\{k: c\[k\] for k in\s*(\([^)]*\))", src)
    if not m:
        return ()
    val = ast.literal_eval(m.group(1))
    return tuple(val)


def res_fab(idx: retrieve.Index, requete: str,
            dense: list[tuple[int, float]],
            bm25: list[tuple[int, float]] | None = None,
            pen: int | None = None) -> dict:
    """Rend le meme dictionnaire a 8 cles que `retrieve.cherche()`, classement fourni a la main."""
    pen = retrieve.PENALITE_STRUCTURE if pen is None else pen
    l_dense = list(dense)
    l_bm25 = list(bm25 or [])
    ret, ec = retrieve.fusionne(idx, l_bm25, l_dense, k_final=retrieve.K_FINAL,
                               max_par_doc=retrieve.MAX_PAR_DOC, penalite_structure=pen)
    porte = retrieve.statistiques_porte(idx, requete, ret, l_dense, l_bm25)
    for r in ret:
        c = idx.chunks[r["i"]]
        r["provenance"] = {k: c[k] for k in CLES_PROVENANCE}
        r["texte"] = c["texte"]
        r["voisins_meme_page"] = retrieve.voisins_meme_page(idx, r["i"])
    return {
        "requete": requete,
        "jetons": retrieve.jetons(requete),
        "parametres": {"rrf_k": retrieve.RRF_K, "k_pool": retrieve.K_POOL,
                       "k_final": retrieve.K_FINAL, "max_par_doc": retrieve.MAX_PAR_DOC,
                       "penalite_structure": pen},
        "retenus": ret,
        "ecartes": ec,
        "porte": porte,
        "bm25_top": l_bm25[:10],
        "dense_top": l_dense[:10],
    }


def unite_de_n_caracteres(n: int) -> str:
    """Une unite de citation de longueur EXACTEMENT n apres `answer.norm()`, avec des espaces.

    Les espaces comptent : la troncature coupe a la derniere frontiere de mot SOUS le plafond, donc
    une chaine sans espace ne prouverait pas la propriete qu'on veut tester.
    """
    graine = "Semer le mil a raison de deux graines par poquet en juin a Maradi "
    t = (graine * (n // len(graine) + 2))[:n]
    if t.endswith(" "):
        t = t[:-1] + "x"
    assert len(answer.norm(t)) == n, (len(answer.norm(t)), n)
    return t


class Compteur:
    """Faux generateur qui COMPTE ses appels. Signature positionnelle : le site historique
    appelait `generateur(prompt)` sans mot-cle, donc une signature en `**kw` seul aurait rendu un
    TypeError lisible comme « 0 appel » — c'est-a-dire un vert par accident."""

    MARQUEUR = "<<<TEXTE-DU-FAUX-GENERATEUR-JAMAIS-ATTENDU>>>"

    def __init__(self) -> None:
        self.n = 0
        self.prompts: list[str] = []

    def __call__(self, prompt: str = "", **kw: object) -> str:
        self.n += 1
        self.prompts.append(prompt)
        return self.MARQUEUR


# =========================================================================================
# 0. Parite des prompts, perimetre re-derive, cles de provenance extraites de la source
# =========================================================================================
def nombres_de(texte: str) -> list[int]:
    """Les entiers d'un texte, espaces de milliers recolles (« 3 180 » compte pour 3180)."""
    plat = re.sub(r"(?<=\d)[\s  ]+(?=\d)", "", texte)
    return [int(x) for x in re.findall(r"\d+", plat)]


def perimetre_coherent(texte: str, n_chunks: int, n_docs: int) -> bool:
    ns = nombres_de(texte)
    return n_chunks in ns and n_docs in ns


def prompts_de_metadata() -> list[dict]:
    """Lit les deux test_prompts. On ne les retape JAMAIS : ils sont recopies mot pour mot dans le
    formulaire, donc toute divergence entre le code et `metadata.json` est une divergence avec ce
    que le juge lit."""
    meta = json.loads(META.read_text(encoding="utf-8"))
    sortie = []
    for item in meta.get("test_prompts", []):
        if not isinstance(item, dict):
            continue
        if "prompt" in item:
            sortie.append({"id": item.get("prompt_id", "?"), "prompt": item["prompt"]})
        else:  # on NE devine pas un nom de cle JSON : on imprime ce qu'on a trouve et on echoue
            echec(f"test_prompts : pas de cle `prompt`, cles presentes = {sorted(item)}")
    return sortie


def section_0(idx: retrieve.Index) -> list[dict]:
    titre(0, "prompts lus de metadata.json · perimetre re-derive · cles de provenance")
    tps = prompts_de_metadata()
    exige(len(tps) == 2, f"metadata.json porte exactement 2 test_prompts (lu : {len(tps)})")
    for t in tps:
        print(f"        {t['id']} : {t['prompt'][:72]}...")

    # -- parite avec le jeu QCM etiquete a la main, quand il est present.
    # 🔴 On apparie par le TEXTE, pas par l'identifiant : mesure du 20/08, les slugs DIVERGENT
    #    (`calendrier_semis_mil_maradi` dans metadata.json contre `tp1_mil_semis_maradi` dans le jeu
    #    etiquete) alors que les prompts sont identiques mot pour mot. La premiere version appariait
    #    par identifiant et rendait une NOTE « aucun item ne porte cet identifiant » : elle mesurait
    #    une convention de nommage en croyant mesurer la couverture. Le texte est ce qui est recopie
    #    dans le formulaire de soumission, donc c'est le texte qui doit correspondre.
    if QCM.is_file():
        items = [json.loads(l) for l in QCM.read_text(encoding="utf-8").splitlines() if l.strip()]
        fr = sum(1 for it in items if not str(it.get("id", "")).endswith("_en"))
        # items ANGLAIS qui citent en niveau 1 une source FRANCAISE : c'est la traversee de
        # langue effectivement mesuree, et non plus supposee. Derive du jeu, jamais recopie.
        tl = sum(1 for it in items
                 if str(it.get("id", "")).endswith("_en")
                 and str(it.get("niveau_attendu")) == "1"
                 and any(p.get("langue") == "fr" for p in (it.get("provenance") or [])))
        print(f"        jeu etiquete : {len(items)} items ({fr} fr / {len(items) - fr} en)")
        for t in tps:
            jumeaux = [it for it in items
                       if str(it.get("requete", "")).strip() == t["prompt"].strip()]
            exige(len(jumeaux) <= 1,
                  f"au plus un item etiquete porte le texte de {t['id']} (trouve : {len(jumeaux)})")
            if jumeaux:
                ok(f"{t['id']} : parite mot pour mot avec l'item etiquete "
                   f"{jumeaux[0].get('id')}")
            else:
                # Pas un defaut de ce controle : un TROU DE COUVERTURE mesure, et il est oriente.
                note(f"{t['id']} n'a AUCUN jumeau dans le jeu etiquete : ce prompt vitrine "
                     f"(recopie mot pour mot dans le formulaire) n'est couvert que par la section "
                     f"ecrite a la main, pas par les statistiques mesurees sur les {len(items)} "
                     f"items. Le trou restant est la parite MOT POUR MOT, plus la capacite "
                     f"translingue : le jeu est a {fr}/{len(items)} en francais, donc "
                     f"{len(items) - fr} items anglais, dont {tl} en niveau 1 sur une source "
                     f"FRANCAISE — la traversee de langue est donc mesuree, sur d'autres textes "
                     f"que celui-ci")
        # contre-epreuve : la meme comparaison DOIT refuser une chaine mutee d'un seul caractere
        mute = tps[0]["prompt"].replace("Maradi", "Maradii", 1)
        exige(mute.strip() != tps[0]["prompt"].strip(),
              "contre-epreuve : la comparaison de parite refuse une chaine mutee d'un caractere")
        exige(not [it for it in items if str(it.get("requete", "")).strip() == mute.strip()],
              "contre-epreuve : la chaine mutee ne trouve aucun jumeau dans le jeu etiquete")
    else:
        note("jeu QCM etiquete absent : la parite prompt<->item etiquete n'est pas prouvee ici "
             "(elle reste prouvee contre metadata.json, qui est la source du formulaire)")

    # -- les deux nombres du perimetre RE-DERIVES de l'index, jamais recopies d'un tableau de synthese.
    n_chunks = len(idx.chunks)
    n_docs = len({c["doc"] for c in idx.chunks})
    ok(f"index charge : {n_chunks} chunks, {n_docs} documents distincts (comptes, pas recopies)")
    exige(perimetre_coherent(answer.PERIMETRE, n_chunks, n_docs),
          f"answer.PERIMETRE annonce les deux nombres mesures ({n_chunks}, {n_docs})")
    exige(not perimetre_coherent("34 documents indexes, 3 180 passages", n_chunks, n_docs),
          "contre-epreuve : un perimetre annoncant 34 documents est refuse")
    exige(nombres_de("3 180 passages") == [3180],
          "contre-epreuve : « 3 180 » est bien lu comme 3180, pas comme 3 puis 180")

    # -- les 13 cles de provenance, EXTRAITES de la source de retrieve.py.
    cles = cles_provenance_de_la_source()
    if not exige(bool(cles), "le tuple de cles de provenance est extrait de la source de retrieve.py"):
        return tps
    reelles = set(idx.chunks[0])
    exige(set(cles) <= reelles,
          f"les {len(cles)} cles de provenance existent toutes sur un chunk reel")
    exige("provenance" not in cles and "texte" not in cles,
          "contre-epreuve : le tuple extrait n'est pas le dictionnaire du chunk entier")
    return tps


# =========================================================================================
# 1. La porte 2/3 : re-derivee du creux, election NON fondee sur l'AUC, aucun litteral dans le code
# =========================================================================================
def valeurs_de_seuil(s: dict) -> set[float]:
    """Les seuls scalaires qui n'ont PAS le droit d'apparaitre en dur dans le code livre.

    On ne balaie pas `seuils.json` recursivement : il contient aussi des entiers de parametrage
    (K_FINAL=8, RRF_K=60) qui sont legitimement ecrits dans `retrieve.py`. Un detecteur qui les
    confondrait crierait a chaque run, donc ne mesurerait plus rien : un detecteur
    # a faux positifs ne mesure rien.
    """
    v: set[float] = set()
    p23 = s["porte_niveau_2_contre_3"]
    v.add(float(p23["seuil"]))
    creux = p23.get("creux_mesure", {})
    for cle in ("plancher", "plafond"):
        if cle in creux:
            v.add(float(creux[cle]))
    for val in (p23.get("invariance_dans_le_creux") or {}).values():
        if isinstance(val, dict):
            for x in val.values():
                if isinstance(x, (int, float)) and 0.2 < float(x) < 0.8:
                    v.add(float(x))
        elif isinstance(val, (int, float)) and 0.2 < float(val) < 0.8:
            v.add(float(val))
    # ⚠️ `alternatives_ecartees` est niche DANS `porte_niveau_2_contre_3` (pas au niveau racine) et
    # sa cle de seuil est `seuil_zero_fuite`, pas `seuil` : la premiere version lisait
    # `s["alternatives_ecartees"][..]["seuil"]` et surveillait donc SILENCIEUSEMENT zero valeur
    # ecartee. Un detecteur qui ne surveille rien passe au vert exactement comme un code propre.
    for alt in (p23.get("alternatives_ecartees") or {}).values():
        if isinstance(alt, dict) and isinstance(alt.get("seuil_zero_fuite"), (int, float)):
            if 0.2 < float(alt["seuil_zero_fuite"]) < 0.8:
                v.add(float(alt["seuil_zero_fuite"]))
    # Les seuils de promotion 1/2 MESURES PUIS REJETES (la porte 1/2 n'est pas un scalaire) :
    # en ecrire un en dur serait ressusciter une option que la mesure a ecartee.
    p12 = s.get("porte_niveau_1_contre_2") or {}
    for st in (p12.get("cout_du_zero_promotion_par_statistique") or {}).values():
        if isinstance(st, dict) and isinstance(st.get("seuil"), (int, float)):
            if 0.2 < float(st["seuil"]) < 0.8:
                v.add(float(st["seuil"]))
    return v


def litteraux_de_seuil(src: str, interdites: set[float]) -> tuple[list[str], list[str]]:
    """Classe les apparitions de seuil dans une source Python. Rend (fautes, mentions_legitimes).

    Un NUMBER egal a un seuil est une faute : le code lirait un scalaire au lieu du fichier genere.
    Une mention en COMMENT/STRING est LEGITIME (retrieve.py explique sa porte dans sa docstring),
    mais seulement si elle est EXACTE : une valeur commentee qui a derive est un renvoi faux, la
    famille d'erreur la plus tenace de ce projet.
    """
    fautes: list[str] = []
    mentions: list[str] = []
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.NUMBER:
            try:
                x = float(tok.string)
            except ValueError:
                continue
            if any(abs(x - y) < 1e-12 for y in interdites):
                fautes.append(f"litteral {tok.string} en dur ligne {tok.start[0]}")
        elif tok.type in (tokenize.STRING, tokenize.COMMENT):
            for m in re.finditer(r"0[.,]4\d{3,6}", tok.string):
                x = float(m.group(0).replace(",", "."))
                if any(abs(x - y) < 1e-12 for y in interdites):
                    mentions.append(f"mention exacte {m.group(0)} ligne {tok.start[0]}")
                else:
                    fautes.append(f"mention PERIMEE {m.group(0)} ligne {tok.start[0]}")
    return fautes, mentions


def section_1() -> dict:
    titre(1, "porte 2/3 re-derivee de seuils.json ; aucun seuil en dur dans le code livre")
    s = answer.charge_seuils()
    p23 = s["porte_niveau_2_contre_3"]
    creux = p23["creux_mesure"]
    seuil, plancher, plafond = float(p23["seuil"]), float(creux["plancher"]), float(creux["plafond"])
    milieu = (plancher + plafond) / 2
    print(f"        seuil={seuil} milieu du creux={milieu} ecart={abs(seuil - milieu):.1e}")
    # L'ecart vaut EXACTEMENT 5e-05 (le seuil est arrondi a 4 decimales) : tester `<= 5e-5` serait
    # une egalite deguisee qui casserait au premier re-arrondi. On imprime les deux valeurs.
    exige(abs(seuil - milieu) <= 1e-4,
          f"le seuil est le milieu du creux mesure a 1e-4 pres ({seuil} vs {milieu})")
    exige(round(plafond - plancher, 4) == round(float(creux["largeur"]), 4),
          f"la largeur declaree du creux est la difference mesuree ({creux['largeur']})")
    exige(plancher < seuil < plafond, "le seuil tombe strictement a l'interieur du creux")

    # -- l'election ne s'est PAS faite sur l'AUC : une alternative fait STRICTEMENT mieux en AUC.
    #    ⚠️ `alternatives_ecartees` est un DICT niche DANS `porte_niveau_2_contre_3`, pas une liste
    #    au niveau racine — la premiere version de ce controle lisait `s["alternatives_ecartees"]`,
    #    obtenait None, et se rabattait sur une NOTE : un controle qui s'excuse au lieu d'echouer
    #    une tolerance qui masque tres exactement le defaut que le controle existe pour attraper.
    #    Chaque alternative est cotee a SON propre point de fonctionnement (`seuil_zero_fuite` = le
    #    plus petit seuil qui ne laisse passer AUCUNE requete hors agriculture), donc la comparaison
    #    des couts est bien de meilleur-a-meilleur, pas biaisee en faveur de l'elue.
    alts = {k: v for k, v in (p23.get("alternatives_ecartees") or {}).items() if isinstance(v, dict)}
    aucs = {k: float(v["auc"]) for k, v in alts.items() if isinstance(v.get("auc"), (int, float))}
    couts = {k: int(v["agricoles_refusees_a_ce_seuil"]) for k, v in alts.items()
             if isinstance(v.get("agricoles_refusees_a_ce_seuil"), (int, float))}
    cout_elu = int(p23["cout_mesure"]["requetes_agricoles_refusees_a_tort"])
    auc_elue = float(p23["auc"])
    exige(bool(aucs) and bool(couts),
          f"les {len(alts)} alternatives ecartees portent auc + cout exploitables")
    if aucs and couts:
        k_auc = max(aucs, key=lambda k: aucs[k])
        k_cout = min(couts, key=lambda k: couts[k])
        print(f"        elue {p23['statistique']} : auc={auc_elue} cout={cout_elu} refus a tort")
        print(f"        meilleure AUC ecartee : {k_auc} auc={aucs[k_auc]} cout={couts.get(k_auc)} · "
              f"meilleur cout ecarte : {k_cout} cout={couts[k_cout]}")
        exige(aucs[k_auc] > auc_elue,
              f"« {k_auc} » fait STRICTEMENT mieux en AUC ({aucs[k_auc]} > {auc_elue}) et a pourtant "
              f"ete ecartee : l'AUC n'a pas decide, le cout mesure a decide")
        exige(couts[k_cout] > cout_elu,
              f"la statistique elue coute moins de refus a tort ({cout_elu}) que la meilleure "
              f"alternative ({k_cout} : {couts[k_cout]})")

    # -- invariance : les 3 points du creux donnent le MEME cout. Un seuil qui bougerait a 1e-4
    #    pres et changerait la decision ne serait pas un seuil, ce serait une coincidence.
    # 🔴 La premiere version asserta `plancher < x < plafond` (interieur STRICT) et a echoue sur
    #    DEUX des trois points. Ce n'etait pas un defaut du fichier genere, c'etait ma premisse :
    #    a) `creux_mesure` publie ses bornes a 4 decimales (`round(plafond_hors, 4)` = 0.4281) tandis
    #       que `invariance_dans_le_creux` publie les seuils OPERANTS a 6 (`round(.., 6)` = 0.428089)
    #       ⇒ l'ecart mesure (1,1e-05 et 2,0e-05) est le quantum d'arrondi, pas une sortie du creux ;
    #    b) et surtout les deux points extremes sont les bornes ELLES-MEMES par construction
    #       (`plafond_hors + 1e-9` et `troisieme_agri`) : l'intervalle sonde est FERME, et c'est
    #       exactement ce qui donne un sens a « invariance dans le creux » — sonder seulement le
    #       milieu ne prouverait rien. Un test d'interieur strict aurait donc ete faux meme sans
    #       aucun arrondi. ⇒ On assert la bonne propriete : les points BALAIENT le creux (chaque
    #       borne est atteinte au quantum pres) et l'ordre est strict entre eux.
    QUANTUM = 5e-05                      # demi-quantum d'un arrondi a 4 decimales
    inv = p23.get("invariance_dans_le_creux") or {}
    couts_inv, points = set(), {}
    for nom, val in inv.items():
        if not isinstance(val, dict):
            continue
        couts_inv.add((val.get("n_agri_refusees"), val.get("n_hors_agri_passees")))
        exige(isinstance(val.get("seuil"), (int, float)),
              f"le point d'invariance « {nom} » porte un seuil numerique")
        if isinstance(val.get("seuil"), (int, float)):
            points[nom] = float(val["seuil"])

    def dans_le_creux(x: float) -> bool:
        """Creux FERME, elargi du seul quantum d'arrondi des bornes publiees (voir ci-dessus)."""
        return plancher - QUANTUM <= x <= plafond + QUANTUM

    exige(len(points) == 3, f"les 3 points du creux sont publies ({sorted(points)})")
    for nom, x in sorted(points.items(), key=lambda kv: kv[1]):
        exige(dans_le_creux(x), f"le point d'invariance « {nom} » ({x}) est dans le creux ferme "
                                f"[{plancher} ; {plafond}] au quantum d'arrondi pres")
    if points:
        bas, haut = min(points.values()), max(points.values())
        print(f"        points sondes : {sorted(points.values())} · creux publie "
              f"[{plancher} ; {plafond}] · quantum {QUANTUM}")
        exige(abs(bas - plancher) <= QUANTUM and abs(haut - plafond) <= QUANTUM,
              f"les points sondes ATTEIGNENT les deux bornes du creux ({bas} / {haut}) : "
              f"l'invariance est mesuree sur tout le creux, pas seulement en son milieu")
        exige(len(set(points.values())) == 3 and bas < points.get("milieu", -1) < haut,
              "les 3 points sont distincts et ordonnes plancher < milieu < plafond")
        exige(abs((haut - bas) - float(creux["largeur"])) <= 1e-4,
              f"l'etendue sondee ({round(haut - bas, 6)}) est la largeur publiee "
              f"({creux['largeur']}) au quantum pres")
        # contre-epreuve : le meme predicat doit REFUSER un point hors du creux. Sans elle,
        # « dans_le_creux » elargi du quantum serait indistinguable d'un predicat toujours vrai.
        exige(not dans_le_creux(plafond + 0.01) and not dans_le_creux(plancher - 0.01),
              "contre-epreuve : un point a 0,01 hors du creux est refuse par le meme predicat")
    exige(len(couts_inv) == 1,
          f"les points d'invariance donnent tous le meme cout : {sorted(couts_inv)}")

    # -- la porte 1/2 reste declarative : le niveau 1 est une verification d'ancre, pas un scalaire.
    exige(str(s["porte_niveau_1_contre_2"]["decision"]).startswith("AUCUN seuil scalaire"),
          "la porte 1/2 declare toujours « AUCUN seuil scalaire »")

    # -- aucun litteral de seuil dans les deux modules livres.
    interdites = valeurs_de_seuil(s)
    print(f"        valeurs interdites en dur : {sorted(interdites)}")
    for chemin in MODULES_LIVRES:
        fautes, mentions = litteraux_de_seuil(chemin.read_text(encoding="utf-8"), interdites)
        exige(not fautes, f"{chemin.name} : aucun seuil en dur ni mention perimee {fautes or ''}")
        if mentions:
            ok(f"{chemin.name} : {len(mentions)} mention(s) EXACTE(s) en commentaire/docstring")
    # 3 contre-epreuves sur des sources fabriquees
    val = sorted(interdites)[0]
    f1, _ = litteraux_de_seuil(f"x = {seuil}\n", interdites)
    f2, m2 = litteraux_de_seuil(f"# la porte 2/3 tranche a {str(seuil).replace('.', ',')}\n", interdites)
    f3, _ = litteraux_de_seuil("# la porte 2/3 tranche a 0,4999\n", interdites)
    exige(bool(f1), "contre-epreuve : un seuil ecrit en NUMBER est refuse")
    exige(not f2 and bool(m2), "contre-epreuve : le meme seuil cite en commentaire est accepte")
    exige(bool(f3), "contre-epreuve : une valeur de seuil PERIMEE en commentaire est refusee")
    print(f"        (plus petite valeur surveillee : {val})")

    # -- les gardes de charge_seuils : exercees sur des copies corrompues, pas raisonnees.
    gardes = [
        ("porte 1/2 denaturee",
         lambda d: d["porte_niveau_1_contre_2"].__setitem__("decision", "seuil scalaire 0.5")),
        ("sens de la porte 2/3 retire",
         lambda d: d["porte_niveau_2_contre_3"].__setitem__("sens", "valeur superieure au seuil")),
        ("repli_si_absente supprime",
         lambda d: d["porte_niveau_2_contre_3"].pop("repli_si_absente", None)),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        temoin = pathlib.Path(tmp) / "temoin.json"
        shutil.copyfile(answer.SEUILS, temoin)
        try:
            answer.charge_seuils(temoin)
            ok("temoin : une copie NON modifiee de seuils.json se charge sans lever")
        except Exception as e:                                   # noqa: BLE001
            echec(f"temoin : une copie intacte leve {type(e).__name__} : {e}")
        for nom, casse in gardes:
            d = json.loads(answer.SEUILS.read_text(encoding="utf-8"))
            casse(d)
            p = pathlib.Path(tmp) / "casse.json"
            p.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8", newline="\n")
            try:
                answer.charge_seuils(p)
                echec(f"garde MUETTE : « {nom} » n'a pas fait lever charge_seuils")
            except Exception as e:                               # noqa: BLE001
                ok(f"garde active : « {nom} » leve {type(e).__name__}")

    # -- parite des parametres de recuperation avec ceux de la calibration.
    fig = s.get("parametres_retrieve_au_moment_de_la_calibration") or {}
    for cle, attendu in fig.items():
        reel = getattr(retrieve, cle, None)
        exige(reel == attendu,
              f"retrieve.{cle} = {reel} est celui de la calibration ({attendu})")
    if not fig:
        note("seuils.json ne fige pas les parametres de retrieve : la calibration n'est pas "
             "rattachable a un reglage, un seuil calibre sous un autre k_final serait un seuil faux")
    return s


# =========================================================================================
# 2. Plafond de citation : re-derive des unites mesurees, coupe a la frontiere de mot
# =========================================================================================
def section_2(idx: retrieve.Index) -> None:
    titre(2, "plafond de citation, troncature, et l'ellipse qui n'entre pas dans le verbatim")
    plafond = answer.PLAFOND_CITATION_C
    ok(f"answer.PLAFOND_CITATION_C = {plafond}")

    longue = unite_de_n_caracteres(431)
    verbatim, affichee, coupee = answer.tronque(longue)
    exige(coupee, "une unite de 431 c est signalee comme tronquee")
    exige(len(verbatim) <= plafond,
          f"le verbatim tronque tient sous le plafond ({len(verbatim)} <= {plafond})")
    plat = answer.norm(longue)
    exige(plat.startswith(verbatim) and not verbatim.endswith(" ")
          and plat[len(verbatim)] == " ",
          "la coupe tombe a une frontiere de mot : le caractere suivant, dans l'unite, est un espace")
    exige(affichee == verbatim + answer.ELLIPSE,
          "la forme AFFICHEE est le verbatim suivi de l'ellipse")
    exige(answer.ELLIPSE not in verbatim,
          "l'ellipse n'entre PAS dans le verbatim : sinon la citation ne serait plus mot pour mot")
    exige(answer.invariant_verbatim(verbatim, longue),
          "le verbatim tronque reste un extrait exact de l'unite")
    exige(not answer.invariant_verbatim(affichee, longue),
          "contre-epreuve : la forme affichee, elle, n'est PAS un extrait exact (l'ellipse la sort)")

    # frontiere exacte : le plafond n'est pas un « environ ».
    _, _, c300 = answer.tronque(unite_de_n_caracteres(plafond))
    _, _, c301 = answer.tronque(unite_de_n_caracteres(plafond + 1))
    exige(not c300, f"une unite de {plafond} c pile n'est PAS tronquee")
    exige(c301, f"une unite de {plafond + 1} c l'est")

    # re-derivation du cout du plafond sur les unites reellement citables (quand le jeu est la).
    if QCM.is_file():
        items = [json.loads(l) for l in QCM.read_text(encoding="utf-8").splitlines() if l.strip()]
        longueurs: list[int] = []
        for it in items:
            for i in (it.get("bons") or []):
                if 0 <= i < len(idx.chunks):
                    for u in answer.unites_du_chunk(idx.chunks[i]):
                        for typ in (answer.MOTIFS if isinstance(answer.MOTIFS, dict) else {}):
                            if answer.trouve(typ, u):
                                longueurs.append(len(answer.norm(u)))
                                break
        if longueurs:
            mx, med = max(longueurs), statistics.median(longueurs)
            print(f"        unites d'ancre mesurees : n={len(longueurs)} max={mx} c mediane={med} c")
            exige(mx <= plafond,
                  f"le plafond ne coupe AUCUNE unite d'ancre du jeu mesure (marge {plafond - mx} c)")
            note(f"marge du plafond = {plafond - mx} c : non contraignant SUR CE JEU, ce n'est pas "
                 "un ordre de grandeur — une unite plus longue serait tronquee, ce que le code sait "
                 "faire mais qu'aucune mesure ne rend improbable")
        else:
            note("aucune unite d'ancre mesuree : le cout du plafond n'est pas re-derive ici")
    else:
        note("jeu QCM absent : le cout du plafond (max/mediane des unites d'ancre) n'est pas "
             "re-derive ; seul le comportement de troncature est prouve")


# =========================================================================================
# 3. Parite du tokeniseur BM25 entre l'indexation et le runtime — dans les DEUX sens
# =========================================================================================
def section_3(idx: retrieve.Index, echantillon: int, alea: random.Random) -> None:
    titre(3, "parite du tokeniseur BM25 : ce que le runtime produit == ce que l'index contient")
    n = min(echantillon, len(idx.chunks))
    tires = alea.sample(range(len(idx.chunks)), n)

    # sens 1 — tout jeton que le runtime produit doit etre retrouvable dans l'index a ce chunk.
    manquants = 0
    for i in tires:
        for t in set(retrieve.jetons(idx.chunks[i]["texte"])):
            if not idx.contient(t, i):
                manquants += 1
                if manquants <= 3:
                    print(f"        manquant : « {t} » absent du chunk {i}")
    exige(manquants == 0, f"sens 1 : 0 jeton du runtime absent de l'index (sur {n} chunks tires)")

    # sens 2 — et reciproquement : aucun terme en trop cote index. Une seule passe sur les
    # postings, sinon 19 877 termes x n chunks.
    vus: dict[int, set[str]] = {i: set() for i in tires}
    for terme, postes in idx.postings.items():
        for poste in postes:
            i = poste[0] if isinstance(poste, (list, tuple)) else poste
            if i in vus:
                vus[i].add(terme)
    surplus = 0
    for i in tires:
        attendu = set(retrieve.jetons(idx.chunks[i]["texte"]))
        trop = vus[i] - attendu
        surplus += len(trop)
        if trop and surplus <= 3:
            print(f"        en trop : {sorted(trop)[:3]} au chunk {i}")
    exige(surplus == 0, f"sens 2 : 0 terme de l'index absent du runtime (sur {n} chunks tires)")

    # contre-epreuve : un terme fabrique n'est ni dans les postings ni accepte par `contient`.
    faux = "zzqxnotinindex"
    exige(faux not in idx.postings and not idx.contient(faux, tires[0]),
          "contre-epreuve : un terme fabrique est absent de l'index et refuse par `contient`")


# =========================================================================================
# 4. Le test negatif obligatoire passe pour DEUX raisons independantes — assertees SEPAREMENT
# =========================================================================================
def section_4(idx: retrieve.Index) -> None:
    titre(4, "seuil d'intervention chenille : les deux raisons du refus, prouvees separement")
    # Raison A — le corpus ne porte AUCUN seuil d'intervention. C'est une absence MESUREE.
    porteurs = [i for i, c in enumerate(idx.chunks) if answer.trouve("seuil_intervention", c["texte"])]
    exige(not porteurs,
          f"raison A : 0 chunk sur {len(idx.chunks)} porte un seuil d'intervention {porteurs[:3]}")
    # contre-epreuve : le motif n'est pas mort — il attrape les trois formes qu'il vise.
    fabriquees = [
        "Il faut traiter des 5 larves par plant.",
        "Le seuil d intervention est de 10 % de plants attaques.",
        "Traiter a partir de 3 chenilles par pied.",
    ]
    attrapes = [answer.trouve("seuil_intervention", f) for f in fabriquees]
    exige(all(attrapes),
          f"contre-epreuve : le motif attrape les 3 phrases fabriquees {[a[:1] for a in attrapes]}")

    # Raison B — les deux chunks « pieges » (ceux qui parlent de chenilles sur la meme page) ne
    # nomment pas le mais. Independante de A : si le corpus gagnait un seuil demain, B tiendrait.
    pieges: list[int] = []
    if QCM.is_file():
        for ligne in QCM.read_text(encoding="utf-8").splitlines():
            if not ligne.strip():
                continue
            it = json.loads(ligne)
            for p in (it.get("pieges") or []):
                v = p.get("i") if isinstance(p, dict) else p
                if isinstance(v, int):
                    pieges.append(v)
    if not pieges:
        pieges = [1964, 1965]
        note(f"pieges lus par defaut {pieges} : le jeu etiquete est absent, ces indices ne sont "
             "donc pas re-derives de lui")
    total_u = 0
    for i in pieges:
        for u in answer.unites_du_chunk(idx.chunks[i]):
            total_u += 1
            accorde, etage = answer.verifie_entite(idx, idx.chunks[i], u, "mais")
            if accorde:
                echec(f"raison B : le chunk {i} accorde l'entite « mais » (etage {etage}) sur « {u[:60]} »")
                break
        else:
            continue
        break
    else:
        ok(f"raison B : aucune des {total_u} unites de {pieges} ne nomme le mais")
    # contre-epreuve : les MEMES unites accordent bien « mil », donc le refus vient de l'entite
    # demandee, pas d'une verification debranchee.
    accordes_mil = 0
    for i in pieges:
        for u in answer.unites_du_chunk(idx.chunks[i]):
            a, _ = answer.verifie_entite(idx, idx.chunks[i], u, "mil")
            accordes_mil += int(a)
    exige(accordes_mil == total_u,
          f"contre-epreuve : les {total_u} memes unites accordent « mil » ({accordes_mil}) "
          "⇒ la verification d'entite n'est pas debranchee")


# =========================================================================================
# 5. Paire asymetrique : la MEME phrase est citable pour niebe et refusee pour mil
# =========================================================================================
def section_5(idx: retrieve.Index, tp1: str) -> None:
    titre(5, "paire asymetrique : une phrase citable pour une culture, refusee pour l'autre")
    # Le chunk 2093 porte « Il est recommande de semer le niebe 2 semaines apres le premier
    # sarclage du mil. » — il NOMME le mil, il porte une valeur de periode, et il n'est pourtant
    # PAS une source sur la date de semis du mil. C'est le distracteur le plus dangereux du corpus :
    # les deux etages du niveau 1 (entite + valeur) tombent, seule l'attribution est fausse.
    i = 2093
    if not exige(0 <= i < len(idx.chunks), f"le chunk {i} existe dans l'index"):
        return
    apercu = answer.norm(idx.chunks[i]["texte"])[:120]
    print(f"        chunk {i} : {apercu}...")
    exige(answer.nomme_texte("mil", idx.chunks[i]["texte"]),
          f"le chunk {i} NOMME bien le mil (sinon la paire ne serait pas asymetrique)")

    res = res_fab(idx, tp1, [(i, 0.66)], [(i, 9.0)])
    exige([r["i"] for r in res["retenus"]] == [i],
          f"le classement fabrique place bien {i} seul en retenu")

    pour_mil = answer._citations_niveau_1(idx, res, "periode_semis", "mil", tp1)
    pour_niebe = answer._citations_niveau_1(idx, res, "periode_semis", "niebe", tp1)
    exige(pour_mil == [],
          f"le chunk {i} ne produit AUCUNE citation de niveau 1 pour « mil »")
    exige(len(pour_niebe) >= 1,
          f"contre-epreuve : le MEME chunk, la MEME demande, cible « niebe » -> "
          f"{len(pour_niebe)} citation(s) ⇒ le refus vient de l'attribution, pas d'un etage mort")
    if pour_niebe:
        print(f"        citation niebe : {pour_niebe[0].get('citation', '')[:90]}...")


# =========================================================================================
# 6. Zero appel de generation — prouve STRUCTURELLEMENT puis par un compteur
# =========================================================================================
def appels_de_generateur(src: str) -> list[int]:
    """Les lignes ou `generateur` est APPELE (NAME suivi immediatement de `(`).

    On ne cherche pas la sous-chaine « generateur( » : elle apparait dans les docstrings qui
    expliquent precisement que l'appel a ete retire, et un detecteur qui compte les explications
    d'une absence mesure le commentaire, pas le code : on teste la FORME de l'appel, jamais la
    presence d'une sous-chaine.
    """
    lignes: list[int] = []
    precedent = None
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if (precedent is not None and precedent.type == tokenize.NAME
                and precedent.string == "generateur"
                and tok.type == tokenize.OP and tok.string == "("):
            lignes.append(precedent.start[0])
        if tok.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
                            tokenize.DEDENT, tokenize.COMMENT):
            precedent = tok
    return lignes


def section_6(idx: retrieve.Index, tp1: str) -> None:
    titre(6, "la reponse est 100 % extractive : zero appel de generation, structurellement")
    src = (RACINE / "rag" / "answer.py").read_text(encoding="utf-8")
    sites = appels_de_generateur(src)
    exige(not sites, f"answer.py : aucun site d'appel de `generateur` dans le CODE {sites}")
    # contre-epreuves sur des sources FABRIQUEES : le detecteur doit voir un appel, et ne pas en
    # inventer sur une simple affectation ni sur une annotation de type.
    exige(bool(appels_de_generateur("x = generateur(1)\n")),
          "contre-epreuve : un appel fabrique EST detecte")
    exige(not appels_de_generateur("generateur = None\n"),
          "contre-epreuve : une affectation n'est pas prise pour un appel")
    exige(not appels_de_generateur("def f(generateur: object = None) -> None:\n    pass\n"),
          "contre-epreuve : une annotation de parametre n'est pas prise pour un appel")

    # Puis la preuve comportementale. Sans generateur injecte, « 0 appel » serait une tautologie.
    faux = Compteur()
    exige(faux("essai") == Compteur.MARQUEUR and faux.n == 1,
          "le faux generateur sait compter (temoin : sinon « 0 appel » ne prouverait rien)")
    faux = Compteur()
    res = res_fab(idx, tp1, [(1552, 0.62), (2093, 0.55)], [(1552, 8.0)])
    rap = answer.analyse(res, tp1, idx, answer.charge_seuils(), faux)
    exige(rap["generateur_recu"] is True,
          "le rapport atteste que le generateur a bien ete INJECTE (et non simplement absent)")
    exige(rap["appels_generateur"] == 0 and faux.n == 0,
          f"zero appel de generation (rapport={rap['appels_generateur']}, compteur={faux.n})")
    exige(rap.get("extractif_seulement") is True,
          "le rapport declare `extractif_seulement`")
    exige(Compteur.MARQUEUR not in answer.compose(rap),
          "aucun caractere du faux generateur n'atteint le message rendu au lecteur")


# =========================================================================================
# 7. La conjonction francaise « mais » n'est pas lue comme la culture « mais »
# =========================================================================================
def section_7() -> None:
    titre(7, "« mais » conjonction contre « mais » culture — le piege le plus banal du corpus")
    conjonctions = [
        "Le semis est possible des la mi-mai, mais le rendement chute fortement.",
        "Il faut sarcler tot, mais jamais trop pres du poquet.",
    ]
    cultures = [
        "Le maïs est semé en juin.",
        "Maize is sown in June.",
    ]
    for t in conjonctions:
        exige(not answer.nomme_texte("mais", t),
              f"conjonction NON prise pour la culture : « {t[:52]}... »")
    for t in cultures:
        exige(answer.nomme_texte("mais", t),
              f"contre-epreuve : la culture EST reconnue ({answer.mecanisme_texte('mais', t)}) "
              f"« {t[:40]} »")
    # ⚠️ On n'asserte PAS que `veto_espece` rend non-None sur ces quatre phrases : il rend `None`
    # dans les quatre cas (aucun binome rival n'y figure). L'asserter serait un test qui echoue pour
    # la mauvaise raison — et le corriger « jusqu'a ce qu'il passe » fabriquerait un faux veto.
    for t in conjonctions + cultures:
        v = answer.veto_espece("mais", t)
        if v is not None:
            echec(f"veto d'espece inattendu ({v}) sur « {t[:52]}... »")
    ok("aucun veto d'espece parasite sur les 4 phrases (le veto exige un binome, absent ici)")


# =========================================================================================
# 8. Balayage PERMANENT des renvois : un livrable ne cite jamais un chemin absent du depot public
# =========================================================================================
_MARQ_DEBUT = "RENVOIS-" + "DECLARES-DEBUT"
_MARQ_FIN = "RENVOIS-" + "DECLARES-FIN"
SUFFIXES_TEXTE = {".py", ".json", ".md", ".sh", ".yaml", ".yml", ".txt", ".gitattributes",
                  ".jsonl", ".dockerfile", ""}


def lignes_hors_region(texte: str):
    """Numerote les lignes en SAUTANT la region auto-exemptee, bornes comprises."""
    dedans = False
    for n, ligne in enumerate(texte.splitlines(), 1):
        if _MARQ_DEBUT in ligne:
            dedans = True
            continue
        if _MARQ_FIN in ligne:
            dedans = False
            continue
        if not dedans:
            yield n, ligne


def renvois_dans(texte: str, aiguilles: list[str]) -> list[tuple[int, str]]:
    """Cherche chaque aiguille, SANS egard a la casse.

    La casse n'est pas un detail de confort : sensible a la casse, le balayage laissait passer la
    MEME faute a une majuscule pres — un nom de fichier capitalise autrement, un pseudo capitalise,
    un renvoi ecrit en capitales. Un detecteur qu'une majuscule desarme mesure l'orthographe de la
    faute, pas la faute.

    ⚠️ Cette docstring n'ILLUSTRE aucun des cas ci-dessus, et ce n'est pas de la pudeur : la
    section 8 balaye ce fichier-ci comme les autres, donc une aiguille ecrite ici EN EXEMPLE est un
    echec — mesure du 20/08, elle a refuse trois exemples que je venais d'ecrire dans cette
    docstring meme. Un controle qui s'exempterait pour citer ce qu'il interdit ne vaut rien.
    """
    trouves: list[tuple[int, str]] = []
    for n, ligne in lignes_hors_region(texte):
        bas = ligne.lower()
        for a in aiguilles:
            if a.lower() in bas:
                trouves.append((n, a))
    return trouves


def est_exempte(rel: str, aiguille: str) -> bool:
    """Vrai si le couple (fichier, aiguille) est exempte NOMMEMENT. Etroit par construction : ni le
    fichier seul ni l'aiguille seule n'exemptent quoi que ce soit — la section 8 le contre-eprouve."""
    return (rel, aiguille) in RENVOIS_EXEMPTES


def fichiers_committables() -> list[pathlib.Path]:
    """Ce que le depot PUBLIC portera : suivis + non suivis non ignores. Par l'ACTE (git), jamais
    par une liste ecrite a la main qui se perimerait au prochain fichier ajoute."""
    sortie: list[pathlib.Path] = []
    for cmd in (["git", "ls-files"], ["git", "ls-files", "--others", "--exclude-standard"]):
        try:
            brut = subprocess.run(cmd, cwd=RACINE, capture_output=True, text=True,
                                  encoding="utf-8", timeout=60)
        except Exception as e:                                    # noqa: BLE001
            echec(f"impossible d'enumerer les fichiers committables ({cmd[1:]}) : {e}")
            return []
        if brut.returncode != 0:
            echec(f"git {cmd[1:]} a echoue : {brut.stderr.strip()[:120]}")
            return []
        for rel in brut.stdout.splitlines():
            rel = rel.strip()
            if not rel:
                continue
            p = RACINE / rel
            if p.name in RENVOIS_HORS_PERIMETRE or rel in RENVOIS_HORS_PERIMETRE:
                continue
            if p.is_file() and (p.suffix.lower() in SUFFIXES_TEXTE):
                sortie.append(p)
    return sorted(set(sortie))


def section_8() -> None:
    titre(8, "balayage permanent : aucun fichier committable ne cite un chemin hors du depot public")
    moi = pathlib.Path(__file__).read_text(encoding="utf-8")
    # L'exemption doit etre delimitee EXACTEMENT une fois, sinon elle exempterait plus que sa region.
    exige(moi.count(_MARQ_DEBUT) == 1 and moi.count(_MARQ_FIN) == 1,
          "les deux marqueurs de la region auto-exemptee apparaissent exactement une fois")
    print(f"        aiguilles ({len(RENVOIS_INTERDITS)}) : {RENVOIS_INTERDITS}")
    print(f"        hors perimetre, declare : {RENVOIS_HORS_PERIMETRE}")

    fichiers = fichiers_committables()
    if not exige(bool(fichiers), "la liste des fichiers committables est obtenue de git"):
        return
    print(f"        {len(fichiers)} fichier(s) texte committable(s) balaye(s)")
    print(f"        exemptions nommees ({len(RENVOIS_EXEMPTES)} couples) :")
    for (f, a), raison in sorted(RENVOIS_EXEMPTES.items()):
        print(f"          {f} / {a} — {raison}")

    total, utilisees = 0, set()
    for p in fichiers:
        try:
            texte = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        rel = p.relative_to(RACINE).as_posix()
        trouves, exemptes = [], 0
        for n, a in renvois_dans(texte, RENVOIS_INTERDITS):
            if est_exempte(rel, a):
                utilisees.add((rel, a))
                exemptes += 1
            else:
                trouves.append((n, a))
        if exemptes:
            ok(f"{rel} : {exemptes} mention(s) exemptee(s) nommement")
        if trouves:
            total += len(trouves)
            apercu = ", ".join(f"{a}:{n}" for n, a in trouves[:6])
            echec(f"{rel} cite {len(trouves)} chemin(s) hors depot public — {apercu}")
    if not total:
        ok("0 renvoi pendant vers un chemin absent du depot public")
    for couple in sorted(set(RENVOIS_EXEMPTES) - utilisees):
        note(f"exemption qui n'exempte plus rien : {couple[0]} / {couple[1]} — mettre a jour la "
             f"table plutot que de laisser une exemption devenue muette")

    # contre-epreuves : le balayage doit VOIR une faute fabriquee hors region, et NE PAS la voir
    # dans la region declaree. L'aiguille n'est pas ecrite en litteral ici — elle est prise dans la
    # liste, sinon ce fichier porterait une occurrence de plus a exempter.
    a = RENVOIS_INTERDITS[0]
    dehors = "# ceci renvoie a " + a + "/quelque_chose.py\n"
    dedans = f"# {_MARQ_DEBUT}\n# renvoi a {a}/x.py\n# {_MARQ_FIN}\n"
    exige(bool(renvois_dans(dehors, RENVOIS_INTERDITS)),
          "contre-epreuve : un renvoi fabrique HORS region est bien detecte")
    exige(not renvois_dans(dedans, RENVOIS_INTERDITS),
          "contre-epreuve : le meme renvoi DANS la region declaree est exempte")
    # ... et l'exemption doit etre ETROITE : le meme couple sur un AUTRE fichier reste une faute.
    # Sans cette contre-epreuve, une exemption trop large serait indistinguable d'un code propre.
    if RENVOIS_EXEMPTES:
        f_ex, a_ex = sorted(RENVOIS_EXEMPTES)[0]
        exige(est_exempte(f_ex, a_ex), f"contre-epreuve : le couple declare ({f_ex} / {a_ex}) "
                                      f"est bien exempte")
        exige(not est_exempte("rag/un_fichier_qui_n_existe_pas.py", a_ex),
              f"contre-epreuve : la meme aiguille dans un AUTRE fichier n'est PAS exemptee")
        autre = next((x for x in RENVOIS_INTERDITS if x != a_ex), None)
        if autre:
            exige(not est_exempte(f_ex, autre),
                  f"contre-epreuve : une AUTRE aiguille dans le fichier exempte n'est pas exemptee")

    # Contre-epreuves de la famille (c), ajoutees le 20/08 parce que son aiguille etait plus etroite
    # que la faute : elle n'existait qu'avec le signe « ° » et laissait donc passer 15 renvois ecrits
    # en ASCII pur. Le balayage imprimait « 0 renvoi pendant » — un vert qui ne verifiait rien. Les
    # deux orthographes sont maintenant EXERCEES une par une, prises dans la liste (jamais ecrites en
    # litteral, sinon ce fichier porterait la faute qu'il interdit).
    fam_c = [x for x in RENVOIS_INTERDITS if x.lower().startswith(("erreur", "erreurs"))]
    exige(len(fam_c) == 4, f"la famille (c) porte ses DEUX orthographes au singulier et au pluriel "
                           f"({len(fam_c)} aiguilles) — une seule forme suffirait a rendre le vert "
                           f"trompeur")
    for aig in fam_c:
        exige(bool(renvois_dans("# la raison est detaillee en " + aig + "41\n", RENVOIS_INTERDITS)),
              f"contre-epreuve : un renvoi fabrique dans l'orthographe {aig!r} est bien detecte")
    # ... et l'elargissement ne doit RIEN attraper de legitime. C'est la moitie qui manque le plus
    # souvent : une aiguille elargie sans test de faux positif deplace le probleme au lieu de le
    # regler (un detecteur qui crie a tort ne mesure rien). La tournure ci-dessous est du francais
    # courant dans nos propres commentaires, et l'aiguille exige un ESPACE ou un signe, pas une
    # apostrophe — donc elle doit passer.
    faux = "# une erreur n'est pas une mesure, une absence de preuve n'est pas une preuve d'absence\n"
    exige(not renvois_dans(faux, RENVOIS_INTERDITS),
          "contre-epreuve : la tournure « erreur n'est pas » n'est PAS un renvoi — l'elargissement "
          "de l'aiguille n'a pas injecte de faux positif")


# =========================================================================================
# 9. Non-regression du test_prompt 1 — sur le JEU DE CITATIONS, jamais sur le niveau
# =========================================================================================
def contient_plie(aiguille: str, foin: str) -> bool:
    """Recherche accents replies 1->1, cote aiguille ET cote foin : « niebe » doit trouver
    « niébé », et « accumulated rainfall » doit se trouver dans un texte anglais indifferemment."""
    return answer.plie_1a1(aiguille).lower() in answer.plie_1a1(foin).lower()


def citations_du_rapport(rap: dict) -> list[dict]:
    out: list[dict] = []
    for d in rap.get("demandes", []):
        out.extend(d.get("citations") or [])
        r = d.get("reserve")
        if r and r.get("voisin"):
            pass  # le voisin d'une reserve n'est PAS une citation de niveau 1 : il est traite en §10
    return out


def section_9(idx: retrieve.Index, rec: retrieve.Recuperateur, tp1: str, seuils: dict) -> None:
    titre(9, "test_prompt 1 : non-regression sur le JEU DE CITATIONS (le niveau est aveugle)")
    res = rec.cherche(tp1)
    rap = answer.analyse(res, tp1, idx, seuils, None)
    print(f"        niveau(x) global : {rap['niveau_global']} · "
          f"{len(rap['demandes'])} demande(s) · porte={rap['porte_2_3']['valeur']}")
    # 🔴 Le NIVEAU ne prouve rien ici : il vaut [1, 2] AVEC comme SANS la regle d'attribution
    # (mesure C8). Une non-regression qui s'asserterait dessus serait verte le jour ou le chunk
    # niebe reviendrait citer le mil. C'est le jeu de citations qui porte la propriete.
    cites = citations_du_rapport(rap)
    exige(bool(cites), f"tp1 produit au moins une citation de niveau 1 ({len(cites)})")
    exige(all(c["chunk"] != 2093 for c in cites),
          "AUCUNE citation ne vient du chunk 2093 (la phrase niebe qui nomme le mil)")
    demandes_mil = [d for d in rap["demandes"] if d.get("cible") == "mil"]
    for d in demandes_mil:
        for c in (d.get("citations") or []):
            exige(not contient_plie("semer le niebe", c["citation"]),
                  f"aucune citation de mil ne porte « semer le niebe » (chunk {c['chunk']})")
    ancre = [c for c in cites if c["chunk"] == 1552]
    exige(bool(ancre), "l'ancre de juin (chunk 1552, catalogue CNS page 11) est bien citee")
    if ancre:
        print(f"        ancre : {ancre[0]['citation'][:100]}  "
              f"[{ancre[0]['provenance']['editeur']} {ancre[0]['provenance']['annee']} "
              f"p.{ancre[0]['provenance']['page']}]")
    # chaque citation reste verbatim ET sous le plafond — sinon le mot « verbatim » est faux.
    for c in cites:
        ch = idx.chunks[c["chunk"]]
        exige(answer.invariant_verbatim(c["citation"], ch["texte"]),
              f"citation du chunk {c['chunk']} : invariant verbatim tenu")
        exige(len(c["citation"]) <= answer.PLAFOND_CITATION_C,
              f"citation du chunk {c['chunk']} : {len(c['citation'])} c <= plafond")

    # la moitie ABSENTE : le cumul pluviometrique. Elle doit sortir en niveau 2 SITUE — pas une
    # absence nue. C'est l'arbitrage du 18/08 (option (a)) et il est recopie mot pour mot dans le
    # formulaire, donc c'est un test de non-regression obligatoire, pas une propriete esperee.
    pluie = [d for d in rap["demandes"] if d.get("type") == "cumul_pluie"]
    if not exige(bool(pluie), "la demande de cumul pluviometrique est bien reconnue dans tp1"):
        return
    d = pluie[0]
    exige(d["niveau"] == 2, f"le cumul pluviometrique sort en niveau 2 (lu : {d['niveau']})")
    v = d.get("voisin")
    if not exige(bool(v), "le niveau 2 porte un VOISIN documente (absence SITUEE, pas nue)"):
        return
    print(f"        voisin : chunk {v['chunk']} · {v['citation'][:90]}")
    exige(v["chunk"] == 1456,
          f"le voisin est le chunk 1456 (oar_57448 p.8, sorgho/Mali, CC-BY) — lu : {v['chunk']}")
    exige(contient_plie("accumulated rainfall", v["citation"]),
          "le voisin cite bien la valeur documentee la plus proche, ecrite en ANGLAIS "
          "— l'absence n'etait etablie qu'en francais, et le fait est ecrit en anglais")
    exige(v.get("invariant_verbatim") is True, "le voisin est verbatim, lui aussi")
    exige(bool(d.get("ecarts")),
          f"les ecarts a la question sont NOMMES : {d.get('ecarts')}")


# =========================================================================================
# 10. test_prompt 2 : niveau 1 + reserve de niveau 2, sources francaises pour une question anglaise
# =========================================================================================
def section_10(idx: retrieve.Index, rec: retrieve.Recuperateur, tp2: str, seuils: dict) -> None:
    titre(10, "test_prompt 2 : question anglaise, sources francaises, reserve de niveau 2 greffee")
    res = rec.cherche(tp2)
    rap = answer.analyse(res, tp2, idx, seuils, None)
    cites = citations_du_rapport(rap)
    print(f"        niveau(x) : {rap['niveau_global']} · {len(cites)} citation(s)")
    exige(bool(cites), "tp2 produit au moins une citation de niveau 1")
    langues = {c["provenance"].get("langue") for c in cites}
    print(f"        langues des sources citees : {sorted(x for x in langues if x)}")
    exige("fr" in langues,
          "au moins une source FRANCAISE repond a la question anglaise (recuperation translingue)")
    for c in cites:
        exige(answer.invariant_verbatim(c["citation"], idx.chunks[c["chunk"]]["texte"]),
              f"citation du chunk {c['chunk']} : invariant verbatim tenu")

    # la reserve : arbitrage du 19/08, option (b). Le niveau 1 repond, et la meme reponse porte la
    # mesure documentee que la question demandait sans que le corpus la chiffre pour ce cas.
    res_avec = [d for d in rap["demandes"] if d.get("reserve")]
    if not exige(bool(res_avec), "au moins une demande porte une RESERVE de niveau 2"):
        return
    r = res_avec[0]["reserve"]
    v = r.get("voisin")
    if not exige(bool(v), "la reserve porte un voisin documente"):
        return
    print(f"        reserve : chunk {v['chunk']} · {v['citation'][:90]}")
    exige(v["chunk"] == 2094,
          f"le voisin de la reserve est le chunk 2094 — lu : {v['chunk']}")
    exige(v["chunk"] != 755,
          "et ce n'est PAS le chunk 755 (le distracteur mesure du meme corpus)")
    exige(contient_plie("arracher et detruire les plants malades", v["citation"]),
          "la reserve cite bien la mesure prophylactique documentee")
    exige(v.get("invariant_verbatim") is True, "le voisin de la reserve est verbatim")


# =========================================================================================
# 11. Refus reel — niveau 3, perimetre nomme, et AUCUN appel de generation
# =========================================================================================
def section_11(idx: retrieve.Index, rec: retrieve.Recuperateur, seuils: dict) -> None:
    titre(11, "refus reel : le niveau 3 est du CODE en amont du modele, pas une consigne de prompt")
    hors = "Quelle est la capitale de la Mongolie et quel est son PIB par habitant ?"
    faux = Compteur()
    res = rec.cherche(hors)
    rap = answer.analyse(res, hors, idx, seuils, faux)
    p = rap["porte_2_3"]
    print(f"        porte : {p['statistique']}={p['valeur']} seuil={p['seuil']} "
          f"franchie={p['franchie']}")
    exige(rap["niveau_global"] == 3,
          f"niveau_global == 3 (entier, pas une liste) — lu : {rap['niveau_global']!r}")
    exige(p["franchie"] is False, "la porte 2/3 n'est PAS franchie")
    exige(rap["demandes"] == [], "aucune demande n'est instruite sous le seuil bas")
    exige(rap.get("refus", {}).get("perimetre") == answer.PERIMETRE,
          "le refus NOMME le perimetre documente au lieu de refuser sans rien dire")
    exige(rap["generateur_recu"] is True and rap["appels_generateur"] == 0 and faux.n == 0,
          "le generateur etait injecte et n'a PAS ete appele (mesure du 18/08 : 0 refus sur 18 "
          "generations nues portant « Si tu ne sais pas, dis-le » ⇒ le refus ne peut pas etre "
          "delegue au modele)")
    msg = answer.compose(rap)
    exige(Compteur.MARQUEUR not in msg, "le message de refus ne contient rien du generateur")
    print(f"        message : {msg.splitlines()[-1][:110]}")


# =========================================================================================
# 12. Rejeu du jeu etiquete a la main — la seule chose qui distingue « mauvais label » de « mauvais
#     classement ». Sans lui, une marge negative accuse indifferemment le systeme ou l'etiquette.
# =========================================================================================
# 🔴 VIDE depuis le 19/08 — les deux manques declares (`niebe_dates_zones`, `mil_cycle_hkp3`) sont
#    fermes, et ce n'est PAS l'egalite des niveaux qui les a fermes : les deux affichaient deja le
#    bon niveau tout en le tirant du mauvais chunk. Ce qui les ferme est mesure, et la mesure est
#    RECOPIEE ICI plutot que citee par renvoi — un renvoi vers un artefact que le lecteur du depot
#    ne peut pas ouvrir transforme une mesure en « crois-moi » :
#      * les 4 croisements (ancre dans le chunk cite, ancre dans la citation, chunk dans `bons`,
#        balayage des 3 180 chunks porteurs de l'ancre) ont distingue « la recuperation s'est
#        reparee » de « l'etiquette etait trop etroite ». Verdict : `niebe_dates_zones` est le
#        premier cas, `mil_cycle_hkp3` le second — son etiquette designait i1746, jamais retenu,
#        alors que la citation servie vient de i1571 ; elle a ete elargie a [1571, 1746] sur le
#        critere « porte l'ancre ET nomme la variete », qui exclut i1732/i1733/i1734 (meme ancre,
#        variete non nommee) ;
#      * les deux bras de chaque regle ecrite pour les fermer : le veto de variete (portee CHUNK)
#        retire 2 citations sur `mil_cycle_hkp3`, la couture de meme page en ajoute 1 sur
#        `niebe_dates_zones`, et l'ancre etiquetee est servie DANS une citation dans les deux cas.
#        La section 15 rejoue ces deux bras ici meme, donc la batterie ne depend d'aucun artefact
#        externe pour le prouver.
#    ⇒ Toute divergence future sur ces items est desormais un ECHEC DUR, plus une note. L'ensemble
#    reste en place (et non supprime) parce que c'est le mecanisme qui permettra de declarer un
#    prochain manque sans transformer la batterie en rouge permanent — mais un ensemble vide ne peut
#    exempter personne, ce qui est exactement l'etat voulu.
#    ⚠️ Residu MESURE qui n'est PAS ferme, declare pour REPORT.md et non masque ici : le chunk i1746
#    nomme HKP3 une fois mais porte des lignes de tableau appartenant a d'AUTRES varietes. La garde
#    de variete est donc necessaire, pas suffisante — le decoupage detruit la liaison
#    ligne<->variete (meme classe que la limite culture<->dose declaree au D4).
OUVERTS_DECLARES: set[str] = set()
REPLI_QCM = [
    {"id": "tp1_calendrier_semis_mil_maradi", "niveau_attendu": "1+2"},
    {"id": "absence_seuil_chenille", "niveau_attendu": "3"},
]


def niveaux_attendus(brut: object) -> list[int]:
    if isinstance(brut, list):
        return [int(x) for x in brut]
    if isinstance(brut, int):
        return [brut]
    return [int(x) for x in str(brut).split("+") if x.strip()]


def section_12(idx: retrieve.Index, rec: retrieve.Recuperateur, seuils: dict) -> None:
    titre(12, "rejeu du jeu etiquete a la main ; les manques declares sont des NOTES, pas des echecs")
    print(f"        manques de recuperation declares ouverts : {sorted(OUVERTS_DECLARES) or 'AUCUN'}"
          f" — toute divergence ci-dessous est un echec dur")
    if not QCM.is_file():
        note("jeu etiquete absent : ce qui reste NON prouve ici = la coherence niveau attendu / "
             "niveau rendu sur les items etiquetes, et la distinction « label faux » / "
             "« classement faux ». "
             "Les sections 9 a 11 couvrent les deux test_prompts et le refus.")
        for it in REPLI_QCM:
            print(f"        (repli autoportant, non rejoue : {it['id']} -> {it['niveau_attendu']})")
        return
    items = [json.loads(l) for l in QCM.read_text(encoding="utf-8").splitlines() if l.strip()]
    ok(f"{len(items)} item(s) etiquete(s) a la main")
    justes = 0
    for it in items:
        ident = str(it.get("id", "?"))
        requete = it.get("requete", "")
        if not requete:
            note(f"{ident} : pas de requete, item ignore")
            continue
        attendu = niveaux_attendus(it.get("niveau_attendu"))
        res = rec.cherche(requete)
        rap = answer.analyse(res, requete, idx, seuils, None)
        niv = rap["niveau_global"]
        obtenu = niv if isinstance(niv, list) else [niv]
        conforme = sorted(obtenu) == sorted(attendu)
        if conforme:
            justes += 1
            ok(f"{ident} : niveau {obtenu} == attendu {attendu}")
            if ident in OUVERTS_DECLARES:
                note(f"{ident} etait declare OUVERT et passe desormais : mettre a jour "
                     "OUVERTS_DECLARES plutot que de laisser une exemption qui n'exempte plus rien")
        elif ident in OUVERTS_DECLARES:
            note(f"{ident} : niveau {obtenu} != attendu {attendu} — manque de recuperation "
                 "DECLARE (label deja verifie juste), suivi hors de cette batterie")
        else:
            echec(f"{ident} : niveau {obtenu} != attendu {attendu}")
    print(f"        {justes}/{len(items)} items conformes")


# =========================================================================================
# 13. PENALITE_STRUCTURE : mesuree a 0 et a 5 avant d'etre figee, vecteur REUTILISE
# =========================================================================================
def section_13(idx: retrieve.Index, rec: retrieve.Recuperateur, tp1: str, tp2: str) -> None:
    titre(13, "PENALITE_STRUCTURE mesuree (0 contre 5) — un reglage figes sans mesure est un reglage devine")
    requetes = [tp1, tp2, "Quelle dose d engrais NPK pour le mil en poquet ?"]
    for q in requetes:
        avant = rec.n_appels_embedding
        vec = rec.plonge_requete(q)
        exige(rec.n_appels_embedding == avant + 1,
              f"un seul appel d'embedding pour cette requete ({avant} -> {rec.n_appels_embedding})")
        # ⚠️ Le MEME vecteur sert les deux reglages : embarquer deux fois deplacerait la marge de
        # 1,8e-03 (plan factoriel du D4) et on lirait un effet de lot comme un effet de penalite.
        avant2 = rec.n_appels_embedding
        r0 = rec.cherche(q, penalite_structure=0, vecteur=vec)
        r5 = rec.cherche(q, penalite_structure=5, vecteur=vec)
        exige(rec.n_appels_embedding == avant2,
              "les deux recherches ont REUTILISE le vecteur (0 appel d'embedding de plus)")
        t0 = [r["i"] for r in r0["retenus"]]
        t5 = [r["i"] for r in r5["retenus"]]
        struct0 = sum(1 for r in r0["retenus"] if r["provenance"].get("structure"))
        struct5 = sum(1 for r in r5["retenus"] if r["provenance"].get("structure"))
        print(f"        « {q[:44]}... »")
        print(f"          pen=0 : {t0}  ({struct0} chunk(s) de structure retenus)")
        print(f"          pen=5 : {t5}  ({struct5} chunk(s) de structure retenus)")
        exige(struct5 <= struct0,
              f"la penalite ne fait jamais REMONTER un chunk de structure ({struct5} <= {struct0})")
        if t0 == t5:
            note(f"penalite sans effet mesurable sur « {q[:40]}... » — attendu : le levier est "
                 "RESIDUEL (203 chunks marques sur 3 180, 0 marge recuperee a 700)")


# =========================================================================================
# 14. Degradation « socle committed seul » : que reste-t-il sans les 26 documents non redistribues ?
# =========================================================================================
def section_14(idx: retrieve.Index, rec: retrieve.Recuperateur, tp1: str, tp2: str,
               seuils: dict) -> None:
    titre(14, "degradation « socle committed seul » — ce que la reponse deviendrait sans le bucket fetched")
    for q in (tp1, tp2):
        res = rec.cherche(q)
        garde = [r for r in res["retenus"] if r["provenance"].get("regime") == "committed"]
        res2 = dict(res)
        res2["retenus"] = garde
        rap2 = answer.analyse(res2, q, idx, seuils, None)
        cites = citations_du_rapport(rap2)
        print(f"        « {q[:44]}... » : {len(garde)}/{len(res['retenus'])} retenus committed "
              f"-> niveau {rap2['niveau_global']}, {len(cites)} citation(s)")
        exige(all(c["provenance"].get("regime") == "committed" for c in cites),
              "aucune citation ne vient d'un document `fetched` dans ce mode degrade")
        for c in cites:
            exige(answer.invariant_verbatim(c["citation"], idx.chunks[c["chunk"]]["texte"]),
                  f"citation committed du chunk {c['chunk']} : invariant verbatim tenu")
        if not cites:
            note("0 citation en socle committed seul sur cette requete : c'est la mesure du cout de "
                 "la decision d'indexer les 33 documents en entier (les 21 fiches de conseil direct "
                 "en francais sont dans le bucket `fetched`) — a declarer dans REPORT.md, pas a "
                 "masquer")


# =========================================================================================
# 15. Les QUATRE commutateurs declares de l'etape 4, mesures dans LEURS DEUX BRAS
#     Meme idiome que la section 13 pour PENALITE_STRUCTURE : une regle dont on ne peut plus exhiber
#     l'etat anterieur n'est plus mesurable, elle devient une croyance.
# 🔴 Pourquoi cette section ne se reduit PAS a la section 12 : la 12 n'assert que le NIVEAU, et
#     aucune des trois premieres regles ne change un niveau sur les 14 items D'ALORS (mesure des
#     deux bras, item par item, faite le 20/08 avant d'ecrire cette section ; les 6 items anglais
#     ajoutes le 23/08 n'ont PAS ete re-mesures a deux bras — ils sont mesures regles ACTIVES,
#     ce qui prouve leur niveau, pas l'independance de leur niveau vis-a-vis des regles).
#     Un niveau 1 servi depuis le mauvais chunk y passait donc invisible — c'est exactement ce qui
#     est arrive a `mil_cycle_hkp3`, qui citait six lignes de tableau appartenant a d'AUTRES varietes
#     tout en affichant le bon niveau. Une batterie qui verifie la porte ne verifie pas la piece.
# 🔴 La QUATRIEME (borne 6, `CITATION_EXIGE_CIBLE_DE_LA_REQUETE`) est mesuree sur `tp2` et non sur le
#     jeu etiquete, parce que le defaut qu'elle ferme a ete trouve LA : le prompt vitrine translingue
#     servait une citation nommant un AUTRE ravageur que celui demande. Son bras (d) est donc place
#     AVANT la porte du jeu etiquete — la seule regle dont le cas nomme est un livrable ne doit pas
#     devenir non mesurable parce qu'un fichier d'outillage git-ignore manque.
#     ⚠️ Ce que le bras (d) ne pretend PAS : que la borne 6 ne change aucun niveau sur le jeu
#     etiquete.
#     Ca n'a pas ete mesure item par item ; ce qui l'est, c'est que le niveau de `tp2` ne bouge pas.
# =========================================================================================
COMMUTATEURS = ("EXIGE_VARIETE_DANS_LE_CHUNK", "MAX_COUTURE_PAR_DEMANDE",
                "COUTURE_EXIGE_TEXTE_ABSENT_DE_LA_SOURCE",
                "CITATION_EXIGE_CIBLE_DE_LA_REQUETE")


def sous_bras(idx: retrieve.Index, res: dict, requete: str, seuils: dict, **etats) -> dict:
    """Rejoue `answer.analyse` avec les commutateurs positionnes, puis les REMET a leur valeur
    livree. Deux precautions, chacune pour une faute deja payee ailleurs :
      * le retablissement est dans un `finally` — un bras qui leve ne doit pas laisser le module dans
        un etat different de celui qui sera committe, sinon les sections suivantes mesurent autre
        chose que le livrable sans que rien ne le dise ;
      * un nom de commutateur mal ecrit est refuse par `assert` au lieu de creer un attribut neuf :
        `setattr` sur une faute de frappe est un no-op MUET, donc un bras « sans la regle » qui
        serait en realite « avec » — et un bras muet se lit dans le rapport comme un bras vert.
    ⚠️ Le `res` est passe TEL QUEL d'un bras a l'autre : les quatre commutateurs vivent en AVAL du
    classement. Re-chercher ferait varier une seconde chose (le lot d'embedding deplace la marge de
    1,8e-03, plan factoriel du D4) et on lirait la somme des deux comme l'effet de la regle."""
    avant = {n: getattr(answer, n) for n in COMMUTATEURS}
    try:
        for nom, val in etats.items():
            assert nom in COMMUTATEURS, f"commutateur inconnu : {nom}"
            setattr(answer, nom, val)
        return answer.analyse(res, requete, idx, seuils, None)
    finally:
        for nom, val in avant.items():
            setattr(answer, nom, val)


def cle_citation(c: dict) -> tuple:
    """Identite d'une citation servie : le chunk et son texte replie. Volontairement PAS le rang —
    un rang qui glisse parce qu'une citation fautive a disparu n'est pas un changement de
    comportement, c'est sa consequence."""
    return (c["chunk"], " ".join(c["citation"].split()))


def section_15(idx: retrieve.Index, rec: retrieve.Recuperateur, tp2: str, seuils: dict) -> None:
    titre(15, "les 4 commutateurs de l'etape 4 dans leurs DEUX bras — la section 12 est aveugle a ca")
    for nom in COMMUTATEURS:
        exige(hasattr(answer, nom), f"le commutateur `{nom}` existe dans answer.py")
    print(f"        etat livre : " + " · ".join(f"{n}={getattr(answer, n, '?')!r}"
                                                for n in COMMUTATEURS))

    # --- (d) borne 6 : la citation nomme la CIBLE SANITAIRE de la requete -----------------------
    # Place AVANT la porte du jeu etiquete : le cas nomme de cette regle est `tp2` lui-meme, un
    # livrable, donc son bras ne doit pas dependre d'un fichier d'outillage git-ignore.
    res = rec.cherche(tp2)
    rap_a = sous_bras(idx, res, tp2, seuils)
    rap_s = sous_bras(idx, res, tp2, seuils, CITATION_EXIGE_CIBLE_DE_LA_REQUETE=False)
    avec = citations_du_rapport(rap_a)
    sans = citations_du_rapport(rap_s)
    cl_a = {cle_citation(c) for c in avec}
    retirees = [c for c in sans if cle_citation(c) not in cl_a]
    demandees = set(answer._cibles_sanitaires(tp2, generiques=False))
    print(f"        (d) borne 6 : {len(avec)} citation(s) avec, {len(sans)} sans "
          f"-> {len(retirees)} refusee(s) ; cible(s) demandee(s) : {sorted(demandees) or '(aucune)'}")
    exige(bool(demandees),
          "le prompt vitrine nomme bien une cible sanitaire SPECIFIQUE : sans elle la borne 6 "
          "s'ouvre par construction (porte 3 du predicat) et ce bras mesurerait une regle inerte")
    exige(bool(retirees),
          "la borne 6 refuse quelque chose de reel sur le prompt vitrine : son refus est MUET (la "
          "citation manque, rien ne le dit), donc sans ce bras « elle ferme le defaut hors-sujet de "
          "tp2 » serait une croyance — une defense qu'on n'a jamais VUE refuser est indistinguable "
          "d'une defense debranchee")
    for c in retirees:
        nommees = set(answer._cibles_sanitaires(c["citation"], generiques=False))
        print(f"            refuse i{c['chunk']} (cible(s) nommee(s) : {sorted(nommees) or '-'}) : "
              f"« {c['citation'][:60]} »")
        exige(not (nommees & demandees),
              f"ce que la borne 6 retire ne nomme AUCUNE des cibles demandees (chunk {c['chunk']}) : "
              "c'est ce qui distingue « elle a retire une citation hors sujet » de « elle a retire "
              "au hasard une citation que les quatre autres controles avaient acceptee »")
    for c in avec:
        nommees = set(answer._cibles_sanitaires(c["citation"], generiques=False))
        exige(bool(nommees & demandees),
              f"la citation SERVIE du chunk {c['chunk']} nomme la cible demandee — sinon la borne 6 "
              "aurait retire le mauvais cote et le defaut vivrait encore dans le livrable")
    # Ce que la borne 6 ne doit PAS faire : degrader le prompt vitrine. Elle mord sur la CITATION,
    # jamais sur le classement, donc le niveau doit survivre a son passage — mesure dans les deux
    # bras, pas suppose.
    print(f"            niveau(x) de tp2 : {rap_s['niveau_global']} sans la borne -> "
          f"{rap_a['niveau_global']} avec")
    exige(rap_a["niveau_global"] == rap_s["niveau_global"],
          "la borne 6 ne deplace AUCUN niveau sur tp2 : elle mord sur le choix de l'unite citee, "
          "jamais sur le classement — si elle deplacait un niveau, elle ferait autre chose que ce "
          "que sa docstring annonce")
    exige(bool(avec), "tp2 garde au moins une citation avec la borne 6 : une regle qui viderait le "
                      "prompt vitrine coute plus qu'elle ne rapporte")
    exige(rap_a.get("appels_generateur", 0) == 0,
          "tp2 reste 100 % extractif avec la borne 6 : le refus d'une unite hors sujet ne doit pas "
          "faire tomber la demande dans une branche qui appelle le modele")

    if not QCM.is_file():
        note("jeu etiquete absent : restent NON prouves ici — que le veto de variete retire des "
             "citations d'un chunk qui ne nomme pas la variete demandee, que la couture est la SEULE "
             "raison pour laquelle l'ancre `niebe_dates_zones` est servie, et que la borne 5 refuse "
             "un contournement de plafond. Aucune autre section ne les couvre. (Le bras (d) de la "
             "borne 6, lui, vient d'etre mesure : il ne depend pas de ce fichier.)")
        return
    items = {}
    for ligne in QCM.read_text(encoding="utf-8").splitlines():
        if ligne.strip():
            it = json.loads(ligne)
            items[str(it.get("id"))] = it
    besoin = ("mil_cycle_hkp3", "niebe_dates_zones", "mil_microdose_dose")
    absents = [i for i in besoin if i not in items]
    if absents:
        note(f"items absents du jeu etiquete : {absents} — les deux bras ne sont pas mesurables "
             "sans eux (chaque regle a ete ecrite POUR un cas nomme, pas dans l'abstrait)")
        return

    # --- (a) veto de VARIETE, portee CHUNK ------------------------------------------------------
    it = items["mil_cycle_hkp3"]
    q = it["requete"]
    res = rec.cherche(q)
    avec = citations_du_rapport(sous_bras(idx, res, q, seuils))
    sans = citations_du_rapport(sous_bras(idx, res, q, seuils, EXIGE_VARIETE_DANS_LE_CHUNK=False))
    noms = answer.variete_demandee(q)
    exige(bool(noms), f"la requete demande bien une variete NOMMEE, sinon le veto serait inerte "
                      f"et son bras « sans » ne prouverait rien — lu : {noms}")
    cl_a = {cle_citation(c) for c in avec}
    cl_s = {cle_citation(c) for c in sans}
    print(f"        (a) veto variete {noms} : {len(avec)} citation(s) avec, {len(sans)} sans")
    exige(cl_a < cl_s, f"le veto RETIRE des citations et n'en ajoute aucune "
                       f"({len(cl_a)} strictement inclus dans {len(cl_s)})")
    # la PREUVE, pas le compte : chaque citation retiree vient d'un chunk qui ne nomme pas la
    # variete. Un veto qui retire le bon nombre de citations pour la mauvaise raison passerait un
    # test de comptage sans rien garantir.
    for c in sorted(cl_s - cl_a):
        exige(not answer.tous_nommes(noms, idx.chunks[c[0]]["texte"]),
              f"citation retiree du chunk {c[0]} : ce chunk ne nomme PAS {noms}")
    ancre = (it.get("ancres") or [""])[0]
    exige(any(contient_plie(ancre, c["citation"]) for c in avec),
          f"l'ancre etiquetee est TOUJOURS servie avec le veto — il coupe le bruit, pas la reponse "
          f"(« {ancre[:44]} »)")
    # ⚠️ Residu MESURE, declare ici pour qu'il ne se redecouvre pas : la garde est necessaire, pas
    # suffisante. Un chunk peut nommer la variete une fois et porter des lignes appartenant a
    # d'autres — le decoupage detruit la liaison ligne<->variete, meme classe que la limite
    # culture<->dose declaree au D4.
    porteurs = [i for i, ch in enumerate(idx.chunks) if answer.tous_nommes(noms, ch["texte"])]
    print(f"            {len(porteurs)} chunk(s) sur {len(idx.chunks)} nomment {noms} ; la garde "
          f"filtre par chunk, elle ne relie pas une LIGNE de tableau a sa variete")

    # --- (b) COUTURE de page ---------------------------------------------------------------------
    it = items["niebe_dates_zones"]
    q = it["requete"]
    ancre = (it.get("ancres") or [""])[0]
    res = rec.cherche(q)
    rap_avec = sous_bras(idx, res, q, seuils)
    avec = citations_du_rapport(rap_avec)
    sans = citations_du_rapport(sous_bras(idx, res, q, seuils, MAX_COUTURE_PAR_DEMANDE=0))
    cousues = [c for c in avec if c.get("couture")]
    retenus = [r["i"] for r in res["retenus"]]
    print(f"        (b) couture : {len(avec)} citation(s) avec, {len(sans)} sans, "
          f"dont {len(cousues)} cousue(s)")
    exige(len(cousues) >= 1, "au moins une couture est servie : sans cela le bras « sans » ne "
                             "prouverait rien, il comparerait deux fois le meme vide")
    # La demande d'ou sort chaque couture, pour que deux coutures LEGITIMES ne se lisent pas comme
    # une ligne dupliquee : le budget est PAR demande, donc le meme couple (chunk, texte) sous deux
    # demandes differentes est conforme, alors que deux fois sous la MEME demande serait un
    # depassement. Sans l'etiquette, les deux cas s'impriment a l'identique et le lecteur doit
    # deviner lequel il a sous les yeux — un rapport qui ressemble a un bug invite a en supposer un.
    # 🔴 Le parcours est PAR DEMANDE et non sur la liste aplatie : `citations_du_rapport` fond les
    #    demandes ensemble, donc y chercher l'origine d'une couture rend les DEUX demandes pour les
    #    DEUX lignes et n'etiquette plus rien. C'est ce cas qui se produit ici (i1646 sert deux fois).
    cousues_par_demande: list[tuple[str, dict]] = []
    for d in rap_avec["demandes"]:
        etiquette = f"{d.get('type')}/{d.get('cible')}"
        n = 0
        for c in (d.get("citations") or []):
            if c.get("couture"):
                n += 1
                cousues_par_demande.append((etiquette, c))
        exige(n <= answer.MAX_COUTURE_PAR_DEMANDE,
              f"budget de couture respecte sur {etiquette} "
              f"({n} <= {answer.MAX_COUTURE_PAR_DEMANDE})")
    # LE fait qui a ferme le manque de recuperation : l'ancre etiquetee n'est servie QUE grace a la
    # couture. Assertee dans les deux sens — presente avec, ABSENTE sans.
    exige(any(contient_plie(ancre, c["citation"]) for c in avec),
          f"l'ancre etiquetee est servie avec la couture (« {ancre[:44]} »)")
    exige(not any(contient_plie(ancre, c["citation"]) for c in sans),
          "et elle est ABSENTE sans la couture — donc la couture est la seule raison pour laquelle "
          "le fait etiquete est servi, ce n'est pas un ornement")
    exige(len(cousues_par_demande) == len(cousues),
          f"chaque couture servie a une demande d'origine identifiee ({len(cousues_par_demande)} = "
          f"{len(cousues)}) — une couture sans demande serait une citation dont on ne peut pas dire "
          "quel budget elle a consomme")
    for pour, c in cousues_par_demande:
        i, src = c["chunk"], c.get("couture_de")
        print(f"            [{pour}] i{i} <- i{src} p.{idx.chunks[i].get('page')} : "
              f"« {c['citation'][:70]} »")
        exige(answer.invariant_verbatim(c["citation"], idx.chunks[i]["texte"]),
              f"couture i{i} : verbatim dans SON chunk (pas dans celui du voisin qui l'a appelee)")
        exige(idx.chunks[i].get("doc") == idx.chunks[src].get("doc")
              and idx.chunks[i].get("page") == idx.chunks[src].get("page"),
              f"couture i{i} : meme document ET meme page que i{src} (borne 2) — une couture entre "
              "pages fabriquerait une citation que la page citee ne porte pas")
        exige(i not in retenus, f"couture i{i} : ce chunk n'est PAS retenu (borne 3), sinon la "
                                "regle doublerait une citation deja classee")
        exige(c.get("cos") is None,
              f"couture i{i} : aucun cosinus — ce chunk n'a pas ete classe, lui attribuer celui du "
              "voisin serait porter sur un passage une mesure faite sur un autre")
        exige(any(cle_citation(x) != cle_citation(c) and x["chunk"] == src for x in avec),
              f"couture i{i} : son appelant i{src} est bien une citation SERVIE (borne 1 : elle "
              "ACHEVE une source deja citee, elle ne PROMEUT jamais)")

    # --- (c) borne 5 : le texte cousu ne doit pas etre DEJA dans sa source -----------------------
    it = items["mil_microdose_dose"]
    q = it["requete"]
    res = rec.cherche(q)
    avec = citations_du_rapport(sous_bras(idx, res, q, seuils))
    sans = citations_du_rapport(sous_bras(idx, res, q, seuils,
                                          COUTURE_EXIGE_TEXTE_ABSENT_DE_LA_SOURCE=False))
    cl_a = {cle_citation(c) for c in avec}
    reprises = [c for c in sans if cle_citation(c) not in cl_a]
    print(f"        (c) borne 5 : {len(avec)} citation(s) avec, {len(sans)} sans "
          f"-> {len(reprises)} refusee(s) par la borne")
    exige(bool(reprises),
          "la borne 5 refuse quelque chose de reel sur ce cas : son refus est MUET (elle ne rend "
          "rien, exactement comme une absence de voisin), donc sans ce bras « elle empeche le "
          "contournement de plafond » serait une croyance — une defense qu'on n'a jamais VUE "
          "refuser est indistinguable d'une defense debranchee")
    for c in reprises:
        src = c.get("couture_de")
        deja = src is not None and answer.invariant_verbatim(c["citation"],
                                                            idx.chunks[src]["texte"])
        print(f"            refuse i{c['chunk']} (source i{src}) : « {c['citation'][:66]} »")
        exige(bool(c.get("couture")),
              f"ce que la borne 5 retire est bien une COUTURE (chunk {c['chunk']}) et non une "
              "citation ordinaire — sinon elle mordrait sur le niveau 1")
        exige(deja, f"le texte refuse est PROUVE deja present dans son chunk source i{src} : c'est "
                    "ce qui distingue « la borne a refuse un contournement de plafond » de « la "
                    "borne a refuse au hasard »")
    # Le chiffre systemique qui justifie la borne, RE-DERIVE ici et non recopie : sans elle, la
    # couture serait un contournement de plafond generalise plutot qu'un achevement.
    # 🔴 Et il est re-derive DANS LES DEUX SENS, parce que le predicat est orientable et que les deux
    #    orientations ne donnent pas le meme compte. Un « X chunks chevauchent un voisin » nu laisse
    #    croire a une grandeur unique : c'est ce flou qui a laisse vivre un « 1 994 (62,7 %) » dans la
    #    docstring de la borne 5, chiffre qui ne se re-derive d'AUCUN des trois predicats. On mesure
    #    donc les trois, on les nomme, et on RELIT la docstring pour la confronter au calcul — un
    #    nombre ecrit dans un livrable que rien ne re-derive est un nombre que personne ne peut
    #    contredire, donc pas une mesure.
    FEN = 60
    queue = tete = union = 0
    for i in range(len(idx.chunks)):
        ti = idx.chunks[i]["texte"]
        q = t = False
        for j in retrieve.voisins_meme_page(idx, i):
            tj = idx.chunks[j]["texte"]
            q = q or answer.invariant_verbatim(ti[-FEN:], tj)
            t = t or answer.invariant_verbatim(tj[:FEN], ti)
            if q and t:
                break
        queue += int(q)
        tete += int(t)
        union += int(q or t)
    n_ch = len(idx.chunks)
    print(f"            chevauchement sur fenetre de {FEN} c, {n_ch} chunks :")
    print(f"              queue de i dans un voisin j : {queue} ({100.0 * queue / n_ch:.1f} %)"
          f"  <- le predicat que nomme la borne 5")
    print(f"              tete d'un voisin j dans i   : {tete} ({100.0 * tete / n_ch:.1f} %)")
    print(f"              union des deux              : {union} ({100.0 * union / n_ch:.1f} %)")
    exige(queue > 0, "le chevauchement est mesure non nul : c'est la raison d'etre de la borne 5")

    def nombre(s: str) -> int:
        for espace in (" ", " ", " "):
            s = s.replace(espace, "")
        return int(s)

    doc = answer._couture_meme_page.__doc__ or ""
    m_q = re.search(r"\*\*([\d   ]+) chunks sur ([\d   ]+)\*\*", doc)
    m_a = re.search(r"voisin dans le chunk\s*:\s*([\d   ]+);\s*union\s*:\s*([\d   ]+)\)",
                    doc)
    exige(m_q is not None and m_a is not None,
          "les chiffres de chevauchement sont RETROUVES dans la docstring de la borne 5 : si le motif "
          "ne matchait plus, la comparaison ci-dessous passerait a vide et le controle se lirait vert "
          "en ne verifiant rien — c'est le defaut que `cles_provenance_de_la_source` a deja paye")
    if m_q and m_a:
        # Contre-epreuve du lecteur lui-meme : sur une docstring FABRIQUEE, il doit rendre le chiffre
        # fabrique et non celui qu'on vient de calculer. Sans elle, un lecteur qui renverrait
        # betement `queue` ferait passer la comparaison quoi qu'il arrive.
        faux = re.search(r"\*\*([\d   ]+) chunks sur ([\d   ]+)\*\*",
                         "**1 234 chunks sur 5 678** ont un voisin")
        exige(faux is not None and nombre(faux.group(1)) == 1234 != queue,
              "contre-epreuve du lecteur de docstring : sur un texte fabrique il rend 1 234, donc il "
              "LIT le fichier et ne recopie pas la valeur calculee")
        exige(nombre(m_q.group(1)) == queue and nombre(m_q.group(2)) == n_ch,
              f"la borne 5 annonce {m_q.group(1).strip()}/{m_q.group(2).strip()} et la mesure rend "
              f"{queue}/{n_ch} : le nombre ecrit dans le livrable est RE-DERIVE, pas recopie")
        exige(nombre(m_a.group(1)) == tete and nombre(m_a.group(2)) == union,
              f"les deux autres orientations annoncees (tete {m_a.group(1).strip()}, union "
              f"{m_a.group(2).strip()}) valent bien {tete} et {union} — les trois predicats sont "
              "nommes ET verifies, aucun ne reste declaratif")


# =========================================================================================
def principal() -> int:
    for flux in (sys.stdout, sys.stderr):
        if hasattr(flux, "reconfigure"):
            flux.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="verification comportementale de retrieve.py + answer.py")
    ap.add_argument("--rapide", action="store_true",
                    help="saute les sections 9 a 15 (les seules qui lancent un serveur)")
    ap.add_argument("--echantillon", type=int, default=8,
                    help="nombre de chunks tires pour la parite du tokeniseur (section 3)")
    args = ap.parse_args()
    alea = random.Random(42)

    global CLES_PROVENANCE
    CLES_PROVENANCE = cles_provenance_de_la_source()

    print(f"index : {INDEX}")
    idx = retrieve.Index(INDEX)
    tps = section_0(idx)
    seuils = section_1()
    section_2(idx)
    section_3(idx, args.echantillon, alea)
    section_4(idx)
    tp1 = tps[0]["prompt"] if tps else ""
    tp2 = tps[1]["prompt"] if len(tps) > 1 else ""
    section_5(idx, tp1)
    section_6(idx, tp1)
    section_7()
    section_8()

    if args.rapide:
        note("--rapide : sections 9 a 15 NON executees. Restent donc NON prouves ici — la "
             "non-regression des deux test_prompts de bout en bout (citations, ancre de juin, "
             "voisin anglais), le refus reel sur une question hors perimetre, le rejeu du jeu "
             "etiquete, la mesure de PENALITE_STRUCTURE, la degradation « committed seul » et les "
             "DEUX bras des quatre commutateurs de l'etape 4 (dont la borne 6, qui n'est couverte "
             "par AUCUNE autre section : la 10 se contente d'au moins une citation). "
             "Les sections 0 a 8 ne lancent aucun serveur : elles ne peuvent pas les couvrir.")
    else:
        with retrieve.Recuperateur() as rec:
            section_9(idx, rec, tp1, seuils)
            section_10(idx, rec, tp2, seuils)
            section_11(idx, rec, seuils)
            section_12(idx, rec, seuils)
            section_13(idx, rec, tp1, tp2)
            section_14(idx, rec, tp1, tp2, seuils)
            section_15(idx, rec, tp2, seuils)
        ok("le serveur d'embedding a ete ferme par le gestionnaire de contexte")

    print()
    print(f"=== {len(ECHECS)} echec(s), {len(NOTES)} note(s)")
    for e in ECHECS:
        print(f"  ECHEC {e}")
    for n in NOTES:
        print(f"  note  {n}")
    return 1 if ECHECS else 0


if __name__ == "__main__":
    raise SystemExit(principal())
