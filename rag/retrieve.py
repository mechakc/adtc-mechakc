"""Bloc D5 — recuperation hybride sur l'index prebati. **AUCUN SEUIL N'EST ECRIT ICI.**

Ce fichier RANGE et TRACE ; il ne DECIDE pas. La politique graduee a 3 niveaux (1 sourcE avec
citation verbatim / 2 utile mais ETIQUETE non source / 3 refus hors agriculture) est calibree
sur une distribution mesuree et consignee dans `rag/index/seuils.json` — `verify_index.py`
verifie d'ailleurs qu'aucun seuil n'a ete glisse dans l'index lui-meme.

⚠️ Une seule des deux frontieres est un seuil. La formulation heritee « il faut deux seuils
cosinus » est REFUTEE par la mesure : la frontiere 2/3 est bien un scalaire (cos 0,4376, milieu
d'un creux mesure), mais la frontiere 1/2 n'en est pas un. Un cosinus mesure une proximite
THEMATIQUE, pas la presence d'un fait — la requete du cumul pluviometrique demande une valeur
ABSENTE du corpus tout en etant maximalement proche de son sujet, donc elle marque PLUS HAUT
que des questions dont la reponse est ecrite noir sur blanc. Mesure sur les 11 statistiques
emises ci-dessous : la meilleure perd encore 5 niveau-1 sur 11 pour ne jamais promouvoir a
tort. Le niveau 1 est donc accorde par VERIFICATION D'ANCRE dans `answer.py`, pas par un seuil.

Pourquoi l'ordre du brief est inverse ici (4a avant la calibration), et ce n'est pas du
confort : la porte de niveau se joue sur le meilleur chunk retenu APRES fusion. Or RRF rend un
score sans unite comparable a un cosinus, donc la statistique de porte ne peut pas etre
calculee avant que la fusion existe. Et surtout, la mesure du 19/08 a montre que le candidat
evident ne marche pas : sur 4 items etiquetes, les bons chunks sortent a cos 0,6143-0,6567 et
les MEILLEURS MAUVAIS chunks a 0,6570-0,7425. Les deux distributions se chevauchent
completement => une porte sur le cosinus du top-1 dense serait quasi aveugle.

Consequence de conception, et c'est le coeur de ce fichier : au lieu de choisir d'avance la
statistique de porte, `cherche()` renvoie SEPT candidats mesurables cote a cote (bloc
`porte`). La calibration mesure laquelle separe reellement les populations et garde celle-la
(mesure : `cos_du_retenu_1`, auc 0,972, repli `cos_max_retenus`). Choisir d'avance serait
supposer ; les emettre toutes est ce que le perimetre autorise.

Cinq contraintes tenues ici, chacune adossee a une mesure et non a un usage :
  1. requete plongee SEULE (lot de 1), pooling `cls` explicite, via `embed_server` — la
     composition du lot deplace la marge dure de 1,8e-03, soit ~4 % de sa valeur (D4). Un
     seuil calibre dans un regime de lot puis applique dans un autre est un seuil faux.
  2. AUCUN prefixe d'instruction sur la requete — mesure M2 du 19/08 : le prefixe de
     bge-small/base-en n'ameliore le rang dans 0 cas sur 4 et le degrade 2 fois (6->20,
     51->157). Son delta de marge moyen positif est un piege : il monte parce que le prefixe
     ecrase tous les cosinus, pas parce qu'il discrimine.
  3. le jeton BM25 est IMPORTE de `index.py`, jamais reecrit. Le repli des accents fusionne
     « mais » et « mais » (la cereale) et un compose a tiret est emis entier ET en morceaux :
     une seconde implementation divergerait EN SILENCE (BM25 simplement moins bon, aucune
     erreur levee). `verify_retrieve.py` doit PROUVER la parite en re-jetonnant des chunks et
     en comparant aux postings livres, pas la supposer.
  4. diversite PAR DOCUMENT — un seul document pese 1 011 chunks sur 3 180 (31,8 % de
     l'index) contre 197 pour le catalogue qui porte tout le conseil direct. Sur la requete
     arachide, les 10 premiers du dense sortent TOUS de ce seul rapport economique.
  5. dedoublonnage par OFFSETS, pas par ressemblance : les chunks se chevauchent d'une phrase
     par construction (`chevauchement_unites: 1`), donc l'intersection des intervalles `off`
     l'etablit exactement (i=1552 et i=1553 partagent « ... 100 cm x 100 cm »).

Ce que ce fichier NE fait pas, volontairement :
  * il ne decide aucun niveau, n'ecrit aucun seuil, n'appelle aucun LLM ;
  * il ne supprime jamais un chunk marque `structure` — il le DEMOTE (D4 : « on marque, on ne
    supprime jamais »), et la penalite est un parametre DECLARE et non mesure, que
    `verify_retrieve.py` doit mesurer a 0 et a sa valeur par defaut avant de la figer.

⚠️ RESERVE NON SOLDEE, a ne pas effacer : `from index import jetons` charge `index.py`, qui
importe `yaml` au niveau module. Aucun effet ici (aucun serveur ouvert, aucun fichier lu a
l'import — verifie), mais si les juges lancent l'appli, l'image doit porter PyYAML en plus de
`llama-server` (invariant 2 du D4). A EXERCER dans l'image au D6/D7, pas a supposer. La
mutualisation du jeton reste le bon choix : une seconde implementation echouerait EN SILENCE,
alors qu'une dependance manquante echoue bruyamment a l'import.
"""
from __future__ import annotations

import bisect
import json
import math
import pathlib
import sys

import numpy as np

RACINE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import embed_server                                    # noqa: E402
# UNE seule implementation du jeton, importee et jamais reecrite (contrainte 3).
# ⚠️ Homonyme a ne pas confondre : `Serveur.jetons()` compte les tokens du modele
# d'embedding ; `index.jetons()` produit les jetons BM25. Deux choses sous un meme nom.
from index import jetons                                # noqa: E402

INDEX = RACINE / "rag" / "index"

# ---- fusion -----------------------------------------------------------------------------
# RRF : score = somme sur les listes de 1/(K + rang). K=60 est la valeur de l'article
# d'origine (Cormack 2009) et n'est PAS mesuree ici — a declarer comme telle. Elle ne
# change pas l'ordre de facon sensible sur des listes courtes ; ce qui la rendrait critique
# serait un pool tres long, que nous n'utilisons pas.
RRF_K = 60

K_POOL = 60          # profondeur de chaque liste avant fusion
K_FINAL = 8          # chunks rendus a `answer.py`
MAX_PAR_DOC = 3      # diversite : plafond par document (contrainte 4)
PENALITE_STRUCTURE = 5   # rangs ajoutes a un chunk `bibliographie|sommaire|mots_cles`

STRUCTURE_DEMOTEE = {"bibliographie", "sommaire", "mots_cles"}


# =========================================================================================
# chargement
# =========================================================================================
class Index:
    """L'index livre, charge une fois. Aucune ecriture, aucun reseau hors 127.0.0.1."""

    def __init__(self, dossier: pathlib.Path = INDEX) -> None:
        self.dossier = dossier
        self.chunks = [json.loads(l) for l in
                       open(dossier / "chunks.jsonl", encoding="utf-8")]
        self.manifeste = json.loads((dossier / "manifest.json").read_text(encoding="utf-8"))
        v = np.load(dossier / "vectors.npy")
        if v.dtype != np.float16:
            raise SystemExit(f"echec  vecteurs stockes en {v.dtype}, float16 attendu")
        if v.shape[0] != len(self.chunks):
            raise SystemExit(f"echec  {v.shape[0]} vecteurs pour {len(self.chunks)} chunks")
        # cast float32 : le cosinus se calcule en float32, la norme stockee est ASSERTEE et
        # jamais reimposee (|norme-1| max mesure = 7,543e-05 au D4).
        self.vecteurs = v.astype(np.float32)
        for k, c in enumerate(self.chunks):
            if c["i"] != k:
                raise SystemExit(f"echec  ligne {k} porte i={c['i']} : appariement i<->vecteur casse")

        b = json.loads((dossier / "bm25.json").read_text(encoding="utf-8"))
        self.postings: dict[str, list[list[int]]] = b["postings"]
        self.doclen: list[int] = b["doclen"]
        self.avgdl: float = b["avgdl"]
        self.k1: float = b["k1"]
        self.b: float = b["b"]
        self.n: int = b["n_chunks"]
        if self.n != len(self.chunks):
            raise SystemExit(f"echec  bm25 batit sur {self.n} chunks, index en porte {len(self.chunks)}")

        # Perimetre re-DERIVE des CHAMPS, jamais d'un id code en dur (invariant D4).
        # Ici c'est une assertion et non un filtre : le filtrage a eu lieu a l'indexation,
        # l'asserter prouve qu'il a tenu. Un filtre muet cacherait une regression d'index.
        mauvais = {c["utilite_conseil"] for c in self.chunks} - {"directe", "indirecte"}
        if mauvais:
            raise SystemExit(f"echec  l'index contient des chunks {sorted(mauvais)} : "
                             "le perimetre des 33 documents a bouge")
        self.docs = sorted({c["doc"] for c in self.chunks})

    # ---------------------------------------------------------------- BM25
    def bm25(self, requete: str, k: int) -> list[tuple[int, float]]:
        """Okapi BM25 sur les postings livres. Les parametres viennent du manifeste, jamais
        d'une constante recopiee ici : tout nombre ecrit se re-derive de sa source."""
        scores: dict[int, float] = {}
        for terme in set(jetons(requete)):
            p = self.postings.get(terme)
            if not p:
                continue
            df = len(p)
            idf = math.log(1.0 + (self.n - df + 0.5) / (df + 0.5))
            for i, tf in p:
                dl = self.doclen[i]
                num = tf * (self.k1 + 1.0)
                den = tf + self.k1 * (1.0 - self.b + self.b * dl / self.avgdl)
                scores[i] = scores.get(i, 0.0) + idf * num / den
        return sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]

    # ---------------------------------------------------------------- idf / appartenance
    def idf(self, terme: str) -> float | None:
        """`None` si le terme est ABSENT de l'index — a ne pas confondre avec un idf eleve.
        Meme formule que `bm25()` ci-dessus, et c'est volontaire : deux formules d'idf
        divergeraient EN SILENCE (le classement resterait plausible, la porte serait fausse)."""
        p = self.postings.get(terme)
        if not p:
            return None
        df = len(p)
        return math.log(1.0 + (self.n - df + 0.5) / (df + 0.5))

    def idf_max_theorique(self) -> float:
        """idf que vaudrait un terme de df=0 : borne SUPERIEURE jamais atteinte par un terme
        indexe. Sert de poids a un jeton de la requete absent du corpus (« nginx », « Tokyo ») :
        un jeton absent est maximalement rare, le compter a 0 le rendrait gratuit."""
        return math.log(1.0 + (self.n + 0.5) / 0.5)

    def contient(self, terme: str, i: int) -> bool:
        """Le chunk `i` porte-t-il ce terme ? Lu dans les POSTINGS LIVRES, pas en re-jetonnant
        le texte : c'est l'artefact soumis qui fait foi, et la liste est triee par `i` donc la
        recherche est dichotomique. `[i]` se compare avant `[i, tf]` (prefixe), d'ou bisect_left."""
        p = self.postings.get(terme)
        if not p:
            return False
        k = bisect.bisect_left(p, [i])
        return k < len(p) and p[k][0] == i

    # ---------------------------------------------------------------- dense
    def dense(self, q: np.ndarray, k: int) -> list[tuple[int, float]]:
        """cos = produit scalaire : les deux cotes sont normalises a la source."""
        cos = self.vecteurs @ q.astype(np.float32)
        ordre = np.argpartition(-cos, min(k, len(cos) - 1))[:k]
        ordre = ordre[np.argsort(-cos[ordre])]
        return [(int(i), float(cos[i])) for i in ordre]

    def cos_tous(self, q: np.ndarray) -> np.ndarray:
        return self.vecteurs @ q.astype(np.float32)


# =========================================================================================
# fusion, diversite, dedoublonnage
# =========================================================================================
def _rangs(liste: list[tuple[int, float]]) -> dict[int, int]:
    return {i: r for r, (i, _s) in enumerate(liste, start=1)}


def _chevauchent(a: dict, b: dict) -> bool:
    """Deux chunks se recouvrent si leurs intervalles d'offsets s'intersectent DANS le meme
    document. Exact par construction : un chunk est une tranche contigue du .txt aux offsets
    declares (invariant D4), et le chevauchement d'indexation vaut 1 phrase."""
    if a["doc"] != b["doc"]:
        return False
    (a0, a1), (b0, b1) = a["off"], b["off"]
    return a0 < b1 and b0 < a1


def fusionne(idx: Index, l_bm25: list[tuple[int, float]], l_dense: list[tuple[int, float]],
             k_final: int = K_FINAL, max_par_doc: int = MAX_PAR_DOC,
             penalite_structure: int = PENALITE_STRUCTURE) -> tuple[list[dict], list[dict]]:
    """RRF + demotion structurelle + plafond par document + dedoublonnage par offsets.

    Rend (retenus, ecartes) — les ecartes portent leur MOTIF, parce qu'un chunk juste
    elimine par le plafond est une information de calibration, pas un dechet : une metrique
    negative peut accuser l'etiquette plutot que le classement, et sans le motif de rejet l'ambiguite
    entre les deux reste indecidable."""
    r_bm25, r_dense = _rangs(l_bm25), _rangs(l_dense)
    s_bm25 = dict(l_bm25)
    s_dense = dict(l_dense)

    brut: list[dict] = []
    for i in set(r_bm25) | set(r_dense):
        c = idx.chunks[i]
        pen = penalite_structure if c["structure"] in STRUCTURE_DEMOTEE else 0
        score = 0.0
        for rangs in (r_bm25, r_dense):
            if i in rangs:
                score += 1.0 / (RRF_K + rangs[i] + pen)
        brut.append({
            "i": i, "rrf": score,
            "rang_bm25": r_bm25.get(i), "rang_dense": r_dense.get(i),
            "score_bm25": s_bm25.get(i), "cos": s_dense.get(i),
            "penalite": pen,
        })
    brut.sort(key=lambda x: (-x["rrf"], x["i"]))

    retenus: list[dict] = []
    ecartes: list[dict] = []
    par_doc: dict[str, int] = {}
    for cand in brut:
        c = idx.chunks[cand["i"]]
        double = next((r for r in retenus if _chevauchent(c, idx.chunks[r["i"]])), None)
        if double is not None:
            ecartes.append(cand | {"motif": f"chevauche i={double['i']} (offsets)"})
            continue
        if par_doc.get(c["doc"], 0) >= max_par_doc:
            ecartes.append(cand | {"motif": f"plafond {max_par_doc}/document sur {c['doc']}"})
            continue
        par_doc[c["doc"]] = par_doc.get(c["doc"], 0) + 1
        retenus.append(cand)
        if len(retenus) >= k_final:
            break
    return retenus, ecartes


def voisins_meme_page(idx: Index, i: int) -> list[int]:
    """Chunks adjacents du MEME document et de la MEME page, pour recoudre une citation
    tronquee. Le D4 a mesure que l'ancrage culture<->dose se perd dans les petits chunks et
    a declare cette recouture comme la recuperation prevue. Jamais a travers une page : un
    chunk a cheval n'aurait aucun numero de page unique a citer."""
    c = idx.chunks[i]
    return [j for j in (i - 1, i + 1)
            if 0 <= j < len(idx.chunks)
            and idx.chunks[j]["doc"] == c["doc"]
            and idx.chunks[j]["page"] == c["page"]]


# =========================================================================================
# statistiques de porte — SEPT candidats, aucun choisi
# =========================================================================================
def couverture_idf(idx: Index, requete: str, i_retenus: list[int]) -> dict:
    """Candidats de porte 8 a 11 : la requete a-t-elle ses jetons RARES dans ce qu'on va citer ?

    Pourquoi cette famille, et pourquoi apres la mesure du 19/08 : les 7 premiers candidats
    CHEVAUCHENT tous entre niveau 1 et niveau 2 — le cosinus mesure la proximite THEMATIQUE,
    et nos absences les plus interessantes sont thematiquement tres proches du corpus (le
    « cumul de pluie avant de semer le mil » sort a 0,7018, au-dessus de 8 items niveau 1 sur
    11). Le constat 3 du brief D5 nommait deja le vrai discriminant : sur le catalogue, tous
    les chunks partagent le meme gabarit (« Choix du sol / Fertilisation / Date de semis ») et
    **seul le nom de l'espece** — jeton rare — separe une culture d'une autre. C'est du ressort
    du lexique, pas du plongement.

    🔴 Ce que ces 4 nombres mesurent, et ce qu'ils NE mesurent pas : la presence LEXICALE du
    vocabulaire rare de la demande dans le passage. Un chunk peut nommer le mil et Maradi sans
    porter la dose demandee. C'est un proxy de l'ancrage, jamais une preuve que le fait est la
    — cette preuve-la est le travail de verification d'ancre de `answer.py`, pas d'un scalaire.

    Rendu sans serveur d'embedding (jetons + postings + ids), donc calculable a posteriori sur
    un rapport de mesure deja paye : c'est ce qui evite de relancer 47 requetes."""
    js = sorted(set(jetons(requete)))
    idfs = {j: idx.idf(j) for j in js}
    presents = [j for j in js if idfs[j] is not None]
    idf_max = idx.idf_max_theorique()

    # 8. vocabulaire rare de la requete present DANS LE CORPUS. Un jeton absent est pondere a
    #    `idf_max` : c'est la seule ponderation qui ne recompense pas l'absence.
    poids_tot = sum(idfs[j] if idfs[j] is not None else idf_max for j in js)
    idf_req = (sum(idfs[j] for j in presents) / poids_tot) if poids_tot else 0.0

    def couvre(i: int) -> tuple[float, float]:
        if not presents:
            return 0.0, 0.0
        tot = sum(idfs[j] for j in presents)
        dedans = [j for j in presents if idx.contient(j, i)]
        rare_present = max(idfs[j] for j in presents)
        rare_couvert = max((idfs[j] for j in dedans), default=0.0)
        return (sum(idfs[j] for j in dedans) / tot if tot else 0.0,
                rare_couvert / rare_present if rare_present else 0.0)

    paires = [couvre(i) for i in i_retenus]
    return {
        "idf_couvert_requete": idf_req,                                   # 8
        "idf_couvert_retenu_1": paires[0][0] if paires else 0.0,          # 9
        "idf_couvert_max_retenus": max((a for a, _b in paires), default=0.0),   # 10
        "idf_rare_dans_retenu_1": paires[0][1] if paires else 0.0,        # 11
        "_n_jetons": len(js), "_n_jetons_dans_corpus": len(presents),
    }


def statistiques_porte(idx: Index, requete: str, retenus: list[dict],
                       l_dense: list[tuple[int, float]],
                       l_bm25: list[tuple[int, float]]) -> dict:
    """Emet les candidats de porte cote a cote, sans en elire aucun.

    Le candidat evident — le cosinus du top-1 dense — est MESURE quasi aveugle : les bons
    chunks sortent a 0,6143-0,6567 et les meilleurs mauvais a 0,6570-0,7425, distributions
    completement chevauchantes. Les six autres existent parce qu'un seuil doit separer, et
    que rien ne dit d'avance quelle grandeur separe. La calibration a tranche par la mesure :
    `cos_du_retenu_1` (auc 0,972 sur la frontiere 2/3), et AUCUNE d'entre elles sur la
    frontiere 1/2 — voir `rag/index/seuils.json`.

    🔴 Mesure du 19/08, apres fusion et sur 47 requetes : les SEPT premiers candidats
    chevauchent entre niveau 1 et niveau 2. D'ou les candidats 8 a 11 (`couverture_idf`),
    lexicaux et non thematiques. Aucun n'est elu ici — ce fichier n'ecrit toujours aucun seuil.
    """
    cos_retenus = [r["cos"] for r in retenus if r["cos"] is not None]
    cos_dense = [c for _i, c in l_dense]
    ens_bm25 = {i for i, _ in l_bm25[:10]}
    ens_dense = {i for i, _ in l_dense[:10]}
    top = retenus[0] if retenus else None
    js = sorted(set(jetons(requete)))
    return {
        # 1. le candidat naif, garde POUR ETRE REFUTE par la mesure du 3
        "cos_top1_dense": cos_dense[0] if cos_dense else None,
        # 2. cosinus du gagnant de la fusion (peut etre None : gagnant venu du BM25 seul)
        "cos_du_retenu_1": top["cos"] if top else None,
        # 3. le meilleur cosinus parmi les chunks effectivement rendus
        "cos_max_retenus": max(cos_retenus) if cos_retenus else None,
        # 4. ecart entre les deux premiers du dense : une requete hors corpus donne un
        #    plateau plat, une requete sourcable un decrochage. A mesurer.
        "ecart_dense_1_2": (cos_dense[0] - cos_dense[1]) if len(cos_dense) > 1 else None,
        # 5. accord des deux moteurs sur leur top-10 : 0 quand chacun part ailleurs, ce qui
        #    est le symptome attendu d'une requete que le corpus ne couvre pas.
        "accord_bm25_dense_top10": len(ens_bm25 & ens_dense) / 10.0,
        # 6. le BM25 du premier : un toponyme ou une espece rare absente du corpus fait
        #    tomber cette valeur, la ou le dense reste bavard (il repond toujours quelque chose).
        "bm25_top1": l_bm25[0][1] if l_bm25 else 0.0,
        # 7. part des jetons de la requete presents dans l'index — mesure lexicale directe,
        #    NON PONDEREE : un « nginx » absent y pese autant qu'un « quelle » banal present.
        #    C'est ce defaut que le candidat 8 corrige en ponderant par l'idf.
        "jetons_couverts": (sum(1 for j in js if j in idx.postings) / len(js)) if js else 0.0,
        # 8 a 11 : couverture ponderee par l'idf, requete puis chunk cite (`couverture_idf`)
        **couverture_idf(idx, requete, [r["i"] for r in retenus]),
        "n_retenus": len(retenus),
        "docs_distincts": len({idx.chunks[r["i"]]["doc"] for r in retenus}),
    }


# =========================================================================================
# facade
# =========================================================================================
class Recuperateur:
    """Ouvre le serveur d'embedding une fois pour plusieurs requetes.

        with Recuperateur() as rec:
            res = rec.cherche("Quand semer le mil a Maradi ?")

    `embed_server.Serveur` ouvre ET ferme (aucun serveur ne survit : `verify_index.py` 9 le
    prouve, contre-epreuve incluse)."""

    def __init__(self, dossier: pathlib.Path = INDEX, port: int = 8646,
                 verbeux: bool = False) -> None:
        self.idx = Index(dossier)
        self._srv_args = {"port": port, "verbeux": verbeux}
        self._srv: embed_server.Serveur | None = None
        self.n_appels_embedding = 0

    def __enter__(self) -> "Recuperateur":
        self._srv = embed_server.Serveur(**self._srv_args).__enter__()
        return self

    def __exit__(self, *exc) -> None:
        srv, self._srv = self._srv, None
        if srv is not None:
            srv.__exit__(*exc)

    def plonge_requete(self, requete: str) -> np.ndarray:
        """LOT DE 1, comme au runtime — contrainte 1 heritee du D4, et AUCUN prefixe (M2)."""
        if self._srv is None:
            raise RuntimeError("Recuperateur doit etre utilise en gestionnaire de contexte "
                               "(with), sinon aucun serveur d'embedding n'est ouvert")
        self.n_appels_embedding += 1
        return self._srv.plonge([requete], lot=1)[0]

    def cherche(self, requete: str, k_final: int = K_FINAL, k_pool: int = K_POOL,
                max_par_doc: int = MAX_PAR_DOC,
                penalite_structure: int = PENALITE_STRUCTURE,
                vecteur: np.ndarray | None = None) -> dict:
        """Rend un dict TRACABLE : les listes des deux moteurs, les retenus, les ecartes avec
        leur motif, et les candidats de porte. Aucun niveau n'est decide ici."""
        q = self.plonge_requete(requete) if vecteur is None else vecteur
        l_bm25 = self.idx.bm25(requete, k_pool)
        l_dense = self.idx.dense(q, k_pool)
        retenus, ecartes = fusionne(self.idx, l_bm25, l_dense, k_final=k_final,
                                   max_par_doc=max_par_doc,
                                   penalite_structure=penalite_structure)
        porte = statistiques_porte(self.idx, requete, retenus, l_dense, l_bm25)
        js = jetons(requete)

        for r in retenus:
            c = self.idx.chunks[r["i"]]
            r["provenance"] = {k: c[k] for k in
                               ("doc", "titre", "editeur", "annee", "page", "langue",
                                "licence", "regime", "citation_verbatim_autorisee",
                                "sujet", "structure", "utilite_conseil", "off")}
            r["texte"] = c["texte"]
            r["voisins_meme_page"] = voisins_meme_page(self.idx, r["i"])
        return {
            "requete": requete,
            "jetons": js,
            "parametres": {"rrf_k": RRF_K, "k_pool": k_pool, "k_final": k_final,
                           "max_par_doc": max_par_doc,
                           "penalite_structure": penalite_structure},
            "retenus": retenus,
            "ecartes": ecartes,
            "porte": porte,
            "bm25_top": l_bm25[:10],
            "dense_top": l_dense[:10],
        }


# =========================================================================================
# sonde manuelle — `py rag/retrieve.py "ma question"`
# =========================================================================================
def _apercu(t: str, n: int = 78) -> str:
    return " ".join(t.split())[:n]


def principal() -> int:
    # La console Windows est en cp1252 : un U+2212 (vrai signe MOINS, present dans le corpus)
    # y leve UnicodeEncodeError et TUE la sonde EN PLEIN MILIEU de sa sortie — mesure a moitie
    # affichee, donc mesure trompeuse. Correction a la couche d'AFFICHAGE seule : les chunks ne
    # sont pas touches : devant une sortie fausse, remonter d'une couche avant de rustiner
    # celle qu'on tient.
    for flux in (sys.stdout, sys.stderr):
        if hasattr(flux, "reconfigure"):
            flux.reconfigure(encoding="utf-8", errors="replace")

    reqs = sys.argv[1:] or [
        # ⚠️ Requetes de SONDE, formulees a la main ici — ce ne sont PAS les requetes du jeu
        # etiquete de calibration (non redistribue, meme regime que corpus/txt/fetched/).
        # Mesure du 19/08 : la 1re ecrit « semer » quand la source ecrit « semis », ce qui
        # fait passer le bon chunk du rang dense 5 (requete etiquetee) au rang 29 (celle-ci).
        # Ne jamais lire un rang de cette sonde comme un rang du systeme.
        "A quelle periode faut-il semer le mil dans la region de Maradi au Niger ?",
        "Quelle est la capitale du Japon ?",
    ]
    with Recuperateur(verbeux=False) as rec:
        print(f"ok     index {len(rec.idx.chunks)} chunks / {len(rec.idx.docs)} documents / "
              f"{len(rec.idx.postings)} termes bm25 / avgdl {rec.idx.avgdl:.2f}")
        for r in reqs:
            res = rec.cherche(r)
            print("=" * 100)
            print(f"requete  {r}")
            p = res["porte"]
            print(f"porte    cos_top1_dense {p['cos_top1_dense']:.4f} | "
                  f"cos_max_retenus {p['cos_max_retenus'] if p['cos_max_retenus'] is None else round(p['cos_max_retenus'], 4)} | "
                  f"ecart_1_2 {p['ecart_dense_1_2']:.4f} | accord {p['accord_bm25_dense_top10']:.2f} | "
                  f"bm25_top1 {p['bm25_top1']:.2f} | jetons_couverts {p['jetons_couverts']:.2f}")
            for rang, x in enumerate(res["retenus"], 1):
                pr = x["provenance"]
                cos = "  --  " if x["cos"] is None else f"{x['cos']:.4f}"
                print(f" {rang}. rrf {x['rrf']:.5f} cos {cos} bm25#{x['rang_bm25']} "
                      f"dense#{x['rang_dense']} i={x['i']:<5} {pr['doc'][:34]:34} p{pr['page']:<4} "
                      f"{_apercu(x['texte'], 60)}")
            if res["ecartes"]:
                print(f"    ecartes ({len(res['ecartes'])}) : "
                      + " | ".join(f"i={e['i']} {e['motif']}" for e in res["ecartes"][:4]))
    return 0


if __name__ == "__main__":
    raise SystemExit(principal())
