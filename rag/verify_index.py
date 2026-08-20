"""Verification COMPORTEMENTALE de rag/index/ — chaque controle EXECUTE l'acte.

Modele : la verification du corpus texte au D3 (meme principe). Le principe qui gouverne ce
fichier est une lecon
payee trois fois sur ce projet — `jq` jamais execute, `grep -c $'\\r'` a motif vide qui
renvoie 89 sur un fichier LF pur, `git check-ignore` qui cite une ligne VIDE — et une
quatrieme fois pendant le D4 : `tasklist | grep llama` n'a rien trouve pendant que
`Get-Process` rendait le PID 23944. Une commande de verification dont l'echec ressemble a un
resultat plausible est pire qu'aucune verification.

D'ou la regle appliquee ici : tout controle qui pourrait etre vert par accident porte sa
CONTRE-EPREUVE — on fabrique la faute que le controle doit attraper, et on exige qu'il
l'attrape. Un test qui ne peut pas echouer ne prouve rien, et une defense jamais vue
refuser ne prouve rien non plus.

Sections :
  0. perimetre DERIVE de sources.yaml puis verrouille a 33 ; documents.json croise
  1. attribution de page prouvee VERBATIM sur les octets du .txt      + contre-epreuve
  2. aucun marqueur [[page N]] dans le texte indexe                   + contre-epreuve
  3. provenance CROISEE contre sources.yaml champ par champ (pas « non vide »)
  4. couverture du corps : aucune perte silencieuse de texte
  5. BM25 REBATI depuis chunks.jsonl et compare octet a octet         + contre-epreuve
  6. 0 octet 0x0D compte en BINAIRE, UTF-8 strict, sur chaque fichier produit
  7. vecteurs : forme, normes, et un ECHANTILLON RE-EMBARQUE serveur neuf sans cache
  8. sanite translingue sur CHUNKS REELS, marge IMPRIMEE
  9. aucun llama-server.exe ne survit au run                          + contre-epreuve

Usage : py rag/verify_index.py [--rapide] [--echantillon N]
  --rapide  saute les sections 7 et 8 (les seules qui lancent un serveur).
Sortie : exit 0 si 0 echec, exit 1 sinon. Les « notes » n'echouent pas mais s'impriment.
"""
from __future__ import annotations

import argparse
import io
import json
import pathlib
import random
import re
import subprocess
import sys

import numpy as np

RACINE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE / "rag"))

import embed_server                     # noqa: E402
import index as idx                     # noqa: E402
from embed_server import Serveur        # noqa: E402

INDEX = RACINE / "rag" / "index"
FICHIERS_TEXTE = ["chunks.jsonl", "documents.json", "bm25.json", "manifest.json"]

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
    return condition


def titre(n: int, texte: str) -> None:
    print(f"\n--- {n}. {texte}")


# =========================================================================================
def charge() -> tuple[list[dict], list[dict], dict, dict]:
    chunks = [json.loads(l) for l in io.open(INDEX / "chunks.jsonl", encoding="utf-8")]
    docs = json.load(io.open(INDEX / "documents.json", encoding="utf-8"))
    bm25 = json.load(io.open(INDEX / "bm25.json", encoding="utf-8"))
    manif = json.load(io.open(INDEX / "manifest.json", encoding="utf-8"))
    return chunks, docs, bm25, manif


# =========================================================================================
def section_0(chunks, docs, manif) -> dict:
    titre(0, "perimetre : DERIVE de sources.yaml, puis verrouille")
    sources = idx.charge_sources()
    retenus, comptes = idx.documents_indexables(sources)      # leve SystemExit si ecart
    ok(f"{comptes['n_sources']} sources - {comptes['exclus_utilite_methodologique']} "
       f"methodologique - {comptes['exclus_champ_exclusion']} exclusion = "
       f"{comptes['n_documents']} (verrou {idx.N_DOCUMENTS_ATTENDU})")

    exige(len(docs) == idx.N_DOCUMENTS_ATTENDU,
          f"documents.json porte {len(docs)} documents, {idx.N_DOCUMENTS_ATTENDU} attendus")
    ids_src = {s["id"] for s in retenus}
    ids_doc = {d["id"] for d in docs}
    ids_chunk = {c["doc"] for c in chunks}
    exige(ids_doc == ids_src, "les ids de documents.json sont EXACTEMENT ceux derives")
    exige(ids_chunk == ids_src,
          f"les {len(ids_chunk)} ids porteurs de chunks sont exactement ceux derives"
          + ("" if ids_chunk == ids_src else f" (absents : {sorted(ids_src - ids_chunk)})"))

    # L'exclusion doit rester une DECISION visible, pas un effet de bord (decision du D3 :
    # une source ecartee garde son entree dans sources.yaml, avec son motif mesure).
    exclus = [s for s in sources if s.get("exclusion")]
    exige(len(exclus) == idx.N_EXCLUSION_ATTENDU
          and all(len(str(s.get("exclusion_motif", ""))) >= 80 for s in exclus),
          f"la {len(exclus)} exclusion porte un motif >= 80 car. et n'est dans aucun chunk")
    exige(not (ids_chunk & {s["id"] for s in exclus}),
          "aucun chunk ne vient d'un document exclu")

    # Le manifeste ne doit pas raconter autre chose que le livrable.
    exige(manif["chunks"]["n"] == len(chunks),
          f"manifest.chunks.n == {len(chunks)} chunks reels")
    exige(manif["seuils_cosinus"].startswith("AUCUN"),
          "aucun seuil cosinus n'est code dans l'index (calibration D5)")
    return {"retenus": retenus, "sources": {s["id"]: s for s in sources}}


# =========================================================================================
def section_1(chunks, docs, echantillon: int, alea: random.Random) -> None:
    titre(1, "attribution de page prouvee VERBATIM sur les octets du .txt")
    textes: dict[str, str] = {}
    for d in docs:
        chemin = RACINE / d["chemin_txt"]
        if not chemin.is_file():
            echec(f"{d['chemin_txt']} absent : l'index ne peut pas etre prouve")
            continue
        t = io.open(chemin, encoding="utf-8", newline="").read()
        textes[d["id"]] = t
        import hashlib
        if hashlib.sha256(t.encode("utf-8")).hexdigest() != d["sha256_txt"]:
            echec(f"{d['id']} : le .txt sur disque n'est plus celui indexe (sha256)")
    if not textes:
        return
    ok(f"{len(textes)} .txt relus et compares par sha256 a documents.json")

    # Bornes de page relues DU .TXT, pas du chunk : c'est ce qui rend la preuve independante.
    pages: dict[str, list[tuple[int, int, int]]] = {}
    for d in docs:
        if d["id"] not in textes:
            continue
        t = textes[d["id"]]
        debut = t.index(idx.SEPARATEUR_CORPS) + len(idx.SEPARATEUR_CORPS)
        pages[d["id"]] = idx.segments_pages(t, debut)

    tires = alea.sample(chunks, min(echantillon, len(chunks)))
    n_verbatim = n_page = 0
    for c in tires:
        t = textes.get(c["doc"])
        if t is None:
            continue
        a, b = c["off"]
        if t[a:b] == c["texte"]:
            n_verbatim += 1
        else:
            echec(f"{c['doc']} chunk {c['i']} : txt[{a}:{b}] != texte indexe")
        seg = [(p, x, y) for p, x, y in pages[c["doc"]] if p == c["page"]]
        if seg and all(seg[0][1] <= a and b <= seg[0][2] for _ in (0,)):
            n_page += 1
        else:
            echec(f"{c['doc']} chunk {c['i']} : offsets hors de la page {c['page']} declaree")
    ok(f"{n_verbatim}/{len(tires)} chunks tires : texte == txt[off0:off1] a l'octet")
    ok(f"{n_page}/{len(tires)} chunks tires : offsets APRES [[page N]] et AVANT le suivant")

    # ---------------- CONTRE-EPREUVE : le controle attrape-t-il une faute fabriquee ?
    c = tires[0]
    t = textes[c["doc"]]
    a, b = c["off"]
    faux_texte = t[a + 1:b + 1]                      # offsets decales de 1 caractere
    pris_1 = faux_texte != c["texte"]
    autres = [p for p, _x, _y in pages[c["doc"]] if p != c["page"]]
    pris_2 = True
    if autres:
        seg = next((x, y) for p, x, y in pages[c["doc"]] if p == autres[0])
        pris_2 = not (seg[0] <= a and b <= seg[1])   # page fausse -> hors bornes
    exige(pris_1 and pris_2,
          "CONTRE-EPREUVE : un decalage de 1 caractere ET une page fausse sont tous deux "
          "attrapes (le controle peut echouer, il prouve donc quelque chose)")


# =========================================================================================
def section_2(chunks) -> None:
    titre(2, "aucun marqueur [[page N]] dans le texte indexe")
    fautifs = [c["i"] for c in chunks if "[[page" in c["texte"]]
    exige(not fautifs, f"0 chunk contient « [[page » ({len(chunks)} inspectes)"
          + ("" if not fautifs else f" — {fautifs[:5]}"))
    # Le BM25 ne doit pas non plus porter le mot « page » venu des marqueurs : on verifie
    # que le jeton n'a pas explose (un marqueur par page = 677 occurrences si indexes).
    fabrique = "avant\n[[page 7]]\napres"
    exige("[[page" in fabrique and "page" in idx.jetons(fabrique),
          "CONTRE-EPREUVE : le detecteur voit bien « [[page » dans un texte fabrique qui en "
          "contient (donc son silence sur l'index est une mesure, pas une panne)")


# =========================================================================================
def section_3(chunks, ctx) -> None:
    titre(3, "provenance CROISEE contre sources.yaml, champ par champ")
    src = ctx["sources"]
    champs = ["titre", "editeur", "annee", "langue", "licence", "regime",
              "sujet", "portee", "utilite_conseil"]

    def norm(x) -> str:
        return "" if x is None else str(x).strip()

    ecarts: dict[str, int] = {}
    vides: dict[str, int] = {}
    for c in chunks:
        s = src.get(c["doc"])
        if s is None:
            echec(f"chunk {c['i']} : doc {c['doc']} absent de sources.yaml")
            continue
        for f in champs:
            if norm(c.get(f)) != norm(s.get(f)):
                ecarts[f] = ecarts.get(f, 0) + 1
            if f != "licence" and not norm(c.get(f)):
                vides[f] = vides.get(f, 0) + 1
    exige(not ecarts, f"les {len(champs)} champs de provenance de {len(chunks)} chunks "
                      f"coincident avec sources.yaml" + ("" if not ecarts else f" — {ecarts}"))
    exige(not vides, "aucun champ de provenance vide (licence exclue : null est une valeur "
                     "legitime, 26 documents sur 33)" + ("" if not vides else f" — {vides}"))

    # La citation verbatim n'est autorisee que la ou la licence a ete CONFIRMEE DANS LE PDF.
    incoherents = [c["i"] for c in chunks
                   if (norm(c["citation_verbatim_autorisee"]).lower() in ("true", "oui"))
                   != bool(src[c["doc"]].get("licence_confirmee_dans_le_pdf"))]
    exige(not incoherents,
          "citation_verbatim_autorisee suit licence_confirmee_dans_le_pdf sur tous les chunks")
    n_cc = len({c["doc"] for c in chunks
                if norm(c["citation_verbatim_autorisee"]).lower() in ("true", "oui")})
    exige(n_cc == 7,
          f"regime declare : {n_cc} documents a citation verbatim autorisee (licence CC-BY "
          f"confirmee DANS le PDF), {len({c['doc'] for c in chunks}) - n_cc} a licence null "
          "— c'est le regime que REPORT.md doit declarer en clair")


# =========================================================================================
def section_4(chunks, docs) -> None:
    titre(4, "couverture du corps : aucune perte silencieuse de texte")
    par_doc: dict[str, list[dict]] = {}
    for c in chunks:
        par_doc.setdefault(c["doc"], []).append(c)
    pire = (1.0, "")
    total_corps = total_couvert = 0
    for d in docs:
        chemin = RACINE / d["chemin_txt"]
        if not chemin.is_file():
            continue
        t = io.open(chemin, encoding="utf-8", newline="").read()
        debut = t.index(idx.SEPARATEUR_CORPS) + len(idx.SEPARATEUR_CORPS)
        segs = idx.segments_pages(t, debut)
        # Corps utile = pages retenues, marqueurs exclus (segments_pages les exclut deja).
        utile = sum(len(t[a:b].strip()) for _p, a, b in segs
                    if len(t[a:b].strip()) >= idx.PLANCHER_PAGE)
        # Couverture = union des intervalles (le chevauchement ne doit pas gonfler le compte).
        iv = sorted((c["off"][0], c["off"][1]) for c in par_doc.get(d["id"], []))
        fus: list[list[int]] = []
        for a, b in iv:
            if fus and a <= fus[-1][1]:
                fus[-1][1] = max(fus[-1][1], b)
            else:
                fus.append([a, b])
        couvert = sum(len(t[a:b].strip()) for a, b in fus)
        total_corps += utile
        total_couvert += couvert
        r = couvert / max(utile, 1)
        if r < pire[0]:
            pire = (r, d["id"])
    r = total_couvert / max(total_corps, 1)
    exige(r >= 0.995, f"couverture du corps utile : {100 * r:.3f} % "
                      f"({total_couvert:,} / {total_corps:,} car.)")
    if pire[0] < 0.99:
        note(f"document le moins couvert : {pire[1]} a {100 * pire[0]:.2f} %")
    else:
        ok(f"document le moins couvert : {pire[1]} a {100 * pire[0]:.2f} %")


# =========================================================================================
def section_5(chunks, bm25) -> None:
    titre(5, "BM25 REBATI depuis chunks.jsonl et compare au fichier livre")
    refait = idx.index_bm25(chunks)
    exige(refait["postings"] == bm25["postings"],
          f"postings identiques ({len(bm25['postings']):,} termes) — le fichier livre EST "
          "reproductible depuis chunks.jsonl")
    exige(abs(refait["avgdl"] - bm25["avgdl"]) < 1e-9,
          f"avgdl identique ({bm25['avgdl']:.4f})")
    exige(refait["doclen"] == bm25["doclen"], "longueurs de documents identiques")

    # ---------------- CONTRE-EPREUVE : la garde de liste d'arret peut-elle crier ?
    sauve = set(idx.LISTE_ARRET)
    try:
        idx.LISTE_ARRET.add("mais")            # le repli d'accents fusionne mais/mais
        crie = False
        try:
            idx.garde_liste_arret()
        except SystemExit:
            crie = True
        exige(crie, "CONTRE-EPREUVE : injecter « mais » dans la liste d'arret fait crier la "
                    "garde (elle protege donc reellement les 5 cultures du BM25)")
    finally:
        idx.LISTE_ARRET.clear()
        idx.LISTE_ARRET.update(sauve)
    idx.garde_liste_arret()
    ok("liste d'arret restauree, garde repassee verte")

    # Les composes a tiret doivent rester retrouvables ENTIERS : c'est ce que le test_prompt
    # n 1 doit citer (« septembre-octobre ») et ce que la decesure du D3 a paye pour garder.
    for compose, morceau in (("15-15-15", "15"), ("septembre-octobre", "septembre")):
        js = idx.jetons(f"apport de {compose} au semis")
        exige(compose in js and morceau in js,
              f"« {compose} » est indexe ENTIER et en morceaux (« {morceau} »)")


# =========================================================================================
def section_6() -> None:
    titre(6, "0 octet 0x0D compte en BINAIRE, UTF-8 strict")
    total = 0
    for nom in FICHIERS_TEXTE:
        chemin = INDEX / nom
        brut = chemin.read_bytes()
        cr = brut.count(b"\x0d")
        total += len(brut)
        if cr:
            echec(f"{nom} : {cr} octets 0x0D sur {len(brut):,}")
        try:
            brut.decode("utf-8", errors="strict")
        except UnicodeDecodeError as e:
            echec(f"{nom} : UTF-8 invalide ({e})")
    ok(f"{len(FICHIERS_TEXTE)} fichiers texte : 0 octet 0x0D sur {total:,}, UTF-8 strict")
    # CONTRE-EPREUVE : le compteur d'octets sait-il compter un CR ? Une commande dont l'echec
    # ressemble a un resultat plausible ne mesure rien :
    # `grep -c $'\\r'` renvoyait 89 sur un fichier LF pur, son motif etant vide.
    exige(b"a\r\nb".count(b"\x0d") == 1,
          "CONTRE-EPREUVE : le compteur voit 1 octet 0x0D dans b'a\\r\\nb'")
    npy = (INDEX / "vectors.npy").read_bytes()
    ok(f"vectors.npy : {len(npy):,} octets, binaire — exclu du controle CR a dessein")


# =========================================================================================
def section_7(chunks, manif, echantillon: int, alea: random.Random) -> None:
    titre(7, "vecteurs : forme, normes, et echantillon RE-EMBARQUE serveur neuf sans cache")
    V = np.load(INDEX / "vectors.npy")
    dtype = manif["embeddings"]["dtype_stocke"]
    exige(V.shape == (len(chunks), embed_server.DIM),
          f"forme {V.shape} == ({len(chunks)}, {embed_server.DIM})")
    exige(str(V.dtype) == dtype, f"dtype stocke {V.dtype} == manifeste ({dtype})")
    normes = np.linalg.norm(V.astype(np.float64), axis=1)
    ecart = float(np.max(np.abs(normes - 1.0)))
    # Tolerance liee au FORMAT, pas choisie a la main : float16 a ~3 chiffres significatifs.
    tol = {"float32": 1e-3, "float16": 5e-3, "int8": 2e-2}[dtype]
    exige(ecart <= tol, f"normes a {ecart:.2e} de 1,0 (tolerance {tol:.0e} pour {dtype})")

    tires = alea.sample(range(len(chunks)), min(echantillon, len(chunks)))
    with Serveur(port=8671, verbeux=False) as srv:
        R = srv.plonge([chunks[i]["texte"] for i in tires], lot=8)
        pid_vivant = pids_llama_server()
    A = V[tires].astype(np.float64)
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    cos = np.einsum("ij,ij->i", A, R.astype(np.float64))
    exige(float(cos.min()) >= 0.9995,
          f"{len(tires)} chunks re-embarques (serveur NEUF, cache contourne) : cosinus min "
          f"{cos.min():.6f} avec la ligne correspondante de vectors.npy")
    return pid_vivant


# =========================================================================================
def section_8(chunks) -> None:
    titre(8, "sanite translingue sur CHUNKS REELS — marge IMPRIMEE")
    metadonnees = json.load(io.open(RACINE / "metadata.json", encoding="utf-8"))
    # Cle LUE dans metadata.json, pas devinee : c'est `prompt_id`, pas `id` — deviner un nom de
    # cle JSON a deja produit de fausses cles ici. Et le prompt est pris MOT POUR MOT du fichier
    # soumis — mesurer sur une paraphrase ne mesurerait pas ce que les juges enverront.
    prompts = {p["prompt_id"]: p["prompt"] for p in metadonnees["test_prompts"]}
    en = prompts["striga_sorgho_cross_lingual"]
    with Serveur(port=8672, verbeux=False) as srv:
        q = srv.plonge([en], lot=1)[0]
    V = np.load(INDEX / "vectors.npy").astype(np.float64)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    cos = V @ q.astype(np.float64)
    pert = [i for i, c in enumerate(chunks) if "striga" in idx.replie(c["texte"])]
    hors = [i for i in range(len(chunks)) if i not in set(pert)]
    if not pert:
        echec("aucun chunk ne mentionne striga : la prémisse du test_prompt n 2 est fausse")
        return
    ordre = np.argsort(-cos)
    rang = next(r for r, i in enumerate(ordre, 1) if i in set(pert))
    marge = float(cos[pert].max() - cos[hors].max())
    med = float(np.median(cos[pert]) - np.median(cos[hors]))
    print(f"        requete EN -> corpus 75 % FR | {len(pert)} chunks pertinents lexicalement")
    print(f"        cos max pertinent {cos[pert].max():.4f} | max hors-sujet "
          f"{cos[hors].max():.4f}")
    print(f"        MARGE DURE {marge:+.4f} | marge des medianes {med:+.4f} | rang du "
          f"premier pertinent {rang}")
    for i in ordre[:3]:
        print(f"          {cos[i]:.4f} [{chunks[i]['langue']}] {chunks[i]['doc'][:28]:28} "
              f"p{chunks[i]['page']:<4} {chunks[i]['texte'][:64]!r}")
    exige(rang == 1, f"le premier resultat de la requete ANGLAISE est un chunk pertinent "
                     f"(rang {rang})")
    trois = [chunks[i]["langue"] for i in ordre[:3]]
    exige(trois.count("fr") >= 2,
          f"au moins 2 des 3 meilleurs sont en FRANCAIS pour une requete ANGLAISE : {trois} "
          "— c'est le differenciateur translingue, mesure et non suppose")
    if marge < 0.03:
        note(f"marge dure {marge:+.4f} : le meilleur distracteur est proche. La reference "
             "+0,3346 mesuree au D4 sur UNE paire de phrases fabriquee n'est PAS comparable "
             "(un maximum sur 3 179 distracteurs reels, tous agronomiques, n'a pas la meme "
             "loi) — c'est la marge des medianes qui l'est. Surveille au D5 lors de la "
             "calibration de la porte : rag/index/seuils.json.")


# =========================================================================================
def pids_llama_server() -> list[int]:
    """PowerShell, PAS `tasklist` : mesure du 17/08 — `tasklist /FO CSV /NH | grep -i llama`
    n'a rien rendu pendant que Get-Process rendait le PID 23944. Un controle bati sur
    tasklist serait vert a jamais en ne mesurant rien."""
    r = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-Process -Name llama-server -ErrorAction SilentlyContinue | "
         "Select-Object -ExpandProperty Id"],
        capture_output=True, text=True, timeout=60)
    return [int(x) for x in r.stdout.split() if x.strip().isdigit()]


def section_9(pid_vivant) -> None:
    titre(9, "aucun llama-server.exe ne survit au run")
    # CONTRE-EPREUVE d'abord : la commande sait-elle TROUVER un serveur ? Sans cela, « 0
    # processus » est indistinguable d'un detecteur debranche : une defense
    # jamais vue refuser ne prouve rien.
    if pid_vivant:
        ok(f"CONTRE-EPREUVE : pendant la section 7, Get-Process trouvait bien "
           f"{len(pid_vivant)} llama-server (PID {pid_vivant}) — le detecteur voit")
    else:
        with Serveur(port=8673, verbeux=False) as srv:      # noqa: F841
            pendant = pids_llama_server()
        exige(bool(pendant), "CONTRE-EPREUVE : Get-Process trouve un llama-server pendant "
                             "qu'un serveur tourne")
    restants = pids_llama_server()
    exige(not restants,
          f"0 llama-server.exe survivant apres tous les arrets" +
          ("" if not restants else f" — PID {restants} A TUER"))


# =========================================================================================
def principal() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--echantillon", type=int, default=120)
    ap.add_argument("--rapide", action="store_true",
                    help="saute les sections 7 a 9 (les seules qui lancent un serveur)")
    a = ap.parse_args()
    alea = random.Random(42)                     # tirage reproductible

    print(f"VERIFICATION de {INDEX}")
    manquants = [f for f in FICHIERS_TEXTE + ["vectors.npy"] if not (INDEX / f).is_file()]
    if manquants:
        print(f"  ECHEC fichiers absents : {manquants}")
        return 1
    chunks, docs, bm25, manif = charge()

    ctx = section_0(chunks, docs, manif)
    section_1(chunks, docs, a.echantillon, alea)
    section_2(chunks)
    section_3(chunks, ctx)
    section_4(chunks, docs)
    section_5(chunks, bm25)
    section_6()
    pid_vivant: list[int] = []
    if a.rapide:
        note("sections 7 a 9 SAUTEES (--rapide) : les vecteurs ne sont pas prouves "
             "correspondre aux chunks, et rien ne prouve qu'aucun serveur ne survit")
    else:
        pid_vivant = section_7(chunks, manif, min(a.echantillon, 24), alea) or []
        section_8(chunks)
        section_9(pid_vivant)

    print(f"\n{'=' * 86}")
    print(f"{len(ECHECS)} echec(s), {len(NOTES)} note(s)")
    for e in ECHECS:
        print(f"  ECHEC {e}")
    for n in NOTES:
        print(f"  note  {n}")
    return 1 if ECHECS else 0


if __name__ == "__main__":
    sys.exit(principal())
