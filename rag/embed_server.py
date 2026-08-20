"""Acces aux embeddings BGE-M3 par `llama-server --embedding` (regle 4 : llama.cpp seul).

Pourquoi ce fichier existe : `llama-embedding` N'EXISTE PAS (absent de WSL, de l'image et du
zip officiel b10175) — llama.cpp l'a absorbe dans `llama-server --embedding`. Recette prouvee
par mesure le 17/08 (sonde d'embedding, outillage de calibration non redistribue) ; elle est
centralisee ici pour que index.py, verify_index.py et le service runtime du D5 ne la
reimplementent pas chacun de son cote.

Invariants ECRITS, pas constates — une sortie saine aujourd'hui ne reste pas saine :
  * `--pooling cls` est passe EXPLICITEMENT. CLS est deja le defaut de ce GGUF (ecart mesure
    0,000e+00 contre le defaut), mais le journal du serveur n'imprime PAS le pooling, meme a
    `-lv 3` : personne ne verrait une derive. Or un cosinus n'a aucun sens hors de son pooling
    (mean donne 0,8274 / 0,6476 la ou cls donne 0,6578 / 0,3232).
  * la norme des vecteurs est ASSERTEE, jamais reimposee. Le serveur normalise deja (norme
    mesuree a 1,000000 exactement) ; renormaliser a l'aveugle masquerait un changement de
    comportement du serveur au lieu de le reveler.
  * la taille du GGUF est verifiee A L'OCTET avant le premier appel : garde, pas doute.

Bornes par construction — un flag verifie EXISTANT mais jamais verifie AGISSANT a fait
ecrire 5,52 Go en 33 min et tuer Docker : aucun volume inscriptible,
attente de /health plafonnee, serveur tue dans un `finally` — jamais laisse en vie.
Reseau : 127.0.0.1 uniquement. Rien ne sort de la machine.
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import time
import urllib.error
import urllib.request

import numpy as np

RACINE = pathlib.Path(__file__).resolve().parent.parent

# Hors du repo : le repo est un fork PUBLIC, garder 18 Mo de binaires dehors est une
# protection par construction. Surchargeable pour le D6 (image Docker, chemin different).
BINAIRE = pathlib.Path(
    os.environ.get(
        "ADTC_LLAMA_SERVER",
        str(RACINE.parent / "llama-b10175-win-cpu-x64" / "llama-server.exe"),
    )
)
GGUF = RACINE / "model" / "bge-m3" / "bge-m3-Q8_0.gguf"

GGUF_OCTETS = 634_553_760          # mesure, download_model.sh
DIM = 1024                         # BGE-M3 dense, mesure
POOLING = "cls"                    # invariant ecrit (voir docstring)
NORME_ATTENDUE = 1.0
NORME_TOLERANCE = 1e-3
ATTENTE_MAX_S = 180.0


class ErreurEmbedding(RuntimeError):
    """Le serveur d'embeddings n'a pas produit ce que le contrat annonce."""


class Serveur:
    """Cycle de vie du `llama-server --embedding`, utilisable en gestionnaire de contexte."""

    def __init__(self, port: int = 8643, n_ctx: int = 4096, n_batch: int = 4096,
                 journal: pathlib.Path | None = None, verbeux: bool = True):
        self.port = port
        self.n_ctx = n_ctx
        self.n_batch = n_batch
        self.base = f"http://127.0.0.1:{port}"
        self.journal = journal or (pathlib.Path(__file__).parent / "_embed_server.log")
        self.verbeux = verbeux
        self.proc: subprocess.Popen | None = None
        self.n_appels = 0
        self.n_textes = 0

    # ---------------------------------------------------------------- gardes
    def gardes(self) -> None:
        if not BINAIRE.is_file():
            raise ErreurEmbedding(f"llama-server introuvable : {BINAIRE}")
        if not GGUF.is_file():
            raise ErreurEmbedding(f"GGUF BGE-M3 introuvable : {GGUF}")
        n = GGUF.stat().st_size
        if n != GGUF_OCTETS:
            raise ErreurEmbedding(
                f"GGUF BGE-M3 : {n} octets, {GGUF_OCTETS} attendus. Un GGUF tronque produit "
                "des vecteurs plausibles et faux — on refuse d'indexer 33 documents dessus."
            )
        if self.verbeux:
            print(f"ok     GGUF BGE-M3 : {n:,} octets == attendu (garde a l'octet)")

    # -------------------------------------------------------------- demarrage
    def __enter__(self) -> "Serveur":
        self.demarre()
        return self

    def __exit__(self, *_exc) -> None:
        self.arrete()

    def demarre(self) -> None:
        self.gardes()
        cmd = [
            str(BINAIRE), "-m", str(GGUF),
            "--embedding",
            "--pooling", POOLING,          # invariant ECRIT
            "--host", "127.0.0.1", "--port", str(self.port),
            "-c", str(self.n_ctx),
            "-b", str(self.n_batch), "-ub", str(self.n_batch),
        ]
        with open(self.journal, "wb") as fh:
            self.proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
        if self.verbeux:
            print(f"       serveur lance (pooling={POOLING}, n_ctx={self.n_ctx}, "
                  f"n_batch={self.n_batch}), attente de /health…")
        debut = time.time()
        while time.time() - debut < ATTENTE_MAX_S:
            if self.proc.poll() is not None:
                raise ErreurEmbedding(
                    f"le serveur est mort (code {self.proc.returncode}) — voir {self.journal}")
            try:
                with urllib.request.urlopen(self.base + "/health", timeout=3) as r:
                    if json.loads(r.read().decode("utf-8")).get("status") == "ok":
                        if self.verbeux:
                            print(f"ok     serveur pret en {time.time() - debut:.1f} s")
                        return
            except (urllib.error.URLError, OSError, json.JSONDecodeError):
                time.sleep(0.5)
        raise ErreurEmbedding(f"/health n'a pas repondu ok en {ATTENTE_MAX_S:.0f} s")

    def arrete(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)
            if self.verbeux:
                print("       serveur arrete")
        self.proc = None

    # ------------------------------------------------------------------ appels
    def _poste(self, route: str, charge: dict, delai: float = 600.0) -> dict:
        req = urllib.request.Request(
            self.base + route,
            data=json.dumps(charge).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST")
        with urllib.request.urlopen(req, timeout=delai) as r:
            return json.loads(r.read().decode("utf-8"))

    def jetons(self, texte: str) -> int:
        """Nombre de jetons du tokenizer du modele — pour dimensionner les lots et mesurer
        le rapport caracteres/jetons du corpus (fr et en ne le partagent pas)."""
        return len(self._poste("/tokenize", {"content": texte})["tokens"])

    def plonge(self, textes: list[str], lot: int = 8) -> np.ndarray:
        """Renvoie (len(textes), 1024) float32. Ordre d'entree preserve."""
        if not textes:
            return np.zeros((0, DIM), dtype=np.float32)
        sortie = np.empty((len(textes), DIM), dtype=np.float32)
        vus = 0
        for depart in range(0, len(textes), lot):
            paquet = textes[depart:depart + lot]
            rep = self._poste("/v1/embeddings", {"input": paquet})
            donnees = rep.get("data")
            if not isinstance(donnees, list) or len(donnees) != len(paquet):
                raise ErreurEmbedding(
                    f"{len(donnees) if isinstance(donnees, list) else '?'} vecteurs pour "
                    f"{len(paquet)} textes : le serveur n'a pas repondu 1 pour 1")
            # L'API renvoie un champ `index` : on s'en sert au lieu de supposer l'ordre.
            donnees = sorted(donnees, key=lambda d: d.get("index", 0))
            for k, d in enumerate(donnees):
                v = d["embedding"]
                if v and isinstance(v[0], list):     # pooling none -> matrice ; refus net
                    raise ErreurEmbedding(
                        "le serveur renvoie une matrice par texte : le pooling n'est pas actif")
                if len(v) != DIM:
                    raise ErreurEmbedding(f"dimension {len(v)}, {DIM} attendue (BGE-M3 dense)")
                sortie[depart + k] = np.asarray(v, dtype=np.float32)
            vus += len(paquet)
            self.n_appels += 1
            self.n_textes += len(paquet)
            if self.verbeux and (self.n_appels % 20 == 0 or vus == len(textes)):
                print(f"       embeddings {vus}/{len(textes)}", flush=True)

        # Norme ASSERTEE, pas reimposee : une derive signifie que le serveur a change de
        # comportement, et on veut le savoir plutot que le masquer.
        normes = np.linalg.norm(sortie.astype(np.float64), axis=1)
        ecart = float(np.max(np.abs(normes - NORME_ATTENDUE)))
        if ecart > NORME_TOLERANCE:
            raise ErreurEmbedding(
                f"norme max ecartee de {ecart:.3e} de 1,0 (tolerance {NORME_TOLERANCE:.0e}) : "
                "le serveur ne normalise plus. Le cosinus ne se reduit plus au produit "
                "scalaire — verifier --embd-normalize avant de continuer.")
        return sortie


def cosinus(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosinus lignes-a-lignes ou matrice-vecteur, sans supposer la normalisation."""
    a = np.atleast_2d(np.asarray(a, dtype=np.float64))
    b = np.atleast_2d(np.asarray(b, dtype=np.float64))
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return a @ b.T
