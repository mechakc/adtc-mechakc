#!/usr/bin/env bash
# download_model.sh — recupere les poids GGUF necessaires a la soumission.
#
# Contraintes officielles (checklist ADTC, cf. CLAUDE.md §6.4) :
#   - idempotent (relancable sans effet de bord) ;
#   - AUCUN credential (URL publiques uniquement) ;
#   - le chemin de sortie doit correspondre a `_runtime.model_path` de metadata.json.
#
# DEUX modeles, deux roles distincts — ne pas les confondre :
#   1. LLM de generation      -> Qwen2.5-0.5B-Instruct Q4_K_M. C'EST LUI que le
#      profileur charge (`_runtime.model_path`), donc lui seul pese sur S_perf/S_eff.
#   2. Modele d'embedding     -> BGE-M3 Q8_0, pour le RAG (rag/index.py, rag/retrieve.py).
#      Charge par NOTRE application uniquement, JAMAIS par le profileur => son
#      RSS ne coute rien sur S_eff. Choisi multilingue parce que le corpus est a
#      75 % en francais ; regle 4 « llama.cpp only » => GGUF via llama.cpp,
#      jamais sentence-transformers.
#      Q8_0 (et non Q4_K_M) : en recherche vectorielle la qualite depend de
#      distinctions fines de cosinus, et le surcout est gratuit ici (jamais
#      charge en telemetrie). Revisable au D4 si Q4_K_M retrouve autant.
#
# Tailles verifiees a l'octet par requete HEAD le 2026-08-16 (pas une estimation) :
#   qwen2.5-0.5b-instruct-q4_k_m.gguf  491 400 032 o  <- identique au fichier local
#   bge-m3-Q8_0.gguf                   634 553 760 o

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$HERE/model"

# ── LLM de generation (obligatoire — sans lui le profileur ne peut pas tourner) ─
LLM_REL="Qwen2.5-0.5B/qwen2.5-0.5b-instruct-q4_k_m.gguf"
LLM_URL="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"
LLM_BYTES=491400032

# ── Modele d'embedding pour le RAG (obligatoire pour repondre, pas pour mesurer) ─
EMB_REL="bge-m3/bge-m3-Q8_0.gguf"
EMB_URL="https://huggingface.co/gpustack/bge-m3-GGUF/resolve/main/bge-m3-Q8_0.gguf"
EMB_BYTES=634553760

fetch() {
  local rel="$1" url="$2" want="$3" label="$4"
  local dest="$MODEL_DIR/$rel"
  local part="$dest.partial"
  mkdir -p "$(dirname "$dest")"

  # Idempotence + verification d'integrite. Un fichier present MAIS tronque (coupure
  # reseau d'un run precedent) est le pire cas : il passerait un simple test -f et
  # ferait echouer le chargement plus tard, avec un message opaque. On compare donc
  # la taille, et on retelecharge si elle ne colle pas.
  if [[ -f "$dest" ]]; then
    local have
    have=$(wc -c < "$dest" | tr -d '[:space:]')
    if [[ "$have" == "$want" ]]; then
      echo "[$label] deja present et complet ($have o) — rien a faire"
      # Un .partial oublie par un essai precedent ne sert plus a rien et serait
      # ramasse par un `git add` (mesure le 17/08 : 44 671 000 o mis en scene).
      rm -f "$part"
      return 0
    fi
    echo "[$label] present mais taille INATTENDUE ($have o, attendu $want o) — retelechargement" >&2
    rm -f "$dest"
  fi

  # Etat d'un telechargement precedent. Trois cas, et le cas ">" n'est pas theorique :
  # une reprise `-C -` au-dela de la fin du fichier distant produirait un GGUF
  # corrompu qui PASSERAIT le test de taille — un faux positif silencieux.
  if [[ -f "$part" ]]; then
    local pre
    pre=$(wc -c < "$part" | tr -d '[:space:]')
    if (( pre == want )); then
      mv "$part" "$dest"
      echo "[$label] .partial deja complet ($pre o) — promu sans retelecharger"
      return 0
    elif (( pre > want )); then
      echo "[$label] .partial de $pre o > cible $want o : irreparable, jete" >&2
      rm -f "$part"
    else
      echo "[$label] reprise d'un essai precedent a $pre o / $want o"
    fi
  fi

  # 🔴 PANNE MESUREE le 17/08 : la version precedente lancait
  #     curl -L --fail --retry 3 --retry-delay 5 --progress-bar -o "$dest.partial"
  # sans `-C -`. Deux defauts qui se combinent :
  #   1. `--retry` de curl RECOMMENCE le transfert a zero ; sur une liaison lente,
  #      634 Mo n'aboutissent jamais (constate : mort a 44 671 000 o).
  #   2. `set -euo pipefail` (ligne 25) abrege le script des que curl sort non-nul,
  #      donc AVANT le nettoyage plus bas — le .partial restait sur le disque, et
  #      il n'etait couvert par aucune regle de .gitignore.
  # D'ou : boucle de reprise explicite, `-C -`, et .partial conserve a dessein en
  # cas d'abandon (il est desormais ignore par git, cf. .gitignore `model/*`).
  local tentative got
  for tentative in 1 2 3 4 5; do
    echo "[$label] tentative $tentative/5 -> $rel ($((want / 1024 / 1024)) Mo)…"
    if command -v curl >/dev/null 2>&1; then
      # --speed-limit/--speed-time : une socket morte abandonne au bout de 60 s au
      # lieu de pendre indefiniment. 1 ko/s est un plancher qu'une liaison lente
      # mais vivante depasse largement (mesure du 17/08 : ~125 ko/s).
      curl -L --fail --retry 3 --retry-delay 5 -C - \
           --speed-limit 1024 --speed-time 60 \
           --progress-bar -o "$part" "$url" || true
    elif command -v wget >/dev/null 2>&1; then
      # Branche de repli : la reprise de wget combinee a -O est moins fiable que
      # celle de curl (avertissement du manuel wget), mais curl est present dans
      # Git Bash, WSL et l'image Docker — ce chemin ne doit jamais servir.
      wget --continue --tries=3 --show-progress --output-document "$part" "$url" || true
    else
      echo "erreur : ni curl ni wget disponible" >&2
      return 1
    fi

    got=0
    [[ -f "$part" ]] && got=$(wc -c < "$part" | tr -d '[:space:]')
    if [[ "$got" == "$want" ]]; then
      # `mv` en dernier : tant que le fichier s'appelle .partial, une interruption
      # ne laisse jamais un GGUF incomplet a l'emplacement attendu.
      mv "$part" "$dest"
      echo "[$label] OK : $rel ($got o)"
      return 0
    fi
    echo "[$label] incomplet : $got o / $want o" >&2
  done

  echo "erreur [$label] : abandon apres 5 tentatives." >&2
  echo "  le .partial est CONSERVE — relancer ce script reprend ou il s'est arrete :" >&2
  echo "  $part" >&2
  return 1
}

fetch "$LLM_REL" "$LLM_URL" "$LLM_BYTES" "LLM"
fetch "$EMB_REL" "$EMB_URL" "$EMB_BYTES" "embedding"

echo
echo "termine. Chemins attendus par metadata.json :"
echo "  _runtime.model_path     = model/$LLM_REL"
echo "  embedding (RAG)         = model/$EMB_REL"
