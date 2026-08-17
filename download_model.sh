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
      return 0
    fi
    echo "[$label] present mais taille INATTENDUE ($have o, attendu $want o) — retelechargement" >&2
    rm -f "$dest"
  fi

  echo "[$label] telechargement -> $rel ($((want / 1024 / 1024)) Mo)…"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 3 --retry-delay 5 --progress-bar -o "$dest.partial" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget --tries=3 --show-progress -O "$dest.partial" "$url"
  else
    echo "erreur : ni curl ni wget disponible" >&2
    return 1
  fi

  local got
  got=$(wc -c < "$dest.partial" | tr -d '[:space:]')
  if [[ "$got" != "$want" ]]; then
    echo "erreur [$label] : $got o telecharges, $want o attendus — fichier ecarte" >&2
    rm -f "$dest.partial"
    return 1
  fi
  # `mv` en dernier : tant que le fichier s'appelle .partial, une interruption ne
  # laisse jamais un GGUF incomplet a l'emplacement attendu.
  mv "$dest.partial" "$dest"
  echo "[$label] OK : $rel ($got o)"
}

fetch "$LLM_REL" "$LLM_URL" "$LLM_BYTES" "LLM"
fetch "$EMB_REL" "$EMB_URL" "$EMB_BYTES" "embedding"

echo
echo "termine. Chemins attendus par metadata.json :"
echo "  _runtime.model_path     = model/$LLM_REL"
echo "  embedding (RAG)         = model/$EMB_REL"
