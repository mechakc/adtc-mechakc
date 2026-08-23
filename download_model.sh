#!/usr/bin/env bash
# download_model.sh — recupere les poids GGUF necessaires a la soumission.
#
# Contraintes officielles de soumission, tenues ici :
#   - idempotent (relancable sans effet de bord) ;
#   - AUCUN credential (URL publiques uniquement) ;
#   - le chemin de sortie doit correspondre a `_runtime.model_path` de metadata.json.
#
# DEUX modeles, deux roles distincts — ne pas les confondre. Ils sont TOUS DEUX en
# Q8_0 depuis le 2026-08-18, mais pour des raisons DIFFERENTES (cf. plus bas) :
#   1. LLM de generation      -> Qwen2.5-0.5B-Instruct Q8_0. C'EST LUI que le
#      profileur charge (`_runtime.model_path`), donc lui seul pese sur S_perf/S_eff.
#      Q8_0 mesure le 2026-08-18 contre Q4_K_M et Q4_0, 3 runs profileur n=300 :
#      Q4_K_M elimine sous tous les regimes (+7,96 points pour
#      Q8_0). Ce n'est PAS « un quant plus large est plus fidele » — Q8_0 rend
#      MOINS d'arc_easy que Q4_K_M (59,33 contre 61,33) ; il gagne parce qu'il va
#      2,27x plus vite et que S_perf sature. Le spot-check en francais a ecarte
#      Q4_0 (2 boucles degenerees sur 6 a temperature 0).
#   2. Modele d'embedding     -> BGE-M3 Q8_0, pour le RAG (rag/index.py, rag/retrieve.py).
#      Charge par NOTRE application uniquement, JAMAIS par le profileur => son
#      RSS ne coute rien sur S_eff. Choisi multilingue parce que le corpus est a
#      75 % en francais ; regle 4 « llama.cpp only » => GGUF via llama.cpp,
#      jamais sentence-transformers.
#      Son Q8_0 tient a une raison SANS RAPPORT avec celle du LLM : en recherche
#      vectorielle la qualite depend de distinctions fines de cosinus, et le
#      surcout est gratuit ici (jamais charge en telemetrie). Arbitrage tranche au
#      D4 et non rouvert : l'index livre est bati avec ce GGUF precis (son sha256
#      est dans rag/index/manifest.json) — en changer invaliderait les 3 180
#      vecteurs deja committes.
#
# Tailles verifiees a l'octet par requete HEAD (pas une estimation) :
#   qwen2.5-0.5b-instruct-q8_0.gguf    675 710 816 o  <- HEAD 2026-08-18 : X-Linked-Size
#                                                        ET content-length du CDN ;
#                                                        identique au fichier local
#   bge-m3-Q8_0.gguf                   634 553 760 o  <- HEAD 2026-08-16

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$HERE/model"

# ── LLM de generation (obligatoire — sans lui le profileur ne peut pas tourner) ─
LLM_REL="Qwen2.5-0.5B/qwen2.5-0.5b-instruct-q8_0.gguf"
LLM_URL="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q8_0.gguf"
# 🔴 LLM_BYTES n'est PAS documentaire : c'est `$want` dans fetch(), compare a QUATRE
# endroits — contre `$have` (fichier deja present), contre `$pre` deux fois (.partial
# d'un essai precedent : egal, ou plus grand donc irreparable), et contre `$got`
# (fin de chaque tentative). Une valeur perimee ici SUPPRIME un GGUF sain
# (`rm -f "$dest"` sur la branche « taille INATTENDUE ») puis fait tourner la boucle
# a vide sur une liaison qui marche.
# ⚠️ Ces sites sont nommes par VARIABLE et non par numero de ligne a dessein : la
# version precedente citait « :70, :87, :91, :131 » et etait deja fausse d'un cran
# AVANT la refonte de la boucle du 2026-08-24, qui l'a decalee de 19 lignes de plus.
# Un renvoi a moitie juste survit a la relecture mieux qu'un renvoi entierement faux.
# Tout changement de quant doit passer par cette ligne : une checklist de changement
# de quant qui l'omet fait croire au tour complet — la notre l'omettait.
LLM_BYTES=675710816

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
  #   2. `set -euo pipefail` (ligne 37) abrege le script des que curl sort non-nul,
  #      donc AVANT le nettoyage plus bas — le .partial restait sur le disque, et
  #      il n'etait couvert par aucune regle de .gitignore.
  # D'ou : boucle de reprise explicite, `-C -`, et .partial conserve a dessein en
  # cas d'abandon (il est desormais ignore par git, cf. .gitignore `model/`).
  # 🔴 PANNE MESUREE le 2026-08-24, en repertoire vierge : la version precedente
  # bornait a 5 tentatives FIXES. Sur une liaison a ~400 ko/s chaque appel curl meurt
  # avant la fin, la reprise `-C -` marche (le .partial grandit a chaque essai), mais
  # les 5 tentatives s'epuisent vers 450 Mo sur 644 => le script sort 1 ALORS QUE LA
  # LIAISON EST VIVANTE. Un juge qui lance le script une fois voit un echec, et
  # « download_model.sh fails » est un disqualifiant nomme.
  # Le compteur etait la mauvaise grandeur : ce qui distingue une liaison lente d'une
  # liaison morte n'est pas le nombre d'essais, c'est le PROGRES. On compte donc les
  # essais STERILES (zero octet gagne). Une liaison lente mais vivante finit toujours ;
  # une liaison morte abandonne toujours au bout de 5 essais steriles. Borne dure a
  # 200 iterations pour qu'aucun cas ne puisse boucler sans fin.
  local tentative=0 got=0 steriles=0 avant=0
  if [[ -f "$part" ]]; then avant=$(wc -c < "$part" | tr -d '[:space:]'); fi
  while (( tentative < 200 )); do
    tentative=$((tentative + 1))
    echo "[$label] tentative $tentative -> $rel ($((want / 1024 / 1024)) Mo, acquis $avant o)…"
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

    # 🔴 `[[ -f "$part" ]] && got=$(…)` etait un piege latent, et precisement sur le
    # chemin d'echec qui nous interesse : si curl meurt avant de creer le .partial
    # (DNS, refus immediat), la liste `&&` sort non-nulle, `set -e` (ligne 37) abrege
    # le script AU MILIEU de la boucle, sans le message d'erreur ni la ligne qui dit
    # ou reprendre. Forme `if` : le test ne peut plus etre le code de sortie.
    got=0
    if [[ -f "$part" ]]; then got=$(wc -c < "$part" | tr -d '[:space:]'); fi
    if [[ "$got" == "$want" ]]; then
      # `mv` en dernier : tant que le fichier s'appelle .partial, une interruption
      # ne laisse jamais un GGUF incomplet a l'emplacement attendu.
      mv "$part" "$dest"
      echo "[$label] OK : $rel ($got o)"
      return 0
    fi
    if (( got > avant )); then
      steriles=0
      echo "[$label] incomplet : $got o / $want o (+$((got - avant)) o) — progres, on continue" >&2
    else
      steriles=$((steriles + 1))
      echo "[$label] incomplet : $got o / $want o — essai sterile $steriles/5" >&2
      if (( steriles >= 5 )); then break; fi
    fi
    avant=$got
  done

  echo "erreur [$label] : abandon apres $tentative tentatives, dont $steriles sans progres." >&2
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
