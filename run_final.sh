#!/bin/bash
# =============================================================================
# run_final.sh — CHEMIN DE PRODUCTION UNIQUE DU D6 (annonce par Dockerfile:34)
# =============================================================================
# Pourquoi un script dedie alors que bakeoff_accuracy_smoke.sh existe : ce
# dernier est INUTILISABLE pour un run final, pour deux raisons mesurees.
#   1. `:60` exclut `.git` de la copie temporaire => reproducibility.git_commit_sha
#      sort a "000000000000". Ce fallback matche ^[a-f0-9]{7,40}$ donc AUCUNE
#      validation de schema ne l'attrape : echec silencieux sur un champ que
#      reproducibility.py exige.
#   2. `:151-162` reecrit model["name"] dans la copie => le libelle du bake-off
#      partirait dans submission.model.name, champ lu par les juges.
# Ici : le repo LUI-MEME est monte en lecture seule (donc `.git` est present, et
# aucun run ne PEUT ecrire dans la soumission), et metadata.json n'est jamais
# reecrit.
#
# Usage :
#   ./run_final.sh preflight     # rapide, sans profileur : prouve les invariants
#   ./run_final.sh runs [N]      # N runs complets (defaut 3), 1 JSON par run
#
# Invariants non negociables (CLAUDE.md §4, §6.2, §6.7, §12.2) :
#   - `--mode participant` EN DUR : `audit` forcerait measured_on=audit_cloud_vm,
#     seul echec dur atteignable par notre propre erreur.
#   - JAMAIS `--skip-accuracy` : un lm-eval qui plante rend accuracy:[] + exit 0.
#   - le verdict se lit DANS LE JSON, jamais sur le code retour.
#   - jamais `-t` : llama-bench resout 2 threads seul ; un -t force produirait un
#     chiffre que l'audit ne peut pas reproduire, c.-a-d. l'echec dur lui-meme.
#   - 3 runs, pas un : tg porte 7,75 a 17,77 % de dispersion interne dans une
#     bande de tolerance de 25 % => un run unique est un tirage, pas une mesure.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="adtc-agri:canari"
OUT_DIR="$REPO_ROOT/final_runs"
HF_CACHE="$REPO_ROOT/.bakeoff_hf_cache"
ACCURACY_TASK="arc_easy"
ACCURACY_LIMIT=300
SEED=42
MEMORY="7.5g"
# Injection de config git PAR L'ENVIRONNEMENT : `safe.directory` sans fichier a
# ecrire, donc compatible avec le montage :ro. Sans elle, un ecart d'UID entre
# l'hote et le conteneur donne « dubious ownership » => sha a 000000000000.
GIT_ENV=(-e GIT_CONFIG_COUNT=1 -e GIT_CONFIG_KEY_0=safe.directory
         -e GIT_CONFIG_VALUE_0=/submission)

# Garde d'argument : aucune option de cette famille ne doit pouvoir se glisser
# depuis la ligne de commande, meme par copier-coller d'un ancien run.
for a in "$@"; do
  case "$a" in
    *audit*|*skip*accuracy*|*-t*[0-9]*)
      echo "REFUS: argument interdit ici ($a)" >&2; exit 2;;
  esac
done

MODE="${1:-preflight}"

# --- outil commun : lit une valeur de metadata.json sans deviner de cle --------
meta() {
  python3 - "$REPO_ROOT/metadata.json" "$1" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
cur = d
for part in sys.argv[2].split("."):
    if not isinstance(cur, dict) or part not in cur:
        print(""); raise SystemExit
    cur = cur[part]
print(cur)
PY
}

# =============================================================================
# PREFLIGHT — tout ce qui doit etre VRAI avant de depenser 3 runs
# =============================================================================
preflight() {
  local fail=0
  echo "=== 1. docker joignable et image presente ==="
  docker info --format 'client={{.ClientInfo.Version}} server={{.ServerVersion}}' \
    || { echo "ECHEC: docker indisponible (Docker Desktop demarre ? WSL Integration Ubuntu-22.04 cochee ?)" >&2; return 2; }
  docker image inspect "$IMAGE" --format 'image Id={{.Id}}' \
    || { echo "ECHEC: image $IMAGE absente — docker build --provenance=false -t $IMAGE ." >&2; return 2; }

  echo
  echo "=== 2. metadata.json : chemin du modele et digest declares ==="
  local mp dg gguf n
  mp="$(meta _runtime.model_path)"
  dg="$(meta _runtime.docker_image)"
  [ -n "$mp" ] || { echo "ECHEC: _runtime.model_path absent"; fail=1; }
  gguf="$REPO_ROOT/$mp"
  if [ -f "$gguf" ]; then
    n=$(stat -c%s "$gguf")
    echo "ok   model_path=$mp ($n octets)"
    [ "$n" = "675710816" ] || { echo "ECHEC: $n octets, 675710816 attendus (Q8_0)"; fail=1; }
  else
    echo "ECHEC: GGUF introuvable a $gguf"; fail=1
  fi
  # docker_image absent => reproducibility.py rend "unknown" SANS RIEN TENTER,
  # c.-a-d. exactement le champ ou 43/44 rivaux echouent. Bloquant, pas cosmetique.
  if [ -z "$dg" ]; then
    echo "ECHEC: _runtime.docker_image absent => docker_image_digest sortira a 'unknown'"; fail=1
  else
    echo "ok   docker_image=$dg"
    case "$dg" in
      *@sha256:*) : ;;
      *) echo "NOTE: reference sans digest — un tag est mutable, un digest ne l'est pas";;
    esac
  fi
  echo
  echo "=== 3. git DANS le conteneur (le champ ou l'echec est silencieux) ==="
  # Le fallback de reproducibility.py est "0"*12, valide au schema => on ne peut
  # PAS le detecter apres coup sur le JSON seul sans le comparer a l'hote. On
  # prouve donc l'appel ici, dans l'image, sur le montage :ro reellement utilise.
  local host_sha bare inj
  host_sha="$(git -C "$REPO_ROOT" rev-parse --short=12 HEAD 2>/dev/null)"
  echo "hote : $host_sha"
  bare="$(docker run --rm -v "$REPO_ROOT:/submission:ro" --entrypoint /bin/sh "$IMAGE" \
            -c 'git -C /submission rev-parse --short=12 HEAD 2>&1' | tr -d '\r')"
  echo "conteneur, sans injection : $bare"
  inj="$(docker run --rm -v "$REPO_ROOT:/submission:ro" "${GIT_ENV[@]}" --entrypoint /bin/sh "$IMAGE" \
            -c 'git -C /submission rev-parse --short=12 HEAD 2>&1' | tr -d '\r')"
  echo "conteneur, avec injection safe.directory : $inj"
  if [ "$inj" = "$host_sha" ] && [ -n "$host_sha" ]; then
    echo "ok   le conteneur lit le MEME commit que l'hote"
  else
    echo "ECHEC: le conteneur ne retrouve pas $host_sha => git_commit_sha sortira faux"; fail=1
  fi
  # L'arbre doit etre propre sur les fichiers soumis : un sha qui designe un arbre
  # different de ce qu'on mesure est un faux champ de reproductibilite.
  local dirty
  dirty="$(git -C "$REPO_ROOT" status --porcelain -- metadata.json run_final.sh Dockerfile download_model.sh)"
  if [ -n "$dirty" ]; then
    echo "NOTE: modifie et non committe (le sha declare ne contiendra pas ces etats) :"
    echo "$dirty"
  else
    echo "ok   metadata.json / run_final.sh / Dockerfile / download_model.sh sont committes"
  fi

  echo
  echo "=== 4. cache HuggingFace chaud pour $ACCURACY_TASK ==="
  # Sans cache chaud, une coupure reseau degrade en accuracy:[] avec exit 0.
  if [ -d "$HF_CACHE/datasets/allenai___ai2_arc" ]; then
    echo "ok   $(du -sh "$HF_CACHE/datasets/allenai___ai2_arc" | cut -f1) en cache"
  else
    echo "NOTE: cache ARC absent — le 1er run telechargera (reseau requis)"
  fi

  echo
  [ $fail -eq 0 ] && echo "PREFLIGHT OK" || echo "PREFLIGHT EN ECHEC"
  return $fail
}
# =============================================================================
# INSPECTION D'UN JSON — le SEUL verdict qui compte (jamais le code retour)
# =============================================================================
inspect_json() {
  python3 - "$1" "$2" <<'PY'
import json, sys
path, host_sha = sys.argv[1], sys.argv[2]
try:
    d = json.load(open(path, encoding="utf-8"))
except Exception as e:
    print("ECHEC JSON illisible (%s)" % e); raise SystemExit
bad = []
sub = d.get("submission") or {}
env = d.get("environment") or {}
tp  = d.get("throughput") or {}
mem = d.get("memory") or {}
rep = d.get("reproducibility") or {}
mi  = d.get("model_info") or {}
th  = d.get("cpu_thermal") or {}
acc = d.get("accuracy") or []

if not acc:
    bad.append("accuracy VIDE (exit 0 trompeur, cli.py:151-161)")
else:
    a = acc[0]
    if not isinstance(a.get("score"), (int, float)) or a.get("score") == 0:
        bad.append("accuracy[0].score=%r" % a.get("score"))
if env.get("measured_on") != "participant_laptop":
    bad.append("measured_on=%r (echec dur structurel)" % env.get("measured_on"))
sha = rep.get("git_commit_sha")
if sha in (None, "000000000000") or sha != host_sha:
    bad.append("git_commit_sha=%r attendu %r" % (sha, host_sha))
dig = rep.get("docker_image_digest")
if not dig or dig == "unknown":
    bad.append("docker_image_digest=%r" % dig)
if rep.get("random_seed") != 42:
    bad.append("random_seed=%r" % rep.get("random_seed"))
# Les 4 champs a tolerance du comparateur : `missing` ou 0 => fail sans passer
# par les bandes, donc un champ absent ne vaut pas mieux qu'un mauvais delta.
NUM = [("throughput.tokens_per_second_generation", tp.get("tokens_per_second_generation")),
       ("throughput.first_token_latency_ms", tp.get("first_token_latency_ms")),
       ("memory.peak_rss_mb", mem.get("peak_rss_mb")),
       ("memory.steady_state_rss_mb", mem.get("steady_state_rss_mb"))]
for k, v in NUM:
    if not isinstance(v, (int, float)) or isinstance(v, bool) or v == 0:
        bad.append("%s=%r" % (k, v))
if mi.get("params_count") != 630167424:
    bad.append("params_count=%r attendu 630167424" % mi.get("params_count"))
if mi.get("params_match") is not True:
    bad.append("params_match=%r" % mi.get("params_match"))
if sub.get("model", {}).get("quantization") != "GGUF Q8_0":
    bad.append("quantization=%r" % sub.get("model", {}).get("quantization"))
if th.get("throttled") is True:
    bad.append("throttled=True (P_thermal -10)")
for k, v in sub.items():
    if isinstance(v, str) and ("your-" in v or "TODO" in v):
        bad.append("placeholder dans submission.%s" % k)

tg  = tp.get("tokens_per_second_generation")
ftl = tp.get("first_token_latency_ms")
rss = mem.get("peak_rss_mb")
if bad:
    print("ECHEC " + " | ".join(bad))
else:
    sperf = min(tg / 15.0, 1.0) * 100.0
    seff = max(0.0, (7.0 - rss / 1000.0) / 7.0) * 100.0
    print("OK acc=%.2f tg=%.2f ftl=%.0f peak=%.2f steady=%.2f Sperf=%.2f Seff=%.2f sha=%s"
          % (acc[0]["score"] * (100.0 if acc[0]["score"] <= 1.0 else 1.0),
             tg, ftl, rss, mem.get("steady_state_rss_mb"), sperf, seff, sha))
PY
}
# =============================================================================
# RUNS — N runs complets, 1 JSON + 1 log brut par run
# =============================================================================
do_runs() {
  local n="${1:-3}"
  preflight || { echo "runs annules : preflight en echec" >&2; return 2; }
  mkdir -p "$OUT_DIR" "$HF_CACHE"
  local host_sha failed=()
  host_sha="$(git -C "$REPO_ROOT" rev-parse --short=12 HEAD)"

  local i out log rc verdict
  for i in $(seq 1 "$n"); do
    out="$OUT_DIR/submission_run$i.json"
    log="$OUT_DIR/submission_run$i.log"
    rm -f "$out"
    echo
    echo "=== RUN $i/$n — $(date -u +%H:%M:%SZ) ==="
    # Montage :ro du repo LUI-MEME : `.git` present (sha reel), et aucun run ne
    # peut ecrire dans la soumission. Seuls /artifacts et /hf_cache sont
    # inscriptibles. Aucun `-t` : llama-bench resout ses threads seul.
    docker run --rm \
      --memory="$MEMORY" \
      -e HF_HOME=/hf_cache \
      "${GIT_ENV[@]}" \
      -v "$REPO_ROOT:/submission:ro" \
      -v "$OUT_DIR:/artifacts" \
      -v "$HF_CACHE:/hf_cache" \
      "$IMAGE" run \
        --submission /submission \
        --mode participant \
        --output "/artifacts/submission_run$i.json" \
        --seed "$SEED" \
        --accuracy-task "$ACCURACY_TASK" \
        --accuracy-limit "$ACCURACY_LIMIT" \
      > "$log" 2>&1
    rc=$?
    echo "  exit docker=$rc (INDICATIF — le verdict est dans le JSON)"
    verdict="$(inspect_json "$out" "$host_sha")"
    echo "  -> $verdict"
    echo "  -> $out | $log"
    case "$verdict" in OK*) : ;; *) failed+=("run$i");; esac
  done

  echo
  echo "=== DISPERSION SUR $n RUNS (la raison d'etre des 3 runs) ==="
  python3 - "$OUT_DIR" <<'PY'
import glob, json, os, statistics, sys
out = sys.argv[1]
rows = []
for p in sorted(glob.glob(os.path.join(out, "submission_run*.json"))):
    try:
        d = json.load(open(p, encoding="utf-8"))
    except Exception:
        continue
    tp, mem = d.get("throughput") or {}, d.get("memory") or {}
    acc = d.get("accuracy") or [{}]
    rows.append((os.path.basename(p), tp.get("tokens_per_second_generation"),
                 tp.get("first_token_latency_ms"), mem.get("peak_rss_mb"),
                 mem.get("steady_state_rss_mb"), (acc[0] or {}).get("score")))
if not rows:
    print("aucun JSON exploitable"); raise SystemExit
print("%-24s %8s %10s %9s %9s %7s" % ("fichier", "tg", "ftl_ms", "peak", "steady", "acc"))
for r in rows:
    print("%-24s %8.2f %10.0f %9.2f %9.2f %7s" % r)
def stat(i, label, unit=""):
    v = [r[i] for r in rows if isinstance(r[i], (int, float))]
    if len(v) < 2:
        return
    m, lo, hi = statistics.mean(v), min(v), max(v)
    cv = statistics.stdev(v) / m * 100.0
    print("%-8s moyenne %10.2f%s  min %9.2f  max %9.2f  etendue %5.2f %%  cv %5.2f %%"
          % (label, m, unit, lo, hi, (hi - lo) / m * 100.0, cv))
stat(1, "tg"); stat(2, "ftl"); stat(3, "peak"); stat(4, "steady")
print()
print("A DECLARER (moyenne des runs, recalculee ici et non recopiee) :")
tg = statistics.mean([r[1] for r in rows if isinstance(r[1], (int, float))])
rss = statistics.mean([r[3] for r in rows if isinstance(r[3], (int, float))])
print("  Sperf = min(tg/15,1)*100      = %.2f   (tg moyen %.2f)" % (min(tg / 15.0, 1.0) * 100.0, tg))
print("  Seff  = max(0,(7-Go)/7)*100   = %.2f   (peak moyen %.2f Mo)"
      % (max(0.0, (7.0 - rss / 1000.0) / 7.0) * 100.0, rss))
print("  ATTENTION: la bande du comparateur est de 25 % et le cv de tg lui en")
print("  mange une partie AVANT tout effet de regime — declarer en connaissance.")
PY

  if [ ${#failed[@]} -gt 0 ]; then
    echo; echo "RUNS EN ECHEC : ${failed[*]}"; return 1
  fi
  echo; echo "TOUS LES RUNS OK — copier le JSON retenu en submission.json puis"
  echo "python3 check_submission.py submission.json  (exit 0 avant tout commit)"
  return 0
}

case "$MODE" in
  preflight) preflight ;;
  runs)      do_runs "${2:-3}" ;;
  *) echo "usage: $0 preflight | runs [N]" >&2; exit 2 ;;
esac

