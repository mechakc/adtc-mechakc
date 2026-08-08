# ADTC 2026 — image de soumission (piste Agriculture, LLM local hors-ligne).
#
# Rôle de cette image : artefact de reproductibilité pour l'audit (Gate 2).
# Le profileur (reproducibility.py:37) se contente d'un `docker image inspect`
# pour renseigner `docker_image_digest` ; il n'exécute PAS cette image lui-même.
# On la construit néanmoins pour qu'elle exécute le profileur de bout en bout
# (pattern README « Docker Execution ») afin d'être valide quel que soit le mode
# de consommation de l'orchestrateur d'audit (pull-and-run OU build-from-repo).
#
# Calquée sur adtc-profiler/Dockerfile (référence officielle). Divergence unique :
# le stage py-build clone le profileur depuis git (nos sources ne sont pas dans
# ce repo de soumission) au lieu de copier des sources locales.
#
# ⚠️ Baseline CPU : AVX/AVX2/AVX512/FMA/F16C TOUS OFF (SSE pur). Sans ça,
# `llama.cpp -march=native` provoque « illegal instruction » sur la VM d'audit.
#
# Build (depuis la racine du repo) :
#   docker build -t adtc-agri:canari .
# Run participant (modèle monté depuis la soumission) :
#   docker run --rm --memory=7.5g -v "<repo>:/submission:ro" -v "<out>:/artifacts" \
#     adtc-agri:canari run --submission /submission --mode participant \
#     --output /artifacts/submission.json --skip-accuracy

# -----------------------------------------------------------------------------
# Stage 1 : llama.cpp portable (baseline SSE, aucun AVX) — parité VM d'audit
# -----------------------------------------------------------------------------
FROM debian:bookworm-slim AS llama-build

ARG LLAMACPP_REF=b10175
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch "${LLAMACPP_REF}" \
      https://github.com/ggerganov/llama.cpp.git /src/llama.cpp \
    && cd /src/llama.cpp \
    && cmake -B build \
        -DBUILD_SHARED_LIBS=OFF \
        -DGGML_NATIVE=OFF \
        -DGGML_AVX=OFF \
        -DGGML_AVX2=OFF \
        -DGGML_AVX512=OFF \
        -DGGML_FMA=OFF \
        -DGGML_F16C=OFF \
        -DGGML_BLAS=OFF \
        -DGGML_CUDA=OFF \
        -DGGML_METAL=OFF \
    && cmake --build build --config Release --target llama-bench llama-cli -j2

# -----------------------------------------------------------------------------
# Stage 2 : wheels du profileur (llama-cpp-python compile depuis source — pas de
# toolchain dans le runtime slim, donc les wheels se construisent ici)
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS py-build

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Build CPU portable — pas de -march=native : le wheel llama-cpp-python (utilisé
# par l'accuracy en mode audit, in-process) doit tourner sur n'importe quelle VM.
ENV CMAKE_ARGS="-DGGML_NATIVE=OFF"

# Épingler PROFILER_REF à un SHA/tag au J5 pour une repro exacte.
ARG PROFILER_REPO=https://github.com/Africa-Deep-Tech-Foundation/adtc-profiler.git
ARG PROFILER_REF=main
WORKDIR /opt
RUN git clone --depth 1 --branch "${PROFILER_REF}" "${PROFILER_REPO}" adtc-profiler
WORKDIR /opt/adtc-profiler
RUN pip wheel --no-cache-dir --wheel-dir /wheels .

# -----------------------------------------------------------------------------
# Stage 3 : runtime du profileur
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl ca-certificates lm-sensors libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# llama-bench (débit) + llama-cli sur le PATH — shutil.which les localise.
COPY --from=llama-build /src/llama.cpp/build/bin/llama-bench /usr/local/bin/
COPY --from=llama-build /src/llama.cpp/build/bin/llama-cli   /usr/local/bin/

# Profileur + toutes ses deps depuis les wheels prébuildés (aucun toolchain ici).
COPY --from=py-build /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels adtc-profiler \
    && rm -rf /wheels

WORKDIR /work
ENTRYPOINT ["adtc-profiler"]
