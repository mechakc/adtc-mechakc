# Sahelian agronomy assistant — ADTC 2026, agriculture track

An offline advisory assistant for smallholder farming in the francophone Sahel (core: Niger —
millet, sorghum, maize, groundnut, cowpea). It answers from a **committed corpus of 33 agronomic
documents** (INRAN, RECA Niger, ICRISAT, CGIAR) and returns every figure — sowing date, spacing,
seed rate, fertiliser dose, intervention threshold — with a **verbatim citation, publisher, year
and page**.

The full technical write-up, including every measurement and every caveat, is in
**[REPORT.md](REPORT.md)**. This file only says what the repository is and how to run it.

## What is actually here

| | |
|---|---|
| Model | Qwen2.5-0.5B-Instruct, **GGUF Q8_0**, llama.cpp (630 167 424 parameters, measured) |
| Retrieval | prebuilt hybrid index committed to git — 3 180 chunks, BM25 (19 877 terms) + dense float16 (3 180 × 1 024), reciprocal-rank fusion |
| Embeddings | BGE-M3 Q8_0 served by `llama-server --embedding --pooling cls` on `127.0.0.1` |
| Answering | three levels enforced **in code**, upstream of the model: sourced / unsourced-but-flagged / refused |
| Languages | French and English, French first; a question in English is answered from a French source |

Measured on the development laptop in the profiler's non-AVX image, 2 threads, `--memory=7.5g`
(the numbers below are those of the committed `submission.json`, not estimates):

```
tokens_per_second_generation  26.52        peak_rss_mb        761.49
first_token_latency_ms        11 390.99    arc_easy acc_norm  0.5933 (n=300)
```

## Reproducing it

```bash
bash download_model.sh          # 2 public GGUF files -> model/ ; idempotent, no credentials
pip install "git+https://github.com/Africa-Deep-Tech-Foundation/adtc-profiler.git"
adtc-profiler run --submission . --mode participant --output submission.json
```

Nothing in `rag/` opens a socket other than `127.0.0.1`: the index is prebuilt and committed, so
inference makes no network call. `download_model.sh` runs *before* the profiler, as the rules allow.

The exact image the measurements were taken in is pullable by digest, and is recorded in
`metadata.json` under `_runtime.docker_image`.

## Layout

```
metadata.json          submission metadata (2 test prompts, verbatim in the Devpost form)
download_model.sh      fetches both GGUF files into model/ — weights are never committed
submission.json        the profiler run we declare
REPORT.md              technical report: design, measurements, and 11 stated caveats
corpus/sources.yaml    39 sources, provenance and licence of each, read inside each PDF
corpus/txt/committed/  7 documents whose CC-BY licence was confirmed in the PDF itself
rag/                   index.py, retrieve.py, answer.py, verify_retrieve.py + the prebuilt index/
tools_corpus/_qcm.jsonl  the hand-labelled regression set, published so it can be counted
Dockerfile             the non-AVX profiler image, profiler pinned by commit SHA
```

Licence: [GPL v3](LICENSE), inherited from the ADTC submission template this repository forks.
Corpus licensing is a separate matter and is declared document by document in
[REPORT.md](REPORT.md#sources-and-licensing) and in `corpus/sources.yaml`.
