# Technical Report — Sahel Agri: Offline Agronomic Advisor (Francophone Sahel, core Niger)

**Team ID:** adtc-mechakc
**Domain:** agriculture
**Model:** Qwen2.5-0.5B-Instruct — GGUF Q8_0

> **Telemetry in this report is the final profiler run**: participant mode, image
> `sha256:e5f5d7ce461c2f9435ee11171f2693582e57fc71c1c90ecf7c2213c102e8e4cd`, repository commit
> `744a26e23168`, 2026-08-21, third of three consecutive runs. That run's JSON is committed as
> `submission.json`, and it is the only source of the figures below: each one was re-derived from
> that file by a script and printed before being written here, never copied from an earlier table.
> Figures from *other* runs do appear — the quantization arbitration of 2026-08-18 — and every one
> of them names its run where it is used.

---

## Problem

Agricultural extension in francophone Sahel — core Niger — runs on paper. The technical content
farmers need (sowing calendars, seed densities, fertiliser doses, striga and fall-armyworm control,
variety catalogues) exists as PDFs published by RECA Niger, INRAN, ICRISAT and CGIAR centres, but it
reaches the field through an extension agent carrying a phone in an area with intermittent
connectivity and intermittent electricity. The five crops that matter here are pearl millet (mil),
sorghum, maize, groundnut and cowpea (niébé).

Target user: the extension agent, and through them the smallholder farmer. The question asked in the
field is narrow and quantitative — *"quelle dose d'engrais par poquet ?"*, *"quand semer le mil à
Maradi ?"* — and a wrong number is not a neutral error: it is a season of fertiliser bought and
misapplied.

That is why this submission is not a chatbot on top of a small model. We measured what the naked
model does on exactly these questions, across all three candidate quantizations (18 generations =
3 quantizations × 6 French agronomic prompts, greedy, single seed, 2026-08-18):

- **0 correct agronomic facts out of 6.**
- **0 refusals out of 18**, although every prompt carried the explicit instruction *"si tu ne sais
  pas, dis-le"*.
- Worst measured case: the model prescribed **100 g of NPK per planting hole against 6 g** in the
  source document (×16.7).

A 0.5B model asked to be careful is not careful. The product therefore consists of the model plus a
sourced retrieval layer plus a refusal that lives in code, upstream of the model, not in a prompt
instruction. Scope of that measurement, stated honestly: n=6, one seed, greedy decoding — it
establishes that the failure mode **exists**, not its rate.

---

## Design Decisions

**Base model: Qwen2.5-0.5B-Instruct.** Selected by measurement, not by size heuristics. A bake-off
(n=50) then a duel (n=300, arc_easy, seed 42) were run inside the official non-AVX profiler image at
2 threads with `--memory=7.5g`, eliminating SmolLM2-135M, Qwen2.5-1.5B, Qwen3-1.7B, Gemma-2-2B and
Llama-3.2-3B. What decided was not a bare composite score but the crossover point: under a
throughput score normalised to the field maximum, the 0.5B wins for any TPS_max below **36.31 t/s**,
and the fastest model we ever measured inside that image is **32.18 t/s** (SmolLM2-135M) — the
crossover sits outside the observable field. Reported `params_count` is **630 167 424**, so the
declared parameter estimate is 0.6B.

**Quantization: Q8_0.** Three full profiler runs (n=300, same image, same regime, 2026-08-18):

| Quantization | arc_easy acc_norm | Generation | First token | Peak RSS |
|---|---|---|---|---|
| Q4_K_M | 61.33 % | 10.27 t/s | 41 145 ms | 587.98 MB |
| Q4_0 | 53.33 % | 39.01 t/s | 7 155 ms | 525.95 MB |
| **Q8_0 (retained)** | **59.33 %** | **23.26 t/s** | **12 843 ms** | **763.30 MB** |

All three rows are the arbitration run of 2026-08-18, and they stay that way: it is a three-way
comparison at variable quantization and *constant* regime, so rewriting one row against a later run
would break the only thing the table is for. The retained quantization was re-measured at the final
run in a different image (`llama-server` added since), and those are the figures we declare:
**26.52 t/s**, **11 391 ms**, **761.49 MB**, accuracy unchanged at **59.33 %** — see Benchmarks
below. The derived ratios in the two paragraphs that follow are therefore arbitration figures
(23.26 t/s against 10.27 and 39.01), not our declared throughput.

Q4_K_M is eliminated under both throughput regimes. Q4_0 is the fastest by a wide margin, and that
margin is worth nothing: the absolute throughput score saturates at 15 t/s, so Q4_0's extra
15.75 t/s convert to exactly zero points while its accuracy loss is real. The tie was broken by a
measurement rather than a preference — the same French spot-check gave **2 degenerate loops out of 6
for Q4_0**, against 1 partial for Q8_0 and 0 for Q4_K_M. Honest caveat, carried forward: the
6.00-point accuracy gap between Q8_0 and Q4_0 at n=300 is only **1.50 σ**, so Q8_0's telemetry
victory is decided but not statistically established.

Mechanism, corrected by that measurement: without AVX/FMA/F16C it is the nested super-block scales
of the K-quants that dominate, not memory bandwidth — Q4_0 runs **3.80×** faster than Q4_K_M at
nearly the same file size (−12.7 %). A wider quantization is also not automatically more faithful:
Q8_0 is 2.26× faster than Q4_K_M and **less** accurate on this benchmark (59.33 against 61.33).

**Retrieval: hybrid, prebuilt, committed.** 33 indexable documents → 677 pages → **3 180 chunks**
(target 700 characters, ceiling 955, chosen by measuring retrieval margins across 512/700/1100/1500,
not by convention). BM25 over 19 877 terms plus dense vectors of shape (3180, 1024) in float16,
fused by reciprocal rank fusion. Embeddings are produced by **BGE-M3 Q8_0** served through
`llama-server --embedding --pooling cls` — llama.cpp only, no sentence-transformers, as the rules
require. The index ships in the repository (14.6 MB) so the first run is fully offline.

**Answer policy: three graduated levels, enforced in code.**

1. **Sourced** — the answer exists in the corpus: verbatim citation with publisher, year and page.
2. **Flagged** — agricultural but outside the corpus: a useful answer explicitly labelled as not
   sourced, naming the document that would be needed.
3. **Refusal** — outside agriculture: refuses, and says so.

Levels 2 and 3 are decided by the retrieval function before the model is asked anything about the
substance of the question. This follows directly from the measurement above: refusal delegated to a
prompt instruction produced 0 refusals in 18 generations.

**Alternatives considered and rejected.** Fine-tuning: rejected on factual risk within the
available window — a fine-tuned 0.5B still cannot cite a page, and our failure mode of concern is
exactly the invented number. Paraphrasing corpus content instead of quoting it: rejected, because
verbatim citation is what makes the quantitative answer verifiable. Embedding through
sentence-transformers: excluded by the llama.cpp-only rule. A larger base model: excluded by the
crossover computation above.

---

## Constraints

- **Hardware target:** 8 GB RAM, integrated GPU, Ubuntu 22.04. Pure CPU inference through
  llama.cpp, no GPU acceleration.
- **Measurement regime:** every figure was produced inside the official non-AVX profiler image at
  2 threads with `--memory=7.5g`, in participant mode. Two builds of that image are involved and each
  one is named where its figures are used: the declared telemetry comes from
  `sha256:e5f5d7ce461c2f9435ee11171f2693582e57fc71c1c90ecf7c2213c102e8e4cd` (1.79 GB as reported by
  `docker images`), the digest recorded in `submission.json` and pushed to GHCR; the 2026-08-18
  quantization arbitration comes from the earlier build
  `sha256:e84444783671779437492b0f5a6915e6454d0c9c1706495860bcc2d2ba2d02e3`, which predates the
  addition of `llama-server` to the image. The absence of AVX/FMA/F16C costs roughly a factor of
  six against a native AVX2 build; we publish the slower, comparable numbers rather than the faster,
  incomparable ones.
- **Thread resolution:** `llama-bench` resolves its own thread count and the profiler never passes
  `-t`. Our environment resolves 2 threads, and a 4-vCPU cloud VM exposes 2 physical cores on the
  large majority of SKUs, so the two regimes are expected to match. Named residue: SKU families that
  expose one thread per core (GCP T2D, H3, H4D) would resolve 4 threads, which our figures cannot
  reproduce. This is documented rather than corrected, because forcing a thread count would produce
  a number the audit cannot reproduce.
- **Offline:** the retrieval index is prebuilt and committed, so no network call is needed at first
  run. Model weights are fetched by `download_model.sh` (two GGUF files, 675 710 816 bytes for the
  LLM and 634 553 760 bytes for the embedding model, both verified byte-exact), because committing
  weights is disallowed.
- **Corpus licensing:** the governing constraint on corpus growth was licensing, not availability.
  A licence is a legal act inside the document, not a statement on the publisher's website — one
  collection advertised free reuse on its site and reserved all rights on page 3 of the PDF, and was
  dropped. Every licence claim below was read inside the PDF itself.
- **Evaluation data:** no public agricultural validation set exists for this domain, so the multiple
  choice proxy used during model selection is arc_easy. That is a declared bet, not a measurement of
  agronomic accuracy. One corroboration was obtained: across the three quantizations, the order of
  degenerate French generations (2 → 1 → 0) follows the order of arc_easy exactly (53.33 → 59.33 →
  61.33). Three points, n=6, one seed — consistency of ordering, not an established correlation.

---

## Sources and licensing

The corpus was assembled from 39 candidate sources, each downloaded, size-verified, probed page by
page for an extractable text layer, and read for a licence statement in all pages. Five sources are
methodological and are not indexed; one was excluded on a measured defect (a broken font renders
34 % of its characters as fused tokens — `afatoxin` appears 325 times against `aflatoxin` 13 — and
repairing it would require guessing, which would manufacture false verbatim text). The exclusion is
recorded in `corpus/sources.yaml` with its reason rather than silently dropped.

**33 documents are indexed, under two declared regimes:**

- **7 documents under CC BY 4.0**, the licence confirmed inside the PDF itself. Their extracted text
  is redistributed in this repository (`corpus/txt/committed/`).
- **26 documents with no licence statement found anywhere in the PDF**, published by public
  institutions (RECA Niger, INRAN, ICRISAT). Their extracted text is not redistributed as text
  files; the PDFs are fetched from the publisher at setup time by `corpus/fetch_corpus.sh`. Their
  content is present in the committed retrieval index (`rag/index/chunks.jsonl`) so that answers can
  quote them verbatim with publisher, year and page. We state this plainly rather than leave it
  implicit: the committed index reproduces passages from those 26 documents, each chunk carrying its
  full provenance, and we treat the absence of a licence statement as an absence rather than as a
  prohibition.

Every chunk in the index carries a 25-field provenance header derived from the source PDF, and page
boundaries are read from the PDF structure rather than inferred from the text, so a citation is
traceable to the page. Counted over the 39 candidate sources: 25 French and 14 English documents, and
34 of the 39 are Sahel-specific rather than general tropical. Counted over the 33 indexed documents:
all 25 French sources are indexed — the six English documents left out are the five methodological
ones and the excluded review — and 21 of the 33 indexed documents are direct extension advice in
French.

---

## Benchmarks

| Metric | Value |
|---|---|
| Machine | Development workstation, Intel Core i9, WSL2 Ubuntu 22.04 exposing 2 physical cores, inside the official non-AVX profiler image |
| Model measured | Qwen2.5-0.5B-Instruct, GGUF Q8_0, 630 167 424 parameters |
| RAM at peak | 761.49 MB |
| RAM at steady state | 739.71 MB |
| Time to first token | 11 390.99 ms |
| Generation speed | 26.52 t/s |
| Accuracy proxy | arc_easy acc_norm 59.33 % (n=300, seed 42) |
| Thermal throttling | No claim made in either direction — the sensor value we were given is a measured artefact, see below |

Scores under the README formulas, recomputed from the four fields above rather than carried over:

| Score | Formula and input | Value |
|---|---|---|
| S_acc | acc_norm × 100 = 0.5933 × 100 — a declared proxy, not a measurement of agronomic accuracy | **59.33** |
| S_perf | min(26.52 / 15, 1) × 100 — the ratio is **1.7680**, so the score is capped and 11.52 t/s of measured throughput convert to zero points | **100.00** |
| S_eff | max(0, (7 − 0.76149) / 7) × 100 | **89.12** |
| S_total | 0.50 × 59.33 + 0.30 × 100.00 + 0.20 × 89.12 = 29.6650 + 30.0000 + 17.8243 | **77.49** |

S_perf and S_eff are the two scores the submission form collects. They are computed here under the
README's fixed reference of 15 t/s; the challenge's other stated regime normalises against the
highest throughput observed in the field, which is not knowable before the field closes, so the
ranking renormalises what is declared under a fixed reference.

Run identity: participant mode, non-AVX image
`sha256:e5f5d7ce461c2f9435ee11171f2693582e57fc71c1c90ecf7c2213c102e8e4cd`, pushed as
`ghcr.io/mechakc/adtc-agri@sha256:e5f5d7ce…` and recorded under that digest in `submission.json`;
repository commit `744a26e23168`; 2 threads; `--memory=7.5g`; 2026-08-21; arc_easy n=300 seed 42;
512 prompt tokens and 128 generated tokens. Reported CPU model: 13th Gen Intel(R) Core(TM)
i9-13900HX; reported available RAM inside the container: 7.3 GB; reported OS inside the container:
Debian GNU/Linux 13 (trixie) — the Ubuntu 22.04 named in the Machine row above is the host, and the
profiler measures inside the container.

**On the thermal reading — we claim no absence of throttling.** The profiler reported a
`core_temp_c_peak` of 5.0 °C, and it reported the same value at every one of the three final runs; 5 °C
is not the temperature of a die under load, and a value constant across three runs is an artefact
rather than a transient reading. The mechanism is established by reading the profiler's own thermal
module: when it finds no CPU-labelled sensor, a fallback promotes any positive non-CPU sensor into
that field, and the throttling flag is then computed as *peak temperature ≥ 85.0 °C*. Our reported
`throttled: false` is therefore arithmetic performed on 5.0, not an observation that no throttling
occurred. What we did **not** establish, and do not guess: the identity of the sensor that produced
that value — the container's sensor namespace was never enumerated. The field is left in
`submission.json` exactly as the profiler measured it, because editing a measured value would be
falsification. One figure in that block is sound and independent of the broken sensor: the 99th
percentile of CPU utilisation is **53.1** at the declared run (54.1 / 54.2 / 53.1 across the three),
which is consistent with two busy threads and is our only in-file witness of the thread regime.
Conclusion, stated so that no reader has to infer it: P_thermal is decided by the evaluation
environment's own measurement, not by our reading of this field, and we assert nothing about it.

**On the margins around these figures — two different dispersions, deliberately kept apart.** They
answer different questions and conflating them would overstate one while hiding the other, so each is
named with the definition that produced it.

*Dispersion between the three final runs of the final image.* The definition is part of the figure
here, because two correct definitions of the same spread give two different numbers; the one below,
and the one our run script prints, is (max − min) ÷ **mean**:

| Field | Range across the three runs | (max − min) ÷ mean | Coefficient of variation |
|---|---|---|---|
| Generation speed | 25.34 to 26.52 t/s | 4.52 % | 2.55 % |
| Time to first token | 11 390.99 to 12 636.76 ms | 10.50 % | 5.68 % |
| RAM at peak | 761.49 to 763.45 MB | 0.26 % | 0.14 % |
| RAM at steady state | 734.84 to 739.71 MB | 0.66 % | 0.33 % |
| Accuracy proxy | identical at all three runs, 59.33 % | — | — |

The run we declare is the third, and it carries simultaneously the highest generation speed and the
lowest first-token latency of the three — which makes it, for the reason given two paragraphs below,
the most protective of the three in the direction that actually fails. The other two runs are not
shipped as artefacts and we make no auditability claim about them; the dispersion is declared here,
in this table, and the table is the deliverable.

*Dispersion inside a single measurement.* Distinct from the above, and not additive with it: the
per-repetition dispersion that `llama-bench` itself reports for generation throughput at 2 threads has
a coefficient of variation between 7.75 % and 17.77 %. That is spread inside one measurement, where
the table is spread between measurements. A single run is a draw rather than a measurement of our
throughput, which is why three were run and why the range, not the point, is what we describe.

*How far the audit may land from what we declare.* The comparison is field by field: 15 % tolerance
on both memory fields, 25 % on both throughput fields, and the non-failure band is the ratio
audit ÷ declared *per field*. That band is asymmetric, because the value we declare is the
denominator. We state the margin in measured factors rather than as a percentage of tolerance, for a
reason we verified by walking the comparator's own arithmetic: it classifies on the raw delta and
rounds only for display, so two fields showing the same percentage on screen can receive two
different verdicts. Nothing below rests on a boundary being inclusive in either direction.

| Regime factor k | Generation speed | Time to first token | Verdict |
|---|---|---|---|
| k = 1.25 | +25.00 % pass | −20.00 % pass | passes in **both** directions |
| k = 1.3333 | +33.33 % flag | −25.00 % pass | flag |
| k = 1.50 | +50.00 % fail | −33.33 % flag | fail — and the faulty field swaps with the direction |
| k = 1.7645 | **+76.45 % fail** | **−43.33 % flag** | fail |

Read in the fast direction, where the audit resolves more threads than we do. In the slow direction
the two fields exchange roles exactly: at k = 1.50 it is first-token latency that fails and
generation speed that only flags. Located by bisection at floating-point precision, the last ratio
that still passes is exactly 1.25, the last that merely flags is 1.4999999999999998, and the first
that fails is 1.50. The two memory fields are not moved by a change of thread regime — the scale
above holds them constant at a delta of 0.00 % and they stay `pass` throughout it — and the only
variability we measured for them is the 0.26 % and 0.66 % of the first table.

Why no direction of declaration protects us: generation throughput and first-token latency come out
of the same `llama-bench` invocation and move in **opposite** directions when the execution regime
changes, so declaring the higher figure relocates the failure from one field to the other instead of
removing it. k = 1.7645 is not a hypothesis — it is
the factor we measured on this machine between two and four resolved threads — and it is why the
residue named under Constraints matters: the SKU families that expose one thread per core (GCP T2D,
H3, H4D) would resolve 4 threads where we resolve 2. We document that residue rather than correct it,
because forcing a thread count would produce a figure the audit cannot reproduce, which is precisely
the divergence we are trying to avoid. Compounding factor, stated with it rather than separately: the
7.75 % to 17.77 % of intrinsic dispersion above sits inside a 25 % tolerance band, so between a third
and a half of that band is consumed by noise before any regime effect is applied.

First-token latency remains our largest single exposure, and the table above is why: without
AVX/FMA/F16C, prompt processing does not scale down with model size the way generation does, so its
absolute value is high and its relative movement under a regime change is the largest of the four
compared fields. We report the measured value rather than a favourable one.

These are self-reported development benchmarks. Official scores are measured by the ADTC profiler on
the standard evaluation machine.

---

## Reproducibility notes

- The profiler is pinned to a specific upstream commit in our Dockerfile, fetched by SHA at build
  time, so a rebuild resolves the same profiler code rather than a moving branch head.
- The `team_id` above follows the convention `<devpost-id>-<slug>` and is taken from our Devpost
  submission URL. The participant portal named in the challenge README could not be reached through
  any documented channel — the repository issue asking for it went unanswered and the contact
  address bounced — so no portal-issued identifier exists to quote here.
- No dependency of the upstream profiler is pinned tighter than upstream pins it: the four
  tolerance-bearing telemetry fields all come from a pinned `llama-bench` build, and diverging from
  the official dependency specification would break parity with the audit environment.

---

## Known limitations

Collected in one place rather than scattered through the report, each with the figure that bounds it,
and each stating what was measured and what was not.

1. **The accuracy proxy is a declared bet.** arc_easy measures multiple-choice reasoning in English;
   it is not a measurement of agronomic correctness, and no public agricultural validation set exists
   for this domain. The single corroboration we obtained is an *order*, not a correlation: across the
   three quantizations the count of degenerate French generations (2 → 1 → 0) follows the arc_easy
   order exactly (53.33 → 59.33 → 61.33). Three points, n=6, one seed, greedy decoding — consistency
   of ordering, and nothing stronger should be read into it.

2. **No agronomic multiple-choice set was built.** The hand-labelled set that serves as a regression
   harness for retrieval holds 20 items (`tools_corpus/_qcm.jsonl`, shipped so the claim can be
   counted rather than trusted) — 13 in French, 7 in English, across all three answer levels — and no
   50-100 item agronomic quiz exists. The consequence we state rather than absorb: the retrieval
   thresholds are calibrated against a set of that size, so they are a working calibration and not a
   validated one. The cross-lingual capability we claim as our differentiator — an English question
   answered from a French source — was exercised by a single labelled item in an earlier revision of
   this set; it now carries seven, which narrows that blind spot without closing it, since a set of
   twenty items measures a direction and not a rate.

3. **The thermal field is an artefact, and we assert nothing about throttling.** The reported peak
   core temperature is 5.0 °C, identically at all three final runs, produced by a fallback that
   promotes any positive non-CPU sensor; the flag is then arithmetic on that value against 85.0 °C.
   The identity of the sensor is not established and we do not guess it. Mechanism under Benchmarks.

4. **A regime change of factor 1.7645 would fail the field-by-field comparison on its own.** That
   factor is what we measured on this machine between two and four resolved threads; it would put
   generation throughput 76.45 % above the value we declare. The two regimes are expected to match, so
   this is a residue rather than a prediction — but it is not eliminable, it is named (the SKU families
   that expose one thread per core), and forcing a thread count would manufacture the divergence
   instead of avoiding it. Full margin map, in measured factors, under Benchmarks.

5. **0 citations are available from the CC BY socle alone.** The quantitative anchors our answers
   quote at the sourced level come from the 26 documents carrying no licence statement; restricting
   the index to the 7 confirmed CC BY documents would leave the sourced level with nothing to quote.
   That is the measured price of the index regime declared under Sources, stated here as a cost rather
   than discovered later.

6. **The citation length cap is not demonstrated in general.** For documents whose licence status
   does not permit redistributing the whole document, a quoted passage is capped at 300 characters.
   On the twelve citation units labelled at the time of that measurement the longest is 259
   characters, so the cap costs nothing there — a margin of 41 characters, which is not an order of
   magnitude. It is non-binding on that set; it is not shown to be non-binding on an arbitrary
   question, and the margin is small enough that a wider set could close it.

7. **Two measured text-quality defects.** One review was excluded because a broken font renders 34 %
   of its characters as fused tokens (`afatoxin` appears 325 times against `aflatoxin` 13) and
   repairing it would require guessing between real alternatives. Separately, de-hyphenation of
   line-broken words was resolved by the corpus itself in 95 % of 834 decisions; the rule that
   recovers forms such as `septembre-octobre` leaves an estimated 7 out of 35 hyphenated forms
   carrying a parasitic hyphen. That trade is deliberate and asymmetric — a parasitic hyphen leaves
   the token findable, a wrong fusion leaves it findable by nothing.

8. **Two limits of scope in what our end-to-end checks observe.** Both are limits on what the check
   can see, not conclusions about the system, and no result artefact from either one is shipped in
   this repository, so we describe their scope and claim no outcome here. (i) *The offline check
   inside the image.* An audit hook installed in Python observes only that one Python process; it says
   nothing about the syscalls made by the embedding server, which runs as a child process. For that
   child, what is established is narrower and stated as such: its exact argument vector, recorded at
   the moment the process is spawned, and the absence of any route in the container's network
   namespace. Those two facts bound the child's reach without observing its syscalls. (ii) *The
   publication check.* A manifest that reads back is not proof that the layers download, and it is the
   layers that `docker pull` fetches. The check is therefore three stages measured separately: an
   anonymous manifest request whose `Docker-Content-Digest` is compared byte for byte against the
   digest committed in `submission.json`, anonymous requests for the config blob and for each layer
   blob, and a `docker pull` **by digest** rather than by tag.

9. **The declared environment is a development workstation, not the target laptop.** The profiler
   records the CPU it actually ran on, and ours reads i9-13900HX — not the i5 or Ryzen 5 class the
   challenge targets. WSL2 exposes that chip as 2 physical cores, which is what makes our thread
   regime match the expected audit VM, but the throughput and latency figures are this machine's,
   measured inside the official non-AVX image. They are not a prediction of the target laptop.

10. **The quantization is arbitrated on an accuracy gap of 1.50 σ.** The 6.00-point arc_easy
    difference at n=300 separating the retained quantization from the fastest one is decided, not
    statistically established, and the French spot-check that broke the tie is n=6, one seed, greedy.
    Both halves of that arbitration are reported above with their scope.

11. **The refusal gate measures retrieval distance, not domain.** A refusal is produced by comparing
    the best fused cosine against a threshold read from the index (0.4376). That is a distance test,
    so an out-of-domain question sharing vocabulary with an agronomic corpus can clear it and be
    handled at the signalled level instead of being refused. Measured on four out-of-domain French
    questions: a phone-purchase question scores 0.3734 and is refused; a request for a poem about the
    moon scores 0.4898, a paediatric drug dose 0.4754, and a herbal toothache remedy 0.4693 — all
    three clear the gate. What they receive is the signalled level, which states that nothing is
    presented as verified and shows the nearest passages with their provenance; no unsourced claim is
    made, so the outcome is honest rather than wrong. But it is not a refusal, and we do not claim the
    gate recognises a domain. n=4, one language — the same one-language blind spot named in item 2.
