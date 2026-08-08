# DistillRouter: Distilling a Large Language Model Routing Policy into a Lightweight Student Router

**Research report — v1**
**Last revised:** 2026-08-07

## Abstract

Model-routing systems for multi-tier LLM serving use a large model, or a large
dedicated router, to decide which downstream model should answer an incoming
query. This routing step is placed on the critical path of every request,
so its own inference cost and latency are added to the cost and latency of
the model it eventually selects. We propose DistillRouter, a study of
whether the routing policy of a large *teacher router* can be distilled into
a substantially smaller *student router* without a material loss in routing
quality. We formulate routing as a discrete tier-classification problem
supervised jointly by (i) an empirical oracle signal, obtained by executing
each candidate tier and scoring it against task-specific correctness
metrics, and (ii) a soft label distribution elicited from a large language
model prompted to predict the oracle outcome from the query text alone. This
report specifies the benchmark suite, the model family and tier selection,
the teacher-labeling protocol, the student training objective, and the
evaluation methodology used to test this hypothesis.

## 1. Introduction

### 1.1 Motivation

Production systems that serve multiple LLMs of varying capacity and cost
face a routing problem: for each incoming query, decide which model should
generate the response, trading off answer quality against latency and
inference cost. A common architecture places a large, capable model at this
decision point, since the routing decision itself benefits from the same
reasoning capacity needed to judge query difficulty (Ong et al., 2024; Lu et
al., 2023). This is effective but self-defeating in latency terms: the
router now contributes a large model's forward-pass latency to *every*
request, including the large share of requests that a much smaller model
could have answered correctly on its own.

### 1.2 Problem Statement

Can the routing policy learned by a large language model be compressed into
a significantly smaller model while preserving routing quality? We take
"routing quality" to mean agreement with the tier that would, in fact, have
answered the query correctly at the lowest cost — not merely agreement with
the teacher's own predictions, which may themselves be miscalibrated.

### 1.3 Contributions

This report specifies:

1. A benchmark suite spanning mathematical reasoning, open-domain and
   multi-hop question answering, and code generation, selected and verified
   against primary sources (Section 3).
2. A model family and tier taxonomy, with an explicit, evidence-based choice
   of which models occupy the teacher, student, and candidate-answerer
   roles, verified against the current (August 2026) state of each family's
   public releases (Section 4).
3. A teacher-labeling pipeline that separates *oracle* ground truth from
   *predicted* soft labels, avoiding the failure mode of distilling a
   teacher's miscalibration as if it were ground truth (Section 5).
4. A student distillation objective and an evaluation protocol that reports
   fidelity-to-teacher and quality-vs-oracle as distinct, non-interchangeable
   metrics (Sections 6–7).

### 1.4 Working Assumptions

The design rests on three assumptions, stated explicitly so that they can be
falsified rather than taken for granted:

**A1.** Parameter count correlates, on average and across a model
generation, with reasoning capability on complex tasks. This does not hold
for every individual query — a small model will sometimes accidentally
answer a hard question correctly, and a large one will sometimes err on an
easy one — and that per-query noise is exactly what the router has to learn
to tolerate rather than what it is expected to eliminate.

**A2.** Query difficulty is at least partially predictable from the surface
form of the input text, without first executing candidate models. This is a
necessary condition for routing to be useful at inference time at all; if
difficulty were only observable post-hoc, no router — large or small — could
act on it in advance.

**A3.** The mapping from query text to routing tier is a lower-complexity
function than the mapping from query text to a correct answer, and is
therefore learnable by a model with much less capacity than any of the
answering candidates. This is the premise that licenses shrinking the router
by orders of magnitude relative to the tiers it is choosing between.

## 2. Related Work

Several existing systems inform this design, though none combine oracle
grounding with soft-label distillation in exactly the configuration proposed
here.

**RouteLLM** (Ong et al., 2024) trains a router that binarizes the routing
decision between one strong and one weak model, learning from human
preference data augmented with additional signal, and reports over 2×
cost reduction at matched quality on public benchmarks.

**Zooter** (Lu et al., 2023) derives categorical routing labels from an
off-the-shelf reward model's scores over candidate LLM outputs — a reward
model plays a role structurally similar to the oracle proposed here — and
distills that categorical distribution into a small routing classifier via
knowledge distillation, including a tag-based label-smoothing step to
mitigate reward-model noise. This is the closest structural precedent to
the pipeline in Section 5: a distribution over route choices, derived from
a signal grounded in actual model behavior, distilled into a lightweight
classifier.

**RouterBench** (Hu et al., 2024) contributes a benchmark of over 405K
precomputed inference outcomes across eleven LLMs (both open-weight, e.g.
the Llama-2 and Mixtral families, and proprietary, e.g. GPT-4 and
Claude-2) over seven tasks including MMLU, GSM8K, and MBPP, enabling
router evaluation without re-running inference for every candidate model
at evaluation time. The precomputed-outcome pattern is directly applicable
to constructing this project's oracle labels offline, and its per-task
model of a routing benchmark is used as a structural reference here.

What distinguishes the present design from this prior work is the explicit
separation, within a single pipeline, of (a) an oracle label computed from
task-verifiable execution, from (b) a teacher's *predicted* label computed
from text alone — and the treatment of both as separate, simultaneously
logged targets, so that "does the student agree with the teacher" and "is
the student's routing actually good" can be measured independently rather
than conflated into one accuracy number.

## 3. Benchmark Suite

Dataset facts below are drawn from the original dataset papers and official
dataset cards, independent of the routing literature, and were re-verified
by directly loading each source at implementation time (Section 3.5).

### 3.1 Mathematical Reasoning

**GSM8K** (Cobbe et al., 2021) consists of 8,792 grade-school arithmetic
word problems (7,473 train / 1,319 test), each requiring 2–8 steps of
elementary arithmetic to solve. The dataset carries no explicit difficulty
annotation. Correctness is conventionally assessed by exact-match on the
final numeric answer, extracted from a `####`-delimited line in the
reference solution.

**MATH** (Hendrycks et al., 2021) consists of 12,500 competition
mathematics problems (7,500 train / 5,000 test) spanning seven subject
areas — Prealgebra, Algebra, Number Theory, Counting & Probability,
Geometry, Intermediate Algebra, and Precalculus — drawn from sources
including AMC and AIME. MATH carries an explicit five-level difficulty
annotation (1 = easiest, 5 = competition/olympiad-level), which is the only
ground-truth difficulty signal available anywhere in this benchmark suite
and should be used to sanity-check that the learned routing policy's tier
assignments correlate with a real difficulty axis, rather than only with
surface-level features of the query. Correctness is assessed by exact-match
on the content of the final `\boxed{}` expression, following normalization
and, in stricter harnesses, symbolic equivalence checking via computer
algebra (the "Minerva" evaluation protocol).

Note on mirrors: not every Hugging Face upload of MATH reproduces the
original split. `nlile/hendrycks-MATH-benchmark` was found, on direct
inspection, to have been re-split by its uploader into 12,000 train / 500
test — inconsistent with the paper. `EleutherAI/hendrycks_math` (seven
per-subject configs, summed) was verified to reproduce the canonical
7,500/5,000 split exactly and is the source this project's loader uses.

### 3.2 Open-Domain and Multi-Hop Question Answering

**TriviaQA** (Joshi et al., 2017) is a reading-comprehension and
open-domain QA dataset comprising over 650K question-answer-evidence
triples across its subsets (`rc`, `rc.web`, `rc.wikipedia`, and the
open-domain `unfiltered` variant, each with a context-free `.nocontext`
counterpart), covering roughly 95K distinct question-answer pairs. The `rc`
configuration splits as train 138K / validation 17.9K / test 17.2K; since
official test-set answers are withheld for leaderboard purposes, validation
should be used as the held-out set for offline routing evaluation.
Correctness is scored as exact-match or token-F1 against a set of accepted
answer aliases.

**HotpotQA** (Yang et al., 2018) is a multi-hop QA dataset of 112,779
examples, split into train-easy/medium/hard, dev (7,405), and test (7,405,
hidden labels) — dev serves as the held-out set. Its `distractor` setting
provides ten paragraphs per question (two gold, eight distractors), while
`fullwiki` requires retrieval over the entire Wikipedia corpus. Correctness
is measured by exact-match and F1 over the predicted answer span, plus a
separate supporting-fact F1 scoring whether the model identified the gold
evidence paragraphs.

Neither dataset carries an explicit difficulty label comparable to MATH's;
hop count (HotpotQA) and answer rarity / evidence-document count (TriviaQA)
serve only as weak sanity-check proxies, not ground truth.

### 3.3 Code Generation

**HumanEval** (Chen et al., 2021) provides 164 hand-written Python
programming problems, each with an average of 7.7 hidden unit tests,
released as a single evaluation-only set with no train/test split — it
must be reserved entirely for held-out evaluation, never for generating
teacher-labeled training data. Correctness is scored by pass@k.

**MBPP** (Austin et al., 2021) consists of 974 entry-level Python
programming problems (427 in the hand-verified "sanitized" subset), each
with roughly three assert-based unit tests. The canonical partition assigns
task IDs 601–974 to training, 11–510 to evaluation, 511–600 to validation,
and 1–10 to few-shot prompting. Correctness is scored by pass@k.

The coding slice of the training mixture is drawn from MBPP's
train/validation partitions, with HumanEval (in full) and MBPP's evaluation
partition reserved as held-out coding benchmarks.

### 3.4 Distribution Considerations and Synthetic Augmentation

**Table 1. Expected routing skew by dataset slice.**

| Domain | Dataset slice | Expected routing skew |
|---|---|---|
| Math | GSM8K | Small / Medium |
| Math | MATH levels 1–2 | Small / Medium |
| Math | MATH levels 4–5 | Medium / Large |
| Knowledge | TriviaQA (common-knowledge queries) | Ultra-Small / Small |
| Knowledge | HotpotQA (multi-hop) | Medium / Large |
| Coding | MBPP | Small / Medium |
| Coding | HumanEval | Medium / Large |

No individual benchmark alone is balanced across all tiers; a naive
training mixture (e.g. GSM8K + MBPP alone) would starve the "Large" class.
A synthetic query set, targeting 15–20% of total training volume, corrects
for this by generating queries conditioned on (a) a target domain, (b) a
target difficulty descriptor, and (c) real seed examples for stylistic
grounding. Synthetic queries are passed through the identical
oracle-labeling procedure as real data (Section 5) — they are not
hand-labeled, so no separate labeling-noise pathway is introduced.

### 3.5 Implementation Status

The two math datasets in Section 3.1 are implemented (`dataset/gsm8k.py`,
`dataset/math.py`), with raw and processed splits verified against the
source counts above. The remaining benchmarks (TriviaQA, HotpotQA,
HumanEval, MBPP) and the synthetic augmentation set are specified but not
yet implemented; the loader architecture (see project README) is designed
so each is additive — a new dataset requires one new loader module plus one
registration line, no changes to existing code.

## 4. Model Family and Tier Taxonomy

### 4.1 Tier Definitions

Four tiers are defined by parameter count.

**Table 2. Router tier taxonomy — literature convention.**

| Tier | Parameter range | Rationale |
|---|---|---|
| Ultra-Small | < 1B | Negligible latency/cost; ceiling for what "runs ahead of every request for free" can mean |
| Small | 1B–4B | Cheap, fast; the expected majority tier for simple factual/arithmetic queries |
| Medium | 7B–14B | Materially better multi-step reasoning at a moderate cost premium |
| Large | 30B–72B | Reserved for genuinely hard queries; most expensive tier, used sparingly by design |

This is the range convention used for the family survey in Table 3 below,
and matches the general routing literature. This project's deployed rosters
(Section 4.4) use compressed sub-ranges of it, at two different levels of
aggressiveness:

- A **same-family compressed range** (Ultra-Small through 14B, nothing
  above) — Table 6a.
- A **minimal-footprint range** (roughly 270M through 3B) — Table 6b, the
  project's current default — which goes further still, trading away not
  only frontier scale but, as a side effect, single-generation consistency
  too (see Section 4.3).

"Large" and "Teacher" therefore mean different absolute sizes depending on
which configuration is in play; each table below states its own sizes
explicitly rather than relying on Table 2's labels.

### 4.2 Model Catalog (verified, August 2026)

The catalog below was verified directly against first-party sources (each
family's official GitHub org, Hugging Face org page, or vendor blog) rather
than secondary write-ups, several of which were found during this research
to contain unverifiable or inconsistent figures for the most recent (2026)
releases.

**Table 3. Same-family tier coverage, by model family, as of August 2026.**

| Family | Ultra-Small (<1B) | Small (1–4B) | Medium (7–14B) | Large (30–72B) |
|---|---|---|---|---|
| **Qwen3** | 0.6B | 1.7B, 4B | 8B, 14B | 32B |
| **Qwen2.5** | 0.5B | 1.5B, 3B | 7B, 14B | 32B, 72B |
| Llama | — | 3.2: 1B, 3B | 3.1: 8B | 3.1: 70B (current-gen Llama 4 jumps straight to 109B+ MoE; no dense option exists at this tier in that generation) |
| Gemma 3 | 270M | 1B, 4B | 12B | 27B (just under the 30B floor) |
| Phi | — | Phi-4-mini 3.8B | Phi-3-Small 7B, Phi-4 14B | — (no model ≥30B exists in this family, in any generation) |
| Mistral | — | Ministral 3: 3B | Ministral 3: 8B, 14B | Mistral Large 3 (open-weight; parameter count not published) |
| DeepSeek | — | — | — | V4-Flash (~291B total, MoE) / V4-Pro (~1.6T total, MoE); no dense small/medium originals — DeepSeek's smaller checkpoints are R1-distillations onto Qwen/Llama backbones, not native architecture |

Newer Qwen releases exist beyond Qwen3 — Qwen3.5 (Feb–Mar 2026: dense
0.8B/2B/4B/9B/27B, MoE 35B-A3B/122B-A10B/397B-A17B), Qwen3.6 (Apr 2026:
27B dense, 35B-A3B MoE only), and Qwen3.8 (Aug 2026, days old at the time
of writing) — but none of them improve on Qwen3's tier coverage for this
project. Qwen3.5's largest dense model (27B) falls short of the 30B Large
floor for the same reason Gemma 3's does, and its only options at Large
scale are MoE models where *active* parameters (3B–10B) — not the
publicized total — drive real latency and cost, which would quietly
undermine the "Large = slow/expensive" premise the routing study depends
on. Qwen3.6 and Qwen3.8 do not yet have enough released sizes to fill more
than one tier each.

### 4.3 Family Selection

**Qwen3 is the only family, at the time of writing, with a model at every
tier defined in Table 2 within a single generation.** A shared tokenizer
and architecture lineage removes vocabulary/tokenizer/training-recipe
differences as confounds, so any measured accuracy gap is attributable to
parameter count and distillation quality rather than to which lab trained
which tier — this is the standard argument for a same-family configuration,
and it is why Qwen3 was the first choice explored in this study (Table 6a).

That argument, however, was traded away by a later, deliberate decision:
the project's default configuration (Table 6b) targets a specific,
much smaller size at each role (Ultra-Small candidate ≈270M, Small
≈500M, Large ≈800M, teacher 3B) for practicality/cost reasons, and no
single Qwen generation has a released model at every one of those exact
sizes. Hitting them requires drawing from **three different releases**
across two vendors — Google's Gemma 3, Alibaba's Qwen2.5, and Alibaba's
Qwen3.5 — which reintroduces the same confound the same-family argument
above exists to avoid. This is an explicit, acknowledged trade-off (size
and cost precision over architectural cleanliness), not an oversight; see
the comparison in Table 4 and the caveat under Table 6b.

**Table 4. Three configurations considered in this study.**

| | Same-family (Table 6a) | Minimal-footprint (Table 6b, current default) | Cross-family (Table 5) |
|---|---|---|---|
| Composition | Qwen3, all four tiers | Gemma 3 + Qwen2.5 + Qwen3.5, sized to hit specific small targets | Gemma 3 + Llama + Phi + Mistral, one per tier |
| Why it's cross-vendor (if it is) | It isn't — single generation throughout | Incidentally — chasing specific small sizes, not vendor diversity, forced the mix | Deliberately — vendor diversity is the entire point of this configuration |
| Rationale | Isolates parameter count and distillation quality as the explanatory variable | Minimizes compute/cost footprint of every role in the study | Tests generalization across genuinely heterogeneous architectures |
| Position in this study | Architecture-controlled reference point, useful if a clean ablation is wanted later | **Current default** — what Section 5 onward assumes unless stated otherwise | Secondary robustness check, run after a controlled result exists, reusing the same teacher-labeled query set |

The distinction in the second row matters: Table 6b's cross-vendor
composition is a *side effect* of chasing tiny, specific sizes, whereas
Table 5's is the deliberate object of that configuration. Conflating the
two would misattribute why either configuration's results look the way
they do.

No family other than Qwen3 has full literature-scale tier coverage (Table
3), so the cross-family configuration (Table 5) is necessarily assembled
tier-by-tier across vendors rather than drawn from one vendor's release:

**Table 5. Cross-family assignment (secondary configuration).**

| Tier | Model |
|---|---|
| Ultra-Small | Gemma 3 270M |
| Small | Llama 3.2 3B |
| Medium | Phi-4 (14B) |
| Large | Mistral Large 3 (or DeepSeek-V4-Flash, if a MoE large candidate is preferred) |

This is a stronger generalization test than a same-family run — four
different labs, four different architectures — but note it is compiled
from each vendor's best-fit tier rather than one coherent release, so
architecture and training-recipe differences are fully confounded with
whatever routing behavior is observed; it answers "does the distilled
policy transfer to a heterogeneous pool" without attempting to attribute
*why* any particular transfer failure occurred.

### 4.4 Role Assignment

Three distinct roles require model assignment and are not conflated: the
**candidate pool** (models that generate answers, one per tier), the
**teacher router** (the large model whose routing *decision* is
distilled — its answering ability is not the object of study), and the
**student router** (the small model trained to imitate that decision).

**Table 6a. Same-family (Qwen3) configuration — architecture-controlled reference point.**

| Role | Model | Rationale |
|---|---|---|
| Candidate — Small | Qwen3-1.7B | Low cost/latency; expected to correctly answer the majority-easy slice of the query distribution |
| Candidate — Medium | Qwen3-4B | Moderate step up in reasoning capability at a small cost premium |
| Candidate — Large | Qwen3-14B | Strongest model kept in this roster — "Large" relative to this pool, not to the wider Qwen3 lineup (which extends to 32B; see Table 2's note) |
| Teacher router | Qwen3-8B, prompted (Section 5.2) | Deliberately **not** the same weights as any answering candidate — sits between Medium (4B) and Large (14B), avoiding the self-bias risk of a tier judging its own necessity, while staying far cheaper to run over every training query than reusing the 14B tier would be |
| Student router | Qwen3-0.6B, fine-tuned as a tier classifier | Roughly 13× smaller than the teacher (8B) and 23× smaller than the largest candidate (14B) |

This roster stays inside one Qwen generation throughout — the clean,
architecture-controlled option — but every model in it is meaningfully
larger than Table 6b's. It remains available as the reference point for a
future controlled ablation; it is **not** the configuration currently
assumed by default (see Table 6b).

**Table 6b. Minimal-footprint configuration — current default.**

| Role | Model | Size | Rationale |
|---|---|---|---|
| Student router | Gemma 3 270M | ~270M | Fine-tuned as a tier classifier; shares a base checkpoint with Candidate — Small but is tuned for a different task (routing, not answering), so reuse carries none of the self-bias risk that ruled out reusing weights between the teacher and Candidate — Large |
| Candidate — Small | Gemma 3 270M | ~270M | Cheapest tier; answers the easy majority of queries |
| Candidate — Medium | Qwen2.5-0.5B | ~494M | Exact match for the ~500M target |
| Candidate — Large | Qwen3.5-0.8B | ~0.8B | Exact match for the ~800M target; strongest model in the candidate pool |
| Teacher router | Qwen2.5-3B | 3B | Bigger than **every** candidate, including Large — an improvement over Table 6a's structure, where the teacher (8B) sat between two candidate tiers rather than above all of them |

Three things are worth being explicit about with this configuration:

1. **It spans three releases across two vendors** (Google's Gemma 3,
   Alibaba's Qwen2.5, Alibaba's Qwen3.5) — chasing exact small sizes forced
   this, as no single Qwen generation ships a model at every one of these
   targets (Section 4.3). Any accuracy or latency difference observed with
   this roster is not attributable to parameter count alone the way Table
   6a's would be.
2. **The teacher now exceeds every candidate**, rather than sitting between
   two of them — arguably a better judging setup in principle, though
   untested against Table 6a's arrangement.
3. **Nothing in this roster reaches even Table 6a's 14B**, let alone the
   30B–72B literature range in Table 2 — this is the most aggressive
   compression considered in this document, chosen for minimum compute/cost
   footprint over every other consideration; Section 9 revisits whether
   that changes the qualitative result or only the absolute cost/latency
   numbers.

An optional fourth answering tier, or a distinct (rather than shared) base
checkpoint for the student, remain open configuration choices — see Section
9.

## 5. Teacher Supervision Pipeline

A central design decision in this study is the explicit separation of two
signals that are easy to conflate in a naive implementation: an *oracle*
label, grounded in actual model executions, and a *predicted* label,
elicited from the teacher model without execution. Conflating them —
training a student purely on the teacher's self-reported prediction, never
checked against any ground truth — would risk distilling the teacher's
biases and blind spots without any error-correcting signal, and would leave
no way to distinguish "the student learned the teacher faithfully" from
"the teacher's policy was good to begin with." The pipeline below, in
structure though not in labeling source, follows the precedent set by
Zooter (Lu et al., 2023), where a reward model's scores over candidate
outputs are converted into a categorical routing label and distilled into
a lightweight classifier.

### 5.1 Oracle Labeling

For each query in the training pool, the query is executed against every
candidate tier (Section 4.4), and each response is scored using the
correctness metric defined for its source benchmark (Section 3). The
oracle label is assigned as the cheapest tier that answered correctly; if
multiple tiers succeed, the smallest/cheapest is preferred; if no tier
succeeds, the label defaults to Large, as the best available option among
those tried, even though it failed.

### 5.2 Teacher Prediction Protocol

Separately, the teacher model is prompted to predict the routing label from
the query text alone, without access to any oracle execution — matching
exactly the information available to the student at inference time. The
prompt specifies the tier taxonomy, requires a structured JSON output
carrying both an arg-max label and a full probability distribution over
tiers, and is calibrated with few-shot examples drawn from the training
pool, each annotated with its oracle-derived ground-truth label so that the
teacher's stated confidence is anchored to observed outcomes rather than
unguided self-assessment.

```
SYSTEM:
You are a routing policy for a multi-tier LLM serving system. Given a user
query, decide which model tier should answer it. Do not answer the query
yourself — only decide which tier should.

Tiers available (sizes shown for the current default configuration,
Table 6b — substitute Table 6a's sizes if running the same-family
configuration instead):
- Small   (~270M params): fast, cheap; handles short factual lookups,
           single-step arithmetic, boilerplate/short code snippets.
- Medium  (~500M params): handles multi-step reasoning, moderate difficulty
           math (comparable to MATH levels 2-3), multi-hop questions with
           2 supporting facts, medium-complexity functions with edge cases.
- Large   (~800M params): handles competition-level math (MATH levels 4-5),
           multi-hop reasoning requiring synthesis across several documents,
           and algorithmically non-trivial code — the strongest tier in
           this deployment's roster, though not frontier-scale in absolute
           terms (see Table 2's note on this study's compressed ranges).

Respond with a single JSON object, no other text:
{
  "label": "Small" | "Medium" | "Large",
  "probabilities": {"Small": <float>, "Medium": <float>, "Large": <float>},
  "rationale": "<one sentence>"
}
The three probabilities must sum to 1.0 and reflect genuine confidence
across tiers, not a one-hot encoding of the chosen label.

Few-shot examples (2-3 per tier, drawn from the training pool, each paired
with its oracle-derived ground-truth label):
<example query> -> <ground-truth label>
...

USER:
<query text>
```

The `probabilities` field, rather than the arg-max label alone, is the
primary distillation target (Section 6): it preserves information about
near-boundary cases — e.g., a distribution of Small: 0.45 / Medium: 0.50 /
Large: 0.05 signals a genuinely close call that a one-hot label would
discard entirely.

### 5.3 Training Record Schema

**Table 7. Per-query training record schema.**

| Field | Description |
|---|---|
| `query` | Raw input text |
| `dataset_name` | Source benchmark and split, e.g. `gsm8k/train` |
| `oracle_label` | Cheapest tier that answered correctly (ground truth, Section 5.1) |
| `oracle_correctness` | Per-tier pass/fail from executing every candidate |
| `teacher_predicted_label` | Teacher's arg-max prediction (Section 5.2) |
| `teacher_probabilities` | Full soft-label distribution over `{Small, Medium, Large}` |
| `teacher_rationale` | One-sentence explanation string, retained for auditing, not used as a training signal |
| `per_tier_latency` | Wall-clock generation latency for each candidate's oracle execution |
| `per_tier_input_tokens`, `per_tier_output_tokens` | Token counts per candidate |
| `per_tier_inference_cost` | Derived from token counts and published per-tier pricing, or GPU-seconds if self-hosted |

### 5.4 Incremental Labeling and Checkpointing

Because the teacher-prediction step calls a live model per query, the
labeling run is designed to be interruptible and resumable rather than a
single monolithic batch job:

- Runs process a bounded batch of queries at a time (e.g. the first 500
  from a split), not the full split in one pass, so labeling quality and
  cost can be sanity-checked before committing to the remaining queries.
- Output is **appended**, not overwritten: a later run extends the labeled
  set with the next unprocessed batch rather than re-labeling or discarding
  what came before.
- Progress is **checkpointed every 50 queries**, flushed to disk rather
  than held in memory until the end, so an interruption (crash, rate
  limit, manual stop) loses at most the partial batch since the last
  checkpoint, not the entire run.
- Resumption is **id-based**: the runner determines which queries remain
  unlabeled by checking which `Example.id`s already have a record in the
  output file, rather than assuming position/order, so reruns are safe
  even across out-of-order or repeated invocations.

This machinery is specified here as a requirement on the (not yet
implemented) `teacher/` labeling runner.

## 6. Student Distillation

The student model receives only the raw query text as input — precisely
the information available at deployment time, excluding any oracle signal
or teacher rationale. Its primary training target is the teacher's soft
probability distribution (`teacher_probabilities`), optimized via a
cross-entropy or KL-divergence objective; its performance against the
oracle hard label (`oracle_label`) is tracked as a separate metric, so that
distillation fidelity and routing quality remain distinguishable throughout
training rather than being collapsed into one number.

A blended objective is used to prevent the student from purely inheriting
teacher error:

```
L = α · CE(student_output, teacher_probabilities) + (1 − α) · CE(student_output, oracle_label)
```

with α tuned as a hyperparameter rather than fixed a priori. The student is
implemented as a lightweight classification head over the tier labels.
Each training run logs: routing accuracy against the teacher, routing
accuracy against the oracle, student inference latency, and the
compression ratio (student parameter count relative to teacher parameter
count) — this last figure being the headline result the study is designed
to produce.

## 7. Evaluation Protocol

**Table 8. Evaluation metrics.**

| Metric | Definition |
|---|---|
| Routing accuracy vs. teacher | Proportion of queries where the student's label matches the teacher's — measures distillation fidelity |
| Routing accuracy vs. oracle | Proportion of queries where the student's label matches the oracle ground-truth label — measures routing quality, the metric of primary practical interest |
| Task accuracy | End-task correctness (exact-match / F1 / pass@k, per benchmark) of the response produced by whichever tier the student routed to |
| Cost saved | Aggregate cost of the tiers the student selected, relative to a fixed-Large baseline and relative to the teacher's own routing cost |
| Routing latency | The student's own forward-pass latency to emit a decision, compared against the teacher's |
| End-to-end latency | Routing latency plus the selected tier's generation latency, vs. an always-route-to-Large baseline |

Routing accuracy is reported disaggregated by dataset and by oracle label,
not only as a single aggregate figure: a router that is 95% accurate purely
because 90% of evaluation queries are trivially "Small" represents a
substantially weaker result than one that is 85% accurate with balanced
performance across all tiers.

## 8. Deployment Architecture

```
1. Input query arrives
2. Student router executes  → routing decision (single forward pass,
                                sub-tier-model latency)
3. Query is routed to the selected candidate tier
4. The candidate model generates the response
5. Task correctness (where verifiable), end-to-end latency, and cost
   are logged, feeding monitoring and periodic re-distillation
```

Step 5 is operationally significant: as candidate models are upgraded or
replaced, the oracle-optimal routing policy shifts under the student, which
was trained against a now-outdated teacher/oracle pairing. The deployment
loop should periodically re-run a sample of live or replayed traffic
through the oracle-labeling procedure (Section 5.1) and check for drift
between the deployed student's decisions and current oracle-optimal
routing.

## 9. Open Questions

1. **Student output interface.** A generative JSON interface preserves
   symmetry with the teacher's I/O format, but introduces decode-time
   latency variance a fixed classification head avoids. A classification
   head is recommended unless a specific downstream requirement favors the
   generative format.
2. **Fourth answering tier.** Whether to add an Ultra-Small candidate to the
   answering pool (Section 4.4) depends on what fraction of the deployed
   query distribution is genuinely trivial; assess empirically before
   committing the additional oracle-execution cost.
3. **Cross-family generalization run.** Time-boxed as a follow-up once the
   same-family configuration has validated the core hypothesis (Table 4),
   not pursued in parallel with it.
4. **Model catalog re-verification.** Qwen3.5/3.6/3.8 (Table 3) are moving
   targets — Qwen3.8's open-weight release was days old at the time of
   writing. Re-check tier coverage before locking in a final model choice
   if implementation is delayed materially past this revision's date.
5. **Compressed vs. frontier-scale roster.** Table 6b's default roster tops
   out at 0.8B (Table 6a, the same-family alternative, tops out at 14B) —
   both trade away frontier-scale coverage (Table 2's 30B–72B "Large"
   convention) for something cheap and practical to run end-to-end. Whether
   this understates real-world routing gains — where the cost/latency gap
   between tiers, and thus the payoff from routing correctly, is larger at
   frontier scale than it is between sub-1B models — is untested. Re-running
   with Table 6a, and eventually with Qwen3-32B reinstated as the Large
   candidate, is the natural staged way to check whether compression changes
   the qualitative result or only the absolute cost/latency numbers.
6. **Cross-vendor confound in the default configuration.** Table 6b spans
   Gemma 3, Qwen2.5, and Qwen3.5 to hit its specific size targets, which
   means any accuracy or latency effect observed with it cannot cleanly
   distinguish "parameter count" from "which vendor/generation." Table 6a
   exists specifically to isolate that variable; a result that holds under
   both configurations is a much stronger claim than one observed only under
   Table 6b.

## References

Austin, J., Odena, A., Nye, M., Bosma, M., Michalewski, H., Dohan, D., Jiang,
E., Cai, C., Terry, M., Le, Q., & Sutton, C. (2021). Program Synthesis with
Large Language Models. *arXiv:2108.07732*.

Chen, M., Tworek, J., Jun, H., et al. (2021). Evaluating Large Language
Models Trained on Code. *arXiv:2107.03374*.

Cobbe, K., Kosaraju, V., Bavarian, M., et al. (2021). Training Verifiers to
Solve Math Word Problems. *arXiv:2110.14168*.

Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S., Tang, E.,
Song, D., & Steinhardt, J. (2021). Measuring Mathematical Problem Solving
With the MATH Dataset. *arXiv:2103.03874*.

Hu, Q. J., Bieker, J., Li, X., et al. (2024). ROUTERBENCH: A Benchmark for
Multi-LLM Routing System. *arXiv:2403.12031*.

Joshi, M., Choi, E., Weld, D. S., & Zettlemoyer, L. (2017). TriviaQA: A
Large Scale Distantly Supervised Challenge Dataset for Reading
Comprehension. *arXiv:1705.03551*.

Lu, K., Yuan, H., Lin, R., Lin, J., Yuan, Z., Zhou, C., & Zhou, J. (2023).
Routing to the Expert: Efficient Reward-guided Ensemble of Large Language
Models. *arXiv:2311.08692*. (NAACL 2024.)

Ong, I., Almahairi, A., Wu, V., Chiang, W.-L., Wu, T., Gonzalez, J. E.,
Kadous, M. W., & Stoica, I. (2024). RouteLLM: Learning to Route LLMs with
Preference Data. *arXiv:2406.18665*.

Yang, Z., Qi, P., Zhang, S., Bengio, Y., Cohen, W. W., Salakhutdinov, R., &
Manning, C. D. (2018). HotpotQA: A Dataset for Diverse, Explainable
Multi-hop Question Answering. *arXiv:1809.09600*.

Model catalog facts (Section 4) are drawn directly from each vendor's
first-party sources, verified at the time of this revision: the Qwen team's
GitHub organization and Hugging Face model pages (Qwen2.5, Qwen3, Qwen3.5,
Qwen3.6); Google's Gemma 2/3/4 release announcements; Meta AI's official
Llama 4 announcement (ai.meta.com/blog/llama-4-multimodal-intelligence);
Microsoft's Phi-3/Phi-4 technical reports and Azure Phi documentation;
Mistral AI's official models overview (docs.mistral.ai/models/overview);
and DeepSeek's Hugging Face organization page. Dataset facts not otherwise
cited above are drawn from the official dataset cards for GSM8K, MATH,
TriviaQA, HotpotQA, HumanEval, and MBPP, and from direct inspection of each
source dataset's schema and split sizes via the `datasets` library.
