# DistillRouter — A Lightweight Distilled Router

**Status:** design spec v2 (verified) · **Last updated:** 2026-08-07

---

## 1. Problem Statement

Can the routing policy learned by a large language model be compressed into a
significantly smaller model while preserving routing quality?

Modern multi-model LLM applications use a large model (or a large *router*
component) to decide which downstream model should answer an incoming query.
This is accurate but adds a large model's latency and cost to every request,
on the critical path, before the "real" answering model even starts.

**Goal:** distill the routing knowledge of a large **teacher router** into a
much smaller **student router**, keeping routing accuracy close to the
teacher's while cutting routing-decision latency by an order of magnitude or
more.

**Core assumptions** (stated explicitly, carried through the whole design):

1. Larger parameter-count models generally have more reasoning capability
   than smaller ones on complex tasks — this is *why* parameter-tier routing
   is a sensible target to imitate. (True on average across model
   generations; not guaranteed for any single query, which is precisely the
   noise the router has to learn to handle.)
2. Query difficulty is predictable from the input text alone, without
   executing multiple candidate models first — otherwise routing at
   inference time is impossible.
3. A lightweight student model can approximate a large teacher's routing
   function through supervised distillation, because routing is a much
   lower-complexity function (a 3–4-way classification) than actually
   answering the query.

---

## 2. Offline Phase (Training)

### 2.1 Benchmark Datasets

All figures below are pulled from the official dataset cards / papers, not
the routing literature — verify against the source links if you pin exact
numbers into a paper.

#### Math

| | **GSM8K** | **MATH** (Hendrycks et al.) |
|---|---|---|
| Splits | train 7,473 / test 1,319 (8,792 total) | train 7,500 / test 5,000 (12,500 total) |
| Difficulty | 2–8 reasoning steps per problem, grade-school arithmetic, no explicit difficulty label | 5 explicit difficulty levels (1=easiest → 5=competition/olympiad), across 7 subject areas (Prealgebra, Algebra, Number Theory, Counting & Probability, Geometry, Intermediate Algebra, Precalculus) |
| Correctness metric | Exact-match on the final numeric answer, conventionally extracted after a `####` delimiter in the reference solution (not `<answer>` — that was a mislabel in the earlier draft; use `####` or an explicit `Answer:` line in your own prompt template, then regex out the trailing number) | Exact-match on the content of the final `\boxed{}`, after symbolic normalization (LaTeX → canonical form, `\frac{1}{2}` vs `0.5` vs `1/2` treated as equal, units/thousands-separators stripped), then compared as strings or via SymPy symbolic equivalence (the "Minerva" harness approach) |
| Source | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) | [hendrycks-MATH-benchmark](https://huggingface.co/datasets/nlile/hendrycks-MATH-benchmark) |

**Design note:** MATH's level 1–5 tag is the single cleanest ground-truth
"difficulty" signal in the whole benchmark suite — use it directly as a
sanity check that your teacher's routing labels correlate with something
real (e.g., level 1–2 → mostly routed Small, level 4–5 → mostly routed
Large).

#### General Knowledge / Multi-hop QA

| | **TriviaQA** | **HotpotQA** |
|---|---|---|
| Subsets | `rc` (reading-comprehension, evidence docs attached), `rc.web`, `rc.wikipedia`, `unfiltered` (open-domain, no guaranteed answer-bearing doc); each has a `.nocontext` variant that strips evidence down to bare Q/A pairs | `distractor` (10 paragraphs: 2 gold + 8 distractors), `fullwiki` (retrieval over the full Wikipedia dump) |
| Splits | `rc`: train 138k / validation 17.9k / test 17.2k. `unfiltered`: ~110k QA pairs total (train/validation/test are a subsplit of that); use **validation** as your held-out offline-eval set since `test` answers are hidden for the official leaderboard | train 90,447 / dev 7,405 / test 7,405 (test labels hidden — use dev as your held-out set) |
| Correctness metric | Exact-match / token-F1 against a **set of answer aliases** (TriviaQA ships multiple accepted surface forms per answer) | EM and F1 on the predicted answer span, **plus** a separate supporting-fact F1 that scores whether the model cited the correct 2 gold paragraphs — useful as an auxiliary "did it actually reason across hops" signal |
| Total size | ~650K question-answer-evidence triples across all subsets/splits; ~95K distinct question-answer pairs, ~6 evidence docs per question on average | 112,779 examples total across train-easy/medium/hard + dev + test splits |
| Source | [mandarjoshi/trivia_qa](https://huggingface.co/datasets/mandarjoshi/trivia_qa) | [hotpotqa/hotpot_qa](https://huggingface.co/datasets/hotpotqa/hotpot_qa) |

**Design note:** neither dataset carries an explicit difficulty label like
MATH does. Use **number of reasoning hops** as a proxy for HotpotQA
(single-hop-shaped questions vs. genuine 2-hop) and **answer rarity /
document count** as a weak proxy for TriviaQA (common-knowledge trivia vs.
long-tail facts needing the full evidence set). Don't treat these as ground
truth difficulty — they're only useful for *sanity-checking* the teacher's
labels, the same way MATH's levels are.

#### Coding

| | **HumanEval** | **MBPP** |
|---|---|---|
| Splits | Single set, 164 problems, no train/test split — used entirely for eval | Full: train 601–974 (374) / test 11–510 (500) / validation 511–600 (90) / few-shot prompt 1–10 (10). Sanitized subset: 427 hand-verified problems (most commonly reported number today) |
| Correctness metric | pass@k — generated code executed against hidden unit tests, ~7.7 tests/problem on average | pass@k — executed against ~3 assert-based unit tests per problem |
| Total size | 164 problems | 974 (full) / 427 (sanitized) |
| Source | [openai/openai_humaneval](https://huggingface.co/datasets/openai/openai_humaneval) | [google-research/mbpp](https://github.com/google-research/google-research/tree/master/mbpp) |

**Design note:** HumanEval has no train split at all — you cannot use it to
generate teacher-labeled *training* data without contaminating your own
eval set. Use MBPP's train/validation splits (or the sanitized-427 train
portion) to build the coding slice of the routing-training set, and reserve
**HumanEval in full + MBPP test** as held-out coding eval only.

#### Consolidated dataset-to-tier expectation

| Domain | Dataset slice | Expected routing skew |
|---|---|---|
| Math | GSM8K | Skews Small/Medium — most problems solvable by small models |
| Math | MATH levels 1–2 | Skews Small/Medium |
| Math | MATH levels 4–5 | Skews Medium/Large |
| Knowledge | TriviaQA (common facts) | Skews Ultra-Small/Small |
| Knowledge | HotpotQA (multi-hop) | Skews Medium/Large |
| Coding | MBPP | Skews Small/Medium |
| Coding | HumanEval | Skews Medium/Large |

This table matters operationally: if you only train on GSM8K + MBPP, your
router will almost never learn to emit "Large," and the student will
undersample that class. Deliberately balance the training mixture across
rows so all four router labels appear with reasonable frequency.

### 2.2 Synthetic Dataset

The benchmarks above are real but skewed (see table above) and static (finite,
memorizable, eventually leak into model training corpora). Add a synthetic
slice for two reasons:

1. **Class balance** — deliberately generate queries targeting the
   under-represented tiers (e.g., synthetic ultra-easy factual lookups for
   "Ultra-Small," synthetic multi-constraint word problems or obscure
   multi-hop trivia for "Large") so the student sees enough examples of
   every label to avoid majority-class collapse.
2. **Distribution robustness** — paraphrase/perturb real benchmark questions
   (reword surface form, swap named entities, combine two GSM8K-style
   sub-problems into one multi-step problem) so the router doesn't just
   memorize surface patterns tied to a specific benchmark's phrasing style.

**Generation method:** use the teacher-tier LLM itself (or a separate strong
generator model) to produce new queries conditioned on: (a) a target domain
(math/knowledge/code), (b) a target difficulty descriptor, and (c) a few
real seed examples from that domain/difficulty for style-grounding. Every
synthetic query still goes through the same oracle-labeling pipeline
(Section 2.4) — it is **not** hand-labeled, so it can't leak label noise
that the oracle process wouldn't already have.

Target size: roughly 15–20% of the total training set, capped so the
majority of supervision still comes from real, externally-validated
benchmarks.

### 2.3 Model Roster

Three *different* roles need models, and they should not be conflated:

1. **Candidate pool** — the models actually being routed to, one per tier,
   which generate the final answer.
2. **Teacher router** — the large model whose *routing decision* (not
   answering ability) is being distilled.
3. **Student router** — the small model being trained to imitate the
   teacher's routing decision.

#### Tier definitions (verified parameter counts)

| Router Label | Parameter Range | Verified example models |
|---|---|---|
| Ultra Small | < 1B | Qwen2.5-0.5B, Qwen3-0.6B, Gemma 3 270M |
| Small | 1B–4B | Llama 3.2 1B/3B, Qwen2.5 1.5B/3B, Qwen3 1.7B/4B, Gemma 3 1B/4B, Phi-3-Mini/Phi-4-Mini 3.8B |
| Medium | 7B–14B | Qwen3-8B, Qwen2.5-14B, Gemma 2 9B, Gemma 3 12B, Phi-3-Small 7B, Phi-3-Medium/Phi-4 14B |
| Large | 30B–72B | Qwen2.5-32B, Qwen2.5-72B, Llama 3.1-70B |

Corrections from the original draft:
- The GSM8K delimiter is `####`, not `<answer>`.
- "Gemma 4" is real (Google shipped it April 2026: E2B/E4B/12B-dense/26B-MoE/31B-dense), but its sizes don't map cleanly onto this tier table and tooling/eval support is still thin months after release — **use Gemma 3** (270M/1B/4B/12B/27B) for reproducibility; revisit Gemma 4 once the ecosystem catches up.
- Gemma 2 tops out at 9B/27B (no 12B) — the "Gemma 3 12B" and "Gemma 2 9B" entries in the Medium row are both correct and distinct models; don't merge them.

#### Family choice: Option A (same-family) recommended as primary

| | Option A — Same Family | Option B — Cross Family |
|---|---|---|
| Pick | **Qwen** (Qwen2.5 + Qwen3 generations) | Qwen, Llama, Gemma, Phi, Mistral, DeepSeek |
| Why it's primary | Qwen is the **only** family with a released model at every tier — 0.5B/0.6B, 1.5B–4B, 7B–14B, up to 72B — on a shared tokenizer and architecture lineage. That removes vocabulary/tokenizer/instruction-tuning-style as confounding variables, so any accuracy gap you measure is attributable to *parameter count and distillation quality*, not to "which lab trained it." | Matches production reality — real routing stacks front heterogeneous vendors. Confounds architecture/training-recipe differences with the difficulty signal, making it harder to isolate whether a routing error is a genuine difficulty misjudgment or a family-specific quirk. |
| Role | **Run this first.** It's the controlled experiment that validates the core hypothesis (Section 1). | **Run second**, as a generalization/robustness check once Option A's pipeline is validated — reusing the same teacher-labeled queries, just swapping which model answers at each tier. |

#### Recommended model assignment (Option A / Qwen)

| Role | Model | Rationale |
|---|---|---|
| Candidate — Small | Qwen2.5-1.5B-Instruct (or Qwen3-1.7B) | Cheap, fast, answers the easy majority of queries |
| Candidate — Medium | Qwen3-8B (or Qwen2.5-14B-Instruct) | Balances cost against materially better reasoning |
| Candidate — Large | Qwen2.5-72B-Instruct (Qwen2.5-32B as a lower-compute fallback) | Strongest available same-family reasoner; used both as the top answering tier and, per below, doubling as the teacher |
| **Teacher router** | Qwen2.5-72B-Instruct, prompted (Section 2.4) — same weights as the Large candidate | Reuses the model you're already serving at the top tier instead of standing up a fourth, separate frontier model purely for labeling; keeps the whole pipeline single-family and reproducible without an external API dependency. Trade-off: a model that is simultaneously "the answerer" and "the labeler" for its own tier could be mildly optimistic about when Large is needed — this is why routing labels are grounded empirically (Section 2.4) rather than trusted purely from the LLM's own self-assessment. |
| **Student router** | Qwen2.5-0.5B or Qwen3-0.6B, fine-tuned as a classifier | The actual deliverable: >100x smaller than the teacher, cheap enough to run on every request before any candidate model is invoked |

If budget allows, an Ultra-Small *candidate* (Qwen2.5-0.5B) can also be added
as a 4th answering tier, giving the router 4 output classes instead of 3 —
optional, since it roughly doubles oracle-execution cost for a class that
will absorb only the easiest slice of queries.

### 2.4 Teacher Labeling Pipeline

Two things are easy to conflate and must be kept separate:

- **Oracle ground truth** — derived by *actually running* the query on the
  Small/Medium/Large candidates and scoring each with the benchmark's own
  correctness metric (Section 2.1). The ground-truth label is the
  **cheapest tier that answers correctly** (if several tiers succeed, prefer
  the smallest/cheapest; if none succeed, label Large — it's the best
  available option even though it failed).
- **Teacher-predicted label** — the large LLM's *prediction* of that same
  label, made from the query text alone, with no access to the oracle
  executions. This is what the student is actually distilled from, because
  at inference time neither the teacher nor the student gets to run three
  candidate models before deciding — that would defeat the entire point of
  routing.

Training a classifier on the LLM's self-reported prediction alone (never
checked against reality) risks distilling the teacher's biases and
blind spots with no error correction. Grounding every predicted label
against an oracle execution — the way the pipeline below does — keeps the
teacher's calibration honest and gives you a second metric ("teacher
routing accuracy") to report, distinct from "student fidelity to teacher."
This mirrors the approach used by prior routing-distillation systems in the
literature (e.g., Zooter derives categorical routing labels from a reward
model, then distills that categorical distribution into a small BERT-sized
classifier via KD — same shape of pipeline, different labeling source).

#### Teacher prompt design

```
SYSTEM:
You are a routing policy for a multi-tier LLM serving system. Given a user
query, decide which model tier should answer it. Do not answer the query
yourself — only decide which tier should.

Tiers available:
- Small   (1-4B params):  fast, cheap; handles short factual lookups,
           single-step arithmetic, boilerplate/short code snippets.
- Medium  (7-14B params): handles multi-step reasoning, moderate difficulty
           math (comparable to MATH levels 2-3), multi-hop questions with
           2 supporting facts, medium-complexity functions with edge cases.
- Large   (30-72B params): handles competition-level math (MATH levels 4-5),
           multi-hop reasoning requiring synthesis across several documents,
           and algorithmically non-trivial code.

Respond with a single JSON object, no other text:
{
  "label": "Small" | "Medium" | "Large",
  "probabilities": {"Small": <float>, "Medium": <float>, "Large": <float>},
  "rationale": "<one sentence>"
}
The three probabilities must sum to 1.0 and reflect your genuine confidence
distribution across tiers, not just a one-hot on your chosen label.

Few-shot examples (2-3 per tier, drawn from the training pool, each shown
with its oracle-derived ground-truth label so the teacher's stated
confidence is calibrated against real outcomes rather than vibes):
<example query> -> <ground-truth label>
...

USER:
<query text>
```

The `probabilities` field is the actual distillation target for
cross-entropy / KL-based soft-label training — not just the arg-max label —
since it carries more signal about near-boundary cases (e.g., Small: 0.45 /
Medium: 0.5 / Large: 0.05 tells the student this was a genuinely close call,
which a one-hot label would erase).

#### Training record schema

Store one row per query:

| Field | Description |
|---|---|
| `query` | Raw input text |
| `dataset_name` | Source benchmark + split (e.g. `gsm8k/train`) |
| `oracle_label` | Cheapest tier that answered correctly (ground truth) |
| `oracle_correctness` | Per-tier pass/fail from actually running Small/Medium/Large |
| `teacher_predicted_label` | Teacher's arg-max label from the prompt above |
| `teacher_probabilities` | Full soft-label distribution `{Small, Medium, Large}` |
| `teacher_rationale` | One-line explanation string (for debugging/auditing, not used as a training signal) |
| `per_tier_latency` | Wall-clock generation latency for each candidate's oracle run |
| `per_tier_input_tokens` / `per_tier_output_tokens` | Token counts per candidate |
| `per_tier_inference_cost` | Derived from tokens × published per-tier $/token, or GPU-seconds if self-hosted |

### 2.5 Student Distillation

- **Inputs:** `query` text only (exactly what's available at real inference
  time — no oracle signals, no teacher rationale).
- **Targets:** primarily `teacher_probabilities` (soft-label KL/
  cross-entropy against the full distribution); also report metrics against
  `oracle_label` (hard label) so you can separate "did the student learn the
  teacher" from "is the teacher's policy actually any good."
- **Architecture:** fine-tune the ultra-small model with a classification
  head (or constrained-decoding JSON output, if you'd rather keep it
  generative for consistency with the teacher's I/O shape) over the 3
  tier labels.
- **Loss:** cross-entropy against soft teacher labels (optionally a blended
  loss: `α · CE(student, teacher_probs) + (1-α) · CE(student, oracle_label)`
  to prevent the student from inheriting purely the teacher's mistakes).
- **Logged per training run:** student routing accuracy vs. teacher, student
  routing accuracy vs. oracle, student inference latency, student parameter
  count vs. teacher parameter count (the compression ratio being reported).

### 2.6 Evaluation (offline, held-out test splits)

| Metric | Definition |
|---|---|
| Routing accuracy (vs. teacher) | student label == teacher label, over total queries — measures **distillation fidelity** |
| Routing accuracy (vs. oracle) | student label == oracle ground-truth label — measures **routing quality**, the metric that actually matters end-to-end |
| Task accuracy | correctness of the final answer produced by *whichever tier the student routed to*, per the benchmark's own metric (exact-match / F1 / pass@k) — this is the metric a user of the system actually feels |
| Cost saved | Σ cost of tier the student picked vs. Σ cost if every query went to Large (or vs. teacher's own routing cost, including the teacher's own inference cost) |
| Latency (routing decision only) | student's own forward-pass latency to emit a routing decision, compared against the teacher's — this is the number that justifies the whole project |
| Latency (end-to-end) | routing decision latency + the chosen tier's generation latency, vs. always-route-to-Large as a baseline |

Report routing accuracy **broken down by dataset and by oracle label**, not
just as one aggregate number — a router that's 95% accurate but only because
90% of queries are trivially "Small" is a much weaker result than one that's
85% accurate with balanced performance across all three labels.

---

## 3. Deployment Phase (Inference)

```
1. Input query arrives
2. Student router executes  → routing decision (single forward pass,
                                sub-tier-model latency)
3. Route to the chosen candidate model tier
4. Candidate model generates the response
5. Log: task correctness (if verifiable), end-to-end latency, cost
        — feed back into monitoring / periodic re-distillation
```

Step 5 matters operationally: routing policies drift as the underlying
candidate models are upgraded, so the deployment loop should periodically
re-run a slice of live (or replayed) traffic through the oracle pipeline
and check the student hasn't drifted from what oracle-optimal routing now
looks like.

---

## 4. Open Follow-ups (not yet decided — flag before implementation)

- **Exact JSON-vs-classifier-head interface for the student** — generative
  JSON keeps I/O symmetric with the teacher (easier to compare probability
  distributions apples-to-apples) but a plain classification head is faster
  and removes decode-time variance; pick classification head unless you have
  a specific reason to keep the student generative.
- **Whether to add a 4th "Ultra-Small" answering tier** to the candidate
  pool (see Section 2.3) — increases oracle-labeling cost, only worth it if
  a large fraction of your query mix is genuinely trivial.
- **Cross-family run (Option B)** — timebox it as a follow-up once Option A
  is validated; don't build both in parallel.

---

### Sources consulted

- [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k)
- [hendrycks-MATH-benchmark](https://huggingface.co/datasets/nlile/hendrycks-MATH-benchmark)
- [mandarjoshi/trivia_qa](https://huggingface.co/datasets/mandarjoshi/trivia_qa)
- [hotpotqa/hotpot_qa](https://huggingface.co/datasets/hotpotqa/hotpot_qa)
- [openai/openai_humaneval](https://huggingface.co/datasets/openai/openai_humaneval)
- [google-research/mbpp](https://github.com/google-research/google-research/tree/master/mbpp)
- [Qwen2.5 Technical Report](https://arxiv.org/pdf/2412.15115) / [Qwen2.5 collection](https://huggingface.co/collections/Qwen/qwen25)
- [Qwen3 blog](https://qwenlm.github.io/blog/qwen3/) / [Qwen3 Technical Report](https://arxiv.org/pdf/2505.09388)
- [Gemma 3 launch (Hugging Face blog)](https://huggingface.co/blog/gemma3) / [google/gemma-3-270m](https://huggingface.co/google/gemma-3-270m)
- [Gemma 2 launch (Google blog)](https://blog.google/innovation-and-ai/technology/developers-tools/google-gemma-2/)
- [Gemma 4 launch (Google blog)](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/)
- [Llama 3.2 (AWS Bedrock announcement)](https://aws.amazon.com/blogs/aws/introducing-llama-3-2-models-from-meta-in-amazon-bedrock-a-new-generation-of-multimodal-vision-and-lightweight-models/) / [Llama 3.1 (Hugging Face blog)](https://huggingface.co/blog/llama31)
- [Phi-3 Technical Report](https://arxiv.org/pdf/2404.14219) / [Phi-4 Technical Report](https://www.microsoft.com/en-us/research/wp-content/uploads/2024/12/P4TechReport.pdf)
- [ROUTERBENCH](https://arxiv.org/pdf/2403.12031) — multi-LLM routing benchmark, precomputed inference outcomes
- [Awesome-Routing-LLMs](https://github.com/MilkThink-Lab/Awesome-Routing-LLMs) — includes Zooter (reward-guided categorical routing label + KD into a small classifier), the closest prior-art shape to this design
- [Minerva/MATH evaluation harness (lm-evaluation-harness)](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/minerva_math/README.md)
