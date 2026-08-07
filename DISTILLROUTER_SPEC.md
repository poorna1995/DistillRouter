# DistillRouter: Distilling a Large Language Model Routing Policy into a Lightweight Student Router

### 1.2 Problem Statement

Can the routing policy learned by a large language model be compressed into
a significantly smaller model while preserving routing quality? We take
"routing quality" to mean agreement with the tier that would, in fact, have
answered the query correctly at the lowest cost — not merely agreement with
the teacher's own predictions, which may themselves be miscalibrated.

## 3. Benchmark Suite

Dataset facts below are drawn from the original dataset papers and official
dataset cards, independent of the routing literature, and cross-checked
against each source directly.

### 3.1 Mathematical Reasoning

**GSM8K** (Cobbe et al., 2021) consists of 8,792 grade-school arithmetic
word problems (7,473 train / 1,319 test), each requiring 2–8 steps of
elementary arithmetic to solve. The dataset carries no explicit difficulty
annotation. Correctness is conventionally assessed by exact-match on the
final numeric answer, extracted from a `####`-delimited line in the
reference solution (not an `<answer>` tag, as an earlier draft of this
document stated); a custom evaluation prompt may substitute any consistent
delimiter, provided the extraction regex is applied uniformly across models.

**MATH** (Hendrycks et al., 2021) consists of 12,500 competition
mathematics problems (7,500 train / 5,000 test) spanning seven subject
areas — Prealgebra, Algebra, Number Theory, Counting & Probability,
Geometry, Intermediate Algebra, and Precalculus — drawn from sources
including AMC and AIME. Critically, MATH carries an explicit five-level
difficulty annotation (1 = easiest, 5 = competition/olympiad-level), which
is the only ground-truth difficulty signal available anywhere in this
benchmark suite and should be used to sanity-check that the learned routing
policy's tier assignments correlate with a real difficulty axis, rather
than only with surface-level features of the query. Correctness is assessed
by exact-match on the content of the final `\boxed{}` expression, following
normalization (LaTeX canonicalization, equivalence of `\frac{1}{2}`, `0.5`,
and `1/2`, removal of units and thousands separators) and, in stricter
harnesses, symbolic equivalence checking via computer algebra (the
"Minerva" evaluation protocol).

### 3.2 Open-Domain and Multi-Hop Question Answering

**TriviaQA** (Joshi et al., 2017) is a reading-comprehension and
open-domain QA dataset comprising over 650K question-answer-evidence
triples across its subsets (`rc`, `rc.web`, `rc.wikipedia`, and the
open-domain `unfiltered` variant, each with a context-free `.nocontext`
counterpart), covering roughly 95K distinct question-answer pairs with
approximately six evidence documents per question on average. The `rc`
configuration splits as train 138K / validation 17.9K / test 17.2K; because
official test-set answers are withheld for leaderboard purposes, the
validation split should be used as the held-out set for offline routing
evaluation. Correctness is scored as exact-match or token-F1 against a set
of accepted answer aliases, since a single fact typically has multiple
valid surface realizations.

**HotpotQA** (Yang et al., 2018) is a multi-hop QA dataset of 112,779
examples in total, split into train-easy/medium/hard, dev (7,405), and
test (7,405, with hidden labels) — dev should serve as the held-out set.
Its `distractor` setting provides ten paragraphs per question (two gold,
eight distractors), while `fullwiki` requires retrieval over the entire
Wikipedia corpus. Correctness is measured by exact-match and F1 over the
predicted answer span, together with a separate supporting-fact F1 that
scores whether the model correctly identified the gold evidence paragraphs
— a useful auxiliary signal for whether genuine multi-hop reasoning
occurred, as opposed to a lucky single-hop guess.

Neither TriviaQA nor HotpotQA carries an explicit difficulty label
comparable to MATH's. As weak difficulty proxies, this design uses hop
count (single- vs. multi-hop question structure) for HotpotQA and answer
rarity / evidence-document count for TriviaQA — these are heuristics for
sanity-checking learned routing behavior, not substitutes for ground truth.

### 3.3 Code Generation

**HumanEval** (Chen et al., 2021) provides 164 hand-written Python
programming problems, each with a function signature, docstring, and an
average of 7.7 hidden unit tests, released as a single evaluation-only set
with no train/test split. Because no training partition exists, HumanEval
cannot be used to generate teacher-labeled training data without
contaminating it as an evaluation set; it must be reserved entirely for
held-out evaluation. Correctness is scored by pass@k over the hidden test
suite.

**MBPP** (Austin et al., 2021) consists of 974 entry-level Python
programming problems, each with roughly three assert-based unit tests, of
which a 427-problem "sanitized" subset has been hand-verified for solution
and test correctness and is the more commonly reported configuration in
recent literature. The full set's canonical partition assigns task IDs
601–974 to training, 11–510 to evaluation, 511–600 to validation, and 1–10
to few-shot prompting. Correctness is scored by pass@k against the
associated unit tests.

The coding slice of the training mixture is therefore drawn from MBPP's
train/validation partitions, with HumanEval (in full) and MBPP's evaluation
partition reserved as held-out coding benchmarks.

### 3.4 Distribution Considerations and Synthetic Augmentation

Table 1 summarizes the expected routing-label skew of each benchmark slice,
based on the difficulty characteristics above.

**Table 1. Expected routing skew by dataset slice.**

| Domain    | Dataset slice                       | Expected routing skew |
| --------- | ----------------------------------- | --------------------- |
| Math      | GSM8K                               | Small / Medium        |
| Math      | MATH levels 1–2                     | Small / Medium        |
| Math      | MATH levels 4–5                     | Medium / Large        |
| Knowledge | TriviaQA (common-knowledge queries) | Ultra-Small / Small   |
| Knowledge | HotpotQA (multi-hop)                | Medium / Large        |
| Coding    | MBPP                                | Small / Medium        |
| Coding    | HumanEval                           | Medium / Large        |

Because no individual benchmark alone is balanced across all tiers, a
training mixture drawn naively from these sources (e.g., GSM8K and MBPP
alone) would under-represent the "Large" tier and bias the student toward
never emitting it. To correct for this, a synthetic query set is
introduced, constituting approximately 15–20% of total training volume,
generated by conditioning a strong generator model on (a) a target domain,
(b) a target difficulty descriptor, and (c) a small number of real seed
examples for stylistic grounding. Synthetic queries are not hand-labeled;
they are passed through the identical oracle-labeling procedure described
in Section 5, so no additional labeling-noise pathway is introduced beyond
what already exists for real benchmark data. The purpose of the synthetic
set is twofold: correcting class imbalance across routing tiers, and
reducing the risk that the router learns benchmark-specific surface
patterns (a fixed phrasing style, a specific entity distribution) rather
than a generalizable difficulty signal.

## 4. Model Family and Tier Taxonomy

### 4.1 Tier Definitions

Four tiers are defined by parameter count, following common usage in the
routing literature. Table 2 lists representative models per tier,
cross-checked against each family's official release material.

**Table 2. Router tier taxonomy with verified example models.**

| Tier        | Parameter range | Verified example models                                                                        |
| ----------- | --------------- | ---------------------------------------------------------------------------------------------- |
| Ultra-Small | < 1B            | Qwen2.5-0.5B, Qwen3-0.6B, Gemma 3 270M                                                         |
| Small       | 1B–4B           | Llama 3.2 1B/3B, Qwen2.5 1.5B/3B, Qwen3 1.7B/4B, Gemma 3 1B/4B, Phi-3-Mini / Phi-4-Mini (3.8B) |
| Medium      | 7B–14B          | Qwen3-8B, Qwen2.5-14B, Gemma 2 9B, Gemma 3 12B, Phi-3-Small (7B), Phi-3-Medium / Phi-4 (14B)   |
| Large       | 30B–72B         | Qwen2.5-32B, Qwen2.5-72B, Llama 3.1-70B                                                        |

### 4.2 Family Selection

Two configurations are considered: models drawn from a single family
across all tiers, versus models drawn from multiple families.

**Table 3. Family selection: same-family vs. cross-family.**

|                        | Same-family (primary)                                                                                                                                                                                                                                                                                                                                                                                                                                                            | Cross-family (secondary)                                                                                                                                                                                                                                                                                                                                        |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Composition            | Qwen (Qwen2.5 and Qwen3 generations)                                                                                                                                                                                                                                                                                                                                                                                                                                             | Qwen, Llama, Gemma, Phi, Mistral, DeepSeek                                                                                                                                                                                                                                                                                                                      |
| Rationale              | Qwen is, at the time of writing, the only family with a released model at every tier defined in Table 2 — from 0.5B/0.6B through 72B — sharing a tokenizer and architectural lineage across generations. This removes vocabulary, tokenizer, and instruction-tuning style as confounds, so any measured routing-accuracy or task-accuracy gap can be attributed to parameter count and distillation quality rather than to idiosyncrasies of a particular lab's training recipe. | More representative of production routing stacks, which typically front heterogeneous vendors and architectures. However, it confounds architecture- and training-recipe differences with the difficulty signal the router is meant to learn, making it harder to attribute a routing error to a genuine difficulty misjudgment versus a family-specific quirk. |
| Position in this study | Primary configuration; validates the central hypothesis (Section 1.2) under controlled conditions.                                                                                                                                                                                                                                                                                                                                                                               | Secondary robustness/generalization check, run after the primary configuration is validated, reusing the same teacher-labeled query set and substituting which model answers at each tier.                                                                                                                                                                      |

### 4.3 Role Assignment

Three distinct roles require model assignment and should not be conflated:
the **candidate pool** (models that actually generate answers, one per
tier), the **teacher router** (the large model whose routing _decision_ is
being distilled — its answering ability is not the object of study), and
the **student router** (the small model being trained to imitate that
decision).

**Table 4. Recommended model assignment under the same-family (Qwen) configuration.**

| Role               | Model                                                                                                            | Rationale                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| ------------------ | ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Candidate — Small  | Qwen2.5-1.5B-Instruct (alt.: Qwen3-1.7B)                                                                         | Low cost and latency; expected to correctly answer the majority-easy slice of the query distribution.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| Candidate — Medium | Qwen3-8B (alt.: Qwen2.5-14B-Instruct)                                                                            | Intermediate cost/capability point for multi-step reasoning.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| Candidate — Large  | Qwen2.5-72B-Instruct (Qwen2.5-32B as a lower-compute fallback)                                                   | Strongest available same-family reasoning model; serves simultaneously as the top answering tier and as the teacher (below).                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| Teacher router     | Qwen2.5-72B-Instruct, used only for routing prediction (Section 5.2), not as an answering candidate in this role | Reuses a model already deployed at the top serving tier rather than introducing a fourth, external frontier model solely for labeling, preserving single-family provenance and avoiding an external API dependency. This choice carries a trade-off: a model that is simultaneously the top-tier answerer and the routing labeler may be systematically biased toward over-predicting when "Large" is required. This risk is the direct motivation for grounding every predicted label against an independent oracle execution (Section 5.1), rather than trusting the teacher's self-assessment alone. |
| Student router     | Qwen2.5-0.5B or Qwen3-0.6B, fine-tuned as a tier classifier                                                      | The object of this study: more than two orders of magnitude smaller than the teacher, intended to run ahead of every candidate-model invocation at negligible added latency.                                                                                                                                                                                                                                                                                                                                                                                                                            |

## 5. Teacher Supervision Pipeline

A central design decision in this study is the explicit separation of two
signals that are easy to conflate in a naive implementation: an _oracle_
label, grounded in actual model executions, and a _predicted_ label,
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
candidate tier (Section 4.3), and each response is scored using the
correctness metric defined for its source benchmark (Section 3). The
oracle label is assigned as the cheapest tier that answered correctly; if
multiple tiers succeed, the smallest/cheapest is preferred; if no tier
succeeds, the label defaults to Large, as the best available option among
those that were tried, even though it failed. This oracle label constitutes
ground truth for the purpose of measuring routing _quality_, as distinct
from routing _fidelity to the teacher_ (Section 7).

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

Each query in the training pool is stored as a single record with the
fields in Table 5.

**Table 5. Per-query training record schema.**

| Field                                             | Description                                                                             |
| ------------------------------------------------- | --------------------------------------------------------------------------------------- |
| `query`                                           | Raw input text                                                                          |
| `dataset_name`                                    | Source benchmark and split, e.g. `gsm8k/train`                                          |
| `oracle_label`                                    | Cheapest tier that answered correctly (ground truth, Section 5.1)                       |
| `oracle_correctness`                              | Per-tier pass/fail from executing every candidate                                       |
| `teacher_predicted_label`                         | Teacher's arg-max prediction (Section 5.2)                                              |
| `teacher_probabilities`                           | Full soft-label distribution over `{Small, Medium, Large}`                              |
| `teacher_rationale`                               | One-sentence explanation string, retained for auditing, not used as a training signal   |
| `per_tier_latency`                                | Wall-clock generation latency for each candidate's oracle execution                     |
| `per_tier_input_tokens`, `per_tier_output_tokens` | Token counts per candidate                                                              |
| `per_tier_inference_cost`                         | Derived from token counts and published per-tier pricing, or GPU-seconds if self-hosted |

## 6. Student Distillation

The student model receives only the raw query text as input — precisely
the information available at deployment time, excluding any oracle signal
or teacher rationale. Its primary training target is the teacher's soft
probability distribution (`teacher_probabilities`), optimized via a
cross-entropy or KL-divergence objective; its performance against the
oracle hard label (`oracle_label`) is tracked as a separate metric, so that
distillation fidelity and routing quality remain distinguishable
throughout training rather than being collapsed into one number.

A blended objective is used to prevent the student from purely inheriting
\

## 7. Evaluation Protocol

Evaluation is conducted on held-out splits of each benchmark (Section 3),
using the metrics in Table 6.

**Table 6. Evaluation metrics.**

| Metric                       | Definition                                                                                                                                                                         |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Routing accuracy vs. teacher | Proportion of queries where the student's label matches the teacher's — measures distillation fidelity.                                                                            |
| Routing accuracy vs. oracle  | Proportion of queries where the student's label matches the oracle ground-truth label — measures routing quality, the metric of primary practical interest.                        |
| Task accuracy                | End-task correctness (exact-match / F1 / pass@k, per benchmark) of the response produced by whichever tier the student routed to.                                                  |
| Cost saved                   | Aggregate cost of the tiers the student selected, relative to a fixed-Large baseline (and relative to the teacher's own routing cost, including the teacher's own inference cost). |
| Routing latency              | The student's own forward-pass latency to emit a decision, compared against the teacher's — the figure that substantiates the project's central efficiency claim.                  |
| End-to-end latency           | Routing latency plus the selected tier's generation latency, compared against an always-route-to-Large baseline.                                                                   |

Routing accuracy is reported disaggregated by dataset and by oracle label,
not only as a single aggregate figure: a router that is 95% accurate purely
because 90% of evaluation queries are trivially "Small" represents a
substantially weaker result than one that is 85% accurate with balanced
performance across all tiers, and aggregate accuracy alone would not
distinguish the two.

## 8. Deployment Architecture

At inference time, the pipeline is:

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
loop should therefore periodically re-run a sample of live or replayed
traffic through the oracle-labeling procedure (Section 5.1) and check for
drift between the deployed student's decisions and current oracle-optimal
routing.

## 9. Open Questions

Several design decisions are deliberately left open pending further
experimentation rather than assumed by default:

1. **Student output interface.** A generative JSON interface preserves
   symmetry with the teacher's I/O format, simplifying direct comparison of
   probability distributions, but introduces decode-time latency variance
   that a fixed classification head avoids. A classification head is
   recommended unless a specific downstream requirement favors the
   generative format.
2. **Fourth answering tier.** Whether to add an Ultra-Small candidate to the
   answering pool (Section 4.3) depends on what fraction of the deployed
   query distribution is genuinely trivial; this should be assessed
   empirically before committing the additional oracle-execution cost.
3. **Cross-family generalization run.** The cross-family configuration
   (Table 3) should be time-boxed as a follow-up study once the same-family
   configuration has validated the core hypothesis, rather than pursued in
   parallel with it.
