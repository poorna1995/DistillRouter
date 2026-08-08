DistllRouter: An Lightweight Distilled Router

Problem Statement:
Can the routing policy learned by a language model be compressed into a significantly smaller model while preserving routing quality?

Motivation:
Modern LLM applications use a large model to determine which model should process an incoming query. Although accurate, the router itself introduces significant latency into the critical inference path.

The objective of this work is to distill the routing knowledge of a larger teacher router into a much smaller student router while maintaining comparable routing accuracy and significantly reducing inference latency.

Here the assumption is larger means higher parameter size models

Offline Phase:( training)

Dataset Preparation:
Benchmark dataset
For math : GSM8K, Math  
What splits are available
how the correctness are measured
Total dataset,what are the complexity range
For General Knowledge: TriviaQA, HotpotQA
What splits are available
how the correctness are measured
Total dataset,what are the complexity range
For coding : HumanEval , MBPP
What splits are available
how the correctness are measured
Total dataset,what are the complexity range
Synthetic Dataset:

List all the models from a distinct model family or same family models( for teacher, student and for the routing models.
OptionA: Same Family
Qwen, Llama, Gemma, Phi( any one)
OptionB: Cross family
Qwen, Llama, Gemma, Phi, Mistral, Deepseek
Choose which model is for the Teacher and Student model,and ( small,medium large)
List the design choices why,
What are the assumption

Router Label
Parameter Range
Example Models
Ultra Small
<1B
Qwen2.5-0.5B, Qwen3-0.6B
Small
1B–4B
Llama 3.2 1B/3B, Qwen2.5 1.5B/3B, Gemma 3 1B/4B, Phi-3 Mini 3.8B
Medium
7B–14B
Qwen3-8B, Gemma 2 9B, Gemma 3 12B, Phi-3 Small 7B, Phi-3 Medium 14B
Large
30B–72B
Qwen2.5-32B, Qwen2.5-72B, Llama 3.1-70B

Teacher Model :
Prompt Design:
Write a detail instruction in prompt : Output = {Labe1: Small | Medium | Large, Probabilities:{Small:0.23, Medium:0.3,Large:0.5}, models logs: Latency, Cost, answer: correctness
Build a training dataset :
Store: query, dataset_name, teacher_routing_label, teacher_probability_distribution, teacher_latency, input_tokens, output_tokens, inference_cost.

Student Distillation:
Train smaller model against the teacher model supervision data to predict the routing decisions
Hard routing labels
Cross-entropy classification loss.
Track latency,cost( input+output),

Evaluation:
Evaluate the trained student using test dataset
Routing Accuracy : Total prediction correct / total queries
Latency
Task Accuracy

Deployment Phase:
Step1: Input Query
Step2 : Student router Executes
Step3: Route to choose model
Step4: model generate response.
Step5: Accuracy, latency

Assumption:
Larger parameter-count models generally possess greater reasoning capability than smaller models for complex tasks, motivating parameter-based routing.
Query difficulty can be inferred from the input text without executing multiple candidate models.
A lightweight student model can approximate the routing policy of a larger teacher through knowledge distillation.

Benchmarks:
GSM8K (link): is a grade-school arithmetic problems, 2–8 reasoning steps,( Total size 8,792 problems with train: 7,473 / test: 1,319 ) and correctness, Exact-match on the final numeric answer, extracted via the <answer> delimiter.
MATH (link) : is a 5 explicit difficulty levels (1=easiest, 5=competition-level).(total size: 12,500 problems with train: 7,500 / test: 5,000). And Exact-match on the final answer inside \boxed{}, with symbolic/string normalization (fractions, LaTeX, ordering)
General Knowledge

TriviaQA (mandarjoshi/trivia_qa on HF)
HotpotQA
subsets
8 total: rc = reading-comprehension format with evidence docs attached; nocontext variants strip the evidence and give just Q/A pairs; unfiltered = the open-domain variant used for the official leaderboard
distractor setting (10 paragraphs, 2 gold + 8 distractors).
Splits (row counts)
train 138,384 / validation 18,669 / test 17,210. unfiltered / unfiltered.nocontext: train 87,622 / validation 11,313 / test 10,832. use validation as your held-out set for offline routing eval
train: 90,447 / dev: 7,405 / test: 7,405 (hidden)
Correctness metric
Exact-match / F1 computed against the answer's alias
EM and F1 on the answer span, plus a separate supporting-fact F1
Total size
Full dataset: 650K+ question-answer-evidence triples; 95K distinct question-answer pairs, each with ~6 evidence documents on average (per dataset card)
~113K QA pairs

Coding

HumanEval
MBPP
Splits
Single set, no train/test split — 164 problems, used entirely for eval
Full: train 374 / test 500 / validation 90 / few-shot prompt 10. Sanitized subset: 427 hand-verified problems
Correctness metric
pass@k — generated code executed against 7–8 hidden unit tests per problem
pass@k — executed against ~3 assert-based unit tests per problem
Total size
164 problems
974 (full) / 427 (sanitized)

Model selection:
Teacher mod

Models:
Ultra Small Models:
Qwen2.5 = 0.5B → Alibaba
Qwen3 = 0.6B —> Alibaba
Gemma 3 = 270M → Google

Small Mode (SLM)
Llama 3.2 = 1B , 3B
Qwen2.5 = 1.5B , 3B
Gemma 3 = 1B , 4B
Qwen3 = 1.7B , 4B
Phi-3 Mini, Phi-4 Mini = 3.8B

Small-medium (4B–10B)
Qwen3-8B = 8B
Gemma 2 = 9B
Gemma 4 = 12B
Phi-3 Small = 7B
Phi-4 = 14b
