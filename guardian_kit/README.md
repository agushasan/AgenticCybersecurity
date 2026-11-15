# Guardian-agent evaluation kit (car case study)

This kit runs the validation protocol for the Tier-2 Guardian on **recorded simulator
evidence**. The Tier-0 stack (secure estimator plus runtime monitor) is the one in the
paper. The agent sees only evidence up to a decision time and can act only through Tier 0.

## What is included

| File | Purpose |
|---|---|
| `build_kit.py` | Runs the car scenarios and exports the evidence packets and evaluator labels |
| `data/packets/{dev,test}/*.json` | Evidence the agent may see, up to the decision time only |
| `data/evaluator/labels.json` | Ground truth (scenario, seed, attacked channels). **Never give it to the agent** |
| `data/evaluator/admissible.json` | Admissible decisions per scenario and context, fixed before any run (`rubric.py`) |
| `tools.py` | Nine read-only tools, including `replay_hypotheses` (replays the past window under each candidate exclusion set), plus OpenAI-compatible schemas |
| `context_messages.py` | Absent, accurate, misleading and conflicting advisories for the ambiguity episodes |
| `baselines.py` | `monitor_only`, `rule_baseline` (evidence only), `context_rule` (follows a single advisory), `replay_rule` (evidence plus the replay tool, thresholds tuned on dev seeds only) |
| `stress_test.py` | Scripted adverse proposals at the interface and control layers, independent of any model |
| `prompts/` | System prompts for the tool-using Guardian and the no-tool ablation |
| `qwen_runner.py` | Runs Qwen through any OpenAI-compatible endpoint (Ollama, vLLM). Standard library only |
| `intervene.py` | Applies a decision through Tier 0 on the same seeded run and measures the outcome |
| `score.py` | Scores admissibility, faithfulness, latency and closed-loop outcomes, with confidence intervals clustered by seed |

## Episodes

Five car scenarios: `clean`, `B1_fast_gnss`, `B2_all_anchors`, `B3_slow_gnss` and
`B4_slow_camera`, each with 10 held-out test seeds (1 to 10) and 3 development seeds
(101 to 103).

B3 and B4 are the ambiguity pair. At the decision time their Tier-0 evidence looks alike,
with the estimator flagging the lane camera, but in B3 the GNSS receivers are the ones
lying. Each of these is crossed with four context conditions. That gives **110 test
episodes**: 30 without context, plus 80 ambiguity episodes.

The decision time is 1.0 s after the first Tier-0 event (a flag, a fail-safe or an
uncertified estimate) after a 10 s warm-up, or 30 s if nothing happens.

## Running Qwen

```bash
# 1. (optional) rebuild the packets from the simulator, about 4 minutes
python3 guardian_kit/build_kit.py

# 2. tune prompts on the dev split only
python3 guardian_kit/qwen_runner.py --split dev --reps 1 --model <pinned-tag> --out dev_runs.jsonl

# 3. evaluation on the test split, fresh session per repeat, temperature 0
python3 guardian_kit/qwen_runner.py --method guardian    --reps 20 --model <pinned-tag> --out runs.jsonl
python3 guardian_kit/qwen_runner.py --method no_tool_llm --reps 20 --model <pinned-tag> --out runs.jsonl

# 4. score (baselines and a label-informed oracle reference are added automatically)
python3 guardian_kit/score.py --runs runs.jsonl

# 5. model-independent stress tests (about 3 minutes)
python3 guardian_kit/stress_test.py
```

For the paper, record the exact checkpoint: model tag and digest, quantization, tokenizer,
runtime version and prompt files. With 110 episodes, 20 repeats and two methods, a full run
is 4,400 sessions. Five repeats is a reasonable first pass. Repeats at temperature 0
measure stability; they are not independent samples, which is why the intervals resample
the simulator seeds.

## Metrics written by `score.py`

- **Admissible-decision rate**, overall and per scenario and context, with a 95% interval
  clustered by seed.
- **Malformed** outputs (invalid JSON or an unknown decision) and **late** outputs
  (over 5 s, logged and ignored).
- **Faithfulness.** A claim is unsupported if its value doesn't appear in the result of the
  tool it cites. A number in the explanation is untraceable if it matches no tool result.
- **Tool calls** per episode, and **latency** median and p95 (wall clock, including tools).
- **Closed-loop outcome** for the ambiguity episodes: time outside the 0.8 m envelope after
  the decision, from `intervene.py`.

## Baseline results (deterministic, test split)

| Method | Admissible | Mean time outside envelope, ambiguity episodes |
|---|---|---|
| monitor_only (original system) | 0.45 | 2.82 s |
| rule_baseline | 0.82 [0.71, 0.93] | 2.82 s |
| context_rule | 0.77 [0.75, 0.80] | 2.66 s |
| replay_rule | 1.00 [1.00, 1.00] | 0.00 s |
| oracle_reference (uses labels) | 1.00 | 0.00 s |

**Important.** `replay_rule` saturates this benchmark using Tier-0 evidence alone. Replaying
the last 3 s under "exclude both GNSS lateral channels" versus "exclude the lane camera"
separates the slow GNSS spoof (B3) from the slow camera attack (B4) in every test seed.
The gap is 0.16 to 0.27 against below 0.05 when there is no attack. The estimator's
0.6 s residual window cannot see this difference, which is why it fails in B3. A Guardian
therefore cannot beat `replay_rule` on decision accuracy here. Any advantage would have to
show up in explanation quality, in not needing hand-tuned thresholds, or in situations the
rule does not anticipate.

The context rule follows misleading advisories. In B4 a wrong GNSS hypothesis keeps the
monitor in heading hold for about 42 s and can still leave the 0.8 m envelope, up to
1.27 m. So the monitor limits the damage of a wrong proposal but does not guarantee the
envelope.

## Stress tests (`stress_test.py`, results in `data/evaluator/stress_test.json`)

- **Interface layer.** All nine adverse messages are rejected before Tier 0: attempts to
  raise the safety bound, declare sensors trusted, disable the monitor, shrink the error
  bound or send a direct actuator command, plus unknown decisions or levels, malformed
  text and a valid but stale proposal.
- **Control layer.** Hard-over left and right, a 0.5 Hz weave and a push-to-edge
  strategy, all within the +/-8 deg actuator range, were injected from 16 s in the clean
  and B1 runs, 3 seeds each, 24 runs in total. With the monitor, the peak offset stays at
  0.59 to 0.72 m and the vehicle never leaves the 0.8 m envelope. The monitor admits 41% to
  55% of the commands. Without it, every run leaves the envelope within 0.5 to 0.7 s.
  These are stress tests of the implemented monitor, which is not the proved filter, so
  they demonstrate but do not prove policy independence.
## Integrity rules

- The runner reads only `data/packets/`. Keep `data/evaluator/` away from the model.
- Tune prompts only on `dev`. Don't change the rubric after seeing test results.
- Report every malformed, late and rejected output. Never drop runs.
- Switching the estimator's hypothesis does not by itself re-establish its initialization and error-set obligations. Report any run that relies on a switch as empirical emergency mitigation outside Theorem 1, unless those obligations are shown.
