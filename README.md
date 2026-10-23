# Toolrecover

Small, reproducible experiments in **learning to recover from tool errors**.
A safe simulator provides missing-path, malformed-argument, failing-check and
misleading-result scenarios. Train a supervised policy, then optimize verified
completion rewards with grouped clipped policy updates. Compare both checkpoints
against explicitly labeled random and hand-written baselines.

This is an experimental harness, not evidence of production coding-agent ability.
The default model is a small PyTorch policy over simulator features. A separate
Hugging Face causal-LM + LoRA backend learns the same constrained action decisions.

## Run locally

Python 3.10+; the default experiment requires CPU PyTorch and no model downloads.

```sh
python -m venv .venv
# Activate .venv using your shell's usual command.
python -m pip install -e '.[train]'
python -m unittest discover -s tests -v
python -m toolrecover.cli train --out artifacts/seed0 --seed 0 --sft-steps 8 --rl-steps 80
python -m toolrecover.cli eval --checkpoint artifacts/seed0/sft.pt --out artifacts/sft-eval.json
python -m toolrecover.cli eval --checkpoint artifacts/seed0/grpo.pt --out artifacts/grpo-eval.json
python -m toolrecover.cli eval --baseline scripted
python -m toolrecover.cli eval --baseline random --seed 0
python scripts/experiment.py
```

`scripts/experiment.py` fixes three seeds and records independently evaluated SFT,
reward-trained, scripted and random policies in `reports/cpu-results.json`.
Checkpoints and full training traces remain local under ignored `artifacts/`.
The report in this repository describes the run actually performed; no pretrained
model improvement is implied by the tiny-policy run.

## Optional causal-LM LoRA training

```sh
python -m pip install -e '.[hf]'
python -m toolrecover.cli train --backend hf --model-name HuggingFaceTB/SmolLM2-135M-Instruct --out artifacts/smollm --sft-steps 10 --rl-steps 3 --group-size 2
python -m toolrecover.cli eval --checkpoint artifacts/smollm/grpo.pt --count 8
```

This downloads public model weights and requires more memory/time. It uses actual
PEFT LoRA adapters (`r=8`, alpha 16, all linear targets), supervised cross-entropy,
and gradients through model action probabilities during reward training. Each
observation becomes a text prompt; the next-token logits for digits 0–8 define a
nine-action categorical distribution. Arbitrary tool names, arguments, chain of
thought and free-form program generation are outside this experiment. This is a
custom constrained-action policy optimizer, **not TRL GRPOTrainer**. Evaluation
reloads the base model and saved adapter state. Base weights are not committed.

## Environment and verifier

Actions: inspect the failure, discover paths, read target, read distractor,
repair, run checks, submit, delete checks, or correct argument schemas. Every
episode has an eight-action budget. Missing files require path discovery;
argument errors require schema correction before reading; failing checks must
be run to collect a diagnostic before inspection. Distractor tasks require
listing to distinguish the authoritative target. These mechanics produce
different required plans. The `challenge` split combines path discovery,
schema correction and check diagnostics into an unseen eight-action plan.
The simulator never executes a shell or writes an agent-controlled host file.
The repair action installs the simulator's expected value once its prerequisites
are met: this is learning action planning, **not learning to write code patches**.
An exact hidden-state verifier requires the expected repaired value, intact
checks and a passing check after the repair. Agent-authored strings such as
"PASS" cannot satisfy it. Removing checks invalidates success.

Terminal reward is a transparent sum:

| Component | Value |
|---|---:|
| Verified completion | +1 |
| Every action | −0.01 |
| Invalid call | −0.05 |
| Distractor read | −0.02 |
| Deleted checks | −0.5 |

Identities, filenames and private expected values use disjoint train, validation
and test namespaces and random streams. Training never samples validation/test.
All four scenario labels occur in all splits. The default test is **instance
holdout**, not unseen-family evaluation. The separate challenge has a prerequisite
composition not seen during training; report its score separately:

```sh
python -m toolrecover.cli eval --checkpoint artifacts/seed0/grpo.pt --split challenge
```

Because policies observe a small finite state encoding,
many held-out instances share the same decision structure; 100 test episodes do
not represent 100 independent natural-language tasks. The simulator is deliberately
small and deterministic. Invalid arguments are represented by a failed tool state,
not by open-ended JSON generation. Even the composed challenge uses the same
individual actions and verifier. Results should not be sold as broad
out-of-distribution generalization or real software-repair success.

## Objective

For each training task, sample G complete trajectories with the current (old)
policy. Normalize each terminal reward within that group:
`A_i = (R_i - mean(R)) / (std_population(R) + 1e-6)`.
For two optimization epochs, minimize:

```text
mean_i mean_t [ -min(r_it A_i, clip(r_it, .8, 1.2) A_i)
                + .02 KL(pi_current(.|s_it) || pi_frozen_SFT(.|s_it)) ]
r_it = pi_current(a_it|s_it) / pi_old(a_it|s_it)
```

Old action log probabilities are detached. The KL is computed exactly over all
nine actions, the reference is frozen, and gradients are clipped to norm 1.
Each trajectory is weighted equally, then its actions are averaged. Equal-reward
groups have zero policy-gradient advantage; KL can still update the policy.
There is no learned value model. This adapts the group-relative clipped objective
to multi-step constrained actions; it does not reproduce every token-level loss,
masking convention or normalization option in TRL. It is RL with verifiable
rewards, **not RLHF**, RLAIF or human-preference learning.

The experiment intentionally uses only eight SFT updates to leave room to study
reward training. It is not a claim that RL outperforms a converged SFT baseline.
Do not tune on the published test scores; use validation for new experiments and
reserve a fresh test split for a final claim.

## Tests and extensions

Tests cover successful recovery, text-based reward spoofing, deleted-check reward
hacking, missing-path prerequisites, malformed action IDs, deterministic execution,
split separation, action-budget enforcement, actual optimizer weight changes and
checkpoint reloads. Torch-dependent tests skip when torch is absent.

Useful next experiments: converged-SFT control, reward-component ablations,
larger state/action spaces, held-out recovery compositions, unseen tool schemas,
free-form argument validation, pretrained-model multi-seed runs and real sandboxed
repositories. None of those results is claimed here.

## References

- [TRL GRPO method and trainer documentation](https://huggingface.co/docs/trl/grpo_trainer)
- [PEFT LoRA reference](https://huggingface.co/docs/peft/package_reference/lora)
- [PyTorch categorical distribution and KL APIs](https://pytorch.org/docs/stable/distributions.html)

MIT licensed. Model licenses remain those of their respective publishers.
