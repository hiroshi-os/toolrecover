# Executed CPU experiments

Executed 2026-10-23 on Windows 11, Python 3.12.5, CPU Torch 2.14.1.
Exact environment versions are in `reports/validation-environment.json`.

## Checks

- `python -m unittest discover -s tests -v`: 14 passed, including real optimizer updates and checkpoint reload.
- `python tests/integration_hf.py`: actual LoRA updates through a random local causal LM, supervised and grouped RL gradients, adapter save/reload and four evaluation episodes. The model achieved zero task success, as expected for this integration-only random model. This validates the executable backend, not pretrained language-model ability. Output: `reports/hf-integration.json`.

## Negative control and validation selection

`python scripts/experiment.py --sft-steps 8 --output reports/undertrained-cpu-results.json`
produced zero completion for both stages at seeds 0, 1 and 2. RL learned early
submission: no invalid calls and mean reward -0.01, but no verified success.
This exposes the interaction between a sparse completion reward, per-action
costs, and an insufficient supervised warm-start. The result is preserved,
not replaced by a claim of successful learning.

`python scripts/validation_sweep.py` compared 24 and 48 SFT updates at seed 0,
using only 40 validation episodes for each stage. Both settings reached 100%
validation completion; the smaller 24-step setting was chosen. The final protocol
then ran three seeds without selecting a best seed.

## Fixed three-seed run

`python scripts/experiment.py` runs 24 SFT steps, 80 grouped RL iterations,
eight sampled trajectories per group, two clipped updates per group, and
100 episodes per evaluation condition. Exact outputs: `reports/cpu-results.json`.

| Seed | SFT instance test | Reward-trained instance test | SFT composed challenge | Reward-trained challenge |
| --- | ---: | ---: | ---: | ---: |
| 0 | 100% | 100% | 0% | 0% |
| 1 | 100% | 100% | 0% | 0% |
| 2 | 100% | 100% | 0% | 0% |

RL parameters changed in all three runs and every group had nonzero reward
advantages. However, RL did not improve completion over the supervised baseline.
The scripted planner solved the ordinary instance test; the random baseline
solved none in these runs. Reward and invalid-action counts remain in the JSON,
so a completion percentage does not hide policy inefficiency.

These are repeated small finite-state decision structures, not 100 independent
natural-language problems. The composed challenge combines known prerequisites
in an unseen plan. Its zero score is the meaningful generalization limitation.
No pretrained-model improvement, code-patch generation, human-feedback training,
or production coding-agent performance was measured.
