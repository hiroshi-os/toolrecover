"""Fixed three-seed CPU protocol. Does not select on test scores."""
import json
import argparse
from pathlib import Path
from toolrecover.learning import train, evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-steps", type=int, default=24)
    parser.add_argument("--output", type=Path, default=Path("reports/cpu-results.json"))
    args = parser.parse_args()
    results = []
    for seed in (0, 1, 2):
        out = Path("artifacts") / f"seed{seed}"
        run = train(out, seed=seed, sft_steps=args.sft_steps, rl_steps=80, group_size=8)
        stages = {}
        for stage in ("sft", "grpo"):
            stages[stage] = evaluate(out / f"{stage}.pt", count=100, seed=seed)
            stages[stage + "_challenge"] = evaluate(out / f"{stage}.pt", split="challenge", count=100, seed=seed)
        for baseline in ("scripted", "random"):
            stages[baseline] = evaluate(baseline=baseline, count=100, seed=seed)
        results.append({"seed": seed, "config": {"sft_steps": args.sft_steps, "rl_steps": 80, "group_size": 8},
                        "sft_final_loss": run["sft_final_loss"],
                        "rl_parameter_change_l2": run["rl_parameter_change_l2"],
                        "nonzero_advantage_groups": sum(h["advantage_std"] > 0 for h in run["history"]),
                        "results": {name: {k: v for k, v in value.items() if k != "episodes"}
                                    for name, value in stages.items()}})
    destination = args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(destination.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
