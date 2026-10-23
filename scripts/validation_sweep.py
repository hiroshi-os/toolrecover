"""Select warm-start length on validation, never test/challenge scores."""
import json
from pathlib import Path
from toolrecover.learning import train, evaluate

rows = []
for steps in (24, 48):
    output = Path("artifacts") / f"validation-sft{steps}"
    train(output, seed=0, sft_steps=steps, rl_steps=80, group_size=8)
    rows.append({
        "sft_steps": steps,
        "sft": evaluate(output / "sft.pt", split="validation", count=40)["success_rate"],
        "grpo": evaluate(output / "grpo.pt", split="validation", count=40)["success_rate"],
    })
Path("reports").mkdir(exist_ok=True)
Path("reports/validation-sweep.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
print(rows)
