import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Tool-error recovery experiments")
    subs = parser.add_subparsers(dest="command", required=True)
    t = subs.add_parser("train")
    t.add_argument("--out", default="artifacts/seed0")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--sft-steps", type=int, default=150)
    t.add_argument("--rl-steps", type=int, default=100)
    t.add_argument("--group-size", type=int, default=8)
    t.add_argument("--backend", choices=["tiny", "hf"], default="tiny")
    t.add_argument("--model-name", default=None)
    e = subs.add_parser("eval")
    e.add_argument("--checkpoint")
    e.add_argument("--baseline", choices=["model", "scripted", "random"], default="model")
    e.add_argument("--split", choices=["validation", "test", "challenge"], default="test")
    e.add_argument("--count", type=int, default=100)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--out")
    args = vars(parser.parse_args())
    command = args.pop("command")
    from .learning import train, evaluate
    if command == "train":
        report = train(**args)
        print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2))
    else:
        out = args.pop("out")
        report = evaluate(**args)
        if out:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in report.items() if k != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
