"""Actual supervised + grouped clipped policy optimization in PyTorch."""
import copy
import json
import random
from pathlib import Path

from .env import ACTIONS, RecoveryEnv, make_task, scripted_action


def torch_module():
    import torch
    return torch


def policy(backend="tiny", model_name=None):
    torch = torch_module()
    if backend == "hf":
        from .hf_policy import LanguagePolicy
        return LanguagePolicy(model_name or "HuggingFaceTB/SmolLM2-135M-Instruct")
    return torch.nn.Sequential(torch.nn.Linear(16, 48), torch.nn.Tanh(),
                               torch.nn.Linear(48, len(ACTIONS)))


def rollout(model, task, sample=True):
    torch = torch_module()
    env = RecoveryEnv(task)
    states, actions, logs = [], [], []
    while not env.done:
        x = torch.tensor(env.features(), dtype=torch.float32)
        with torch.no_grad():
            dist = torch.distributions.Categorical(logits=model(x))
            a = dist.sample() if sample else model(x).argmax()
            logs.append(dist.log_prob(a).item())
        states.append(x)
        actions.append(a.item())
        env.step(a.item())
    return {"states": torch.stack(states), "actions": torch.tensor(actions),
            "old_logs": torch.tensor(logs), "reward": env.reward(),
            "success": env.verify(), "steps": env.steps, "invalid": env.invalid}


def train(out, seed=0, sft_steps=150, rl_steps=100, group_size=8, backend="tiny", model_name=None):
    if group_size < 2:
        raise ValueError("group-size must be >= 2")
    torch = torch_module()
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    random.seed(seed)
    if sft_steps < 0 or rl_steps < 0:
        raise ValueError("training steps must be nonnegative")
    model = policy(backend, model_name)
    # Keep dropout disabled during sampling AND likelihood updates. eval() does
    # not disable gradients, and keeps old-policy importance ratios coherent.
    model.eval()
    trainable = [p for p in model.parameters() if p.requires_grad]
    before = torch.cat([p.detach().flatten().clone() for p in trainable])
    optimizer = torch.optim.Adam(trainable, lr=0.01 if backend == "tiny" else 2e-4)
    states, actions = [], []
    for index in range(64):
        env = RecoveryEnv(make_task("train", index))
        while not env.done:
            a = scripted_action(env)
            states.append(env.features())
            actions.append(a)
            env.step(a)
    x, y = torch.tensor(states), torch.tensor(actions)
    sft_loss = None
    for _ in range(sft_steps):
        # Bound causal-LM activation memory; tiny policy uses all demonstrations.
        indices = torch.randint(len(x), (8,)) if backend == "hf" else torch.arange(len(x))
        loss = torch.nn.functional.cross_entropy(model(x[indices]), y[indices])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sft_loss = loss.item()
    output = Path(out)
    output.mkdir(parents=True, exist_ok=True)
    def save(stage):
        torch.save({"state_dict": model.adapter_state() if backend == "hf" else model.state_dict(),
                    "backend": backend, "model_name": model_name, "seed": seed, "stage": stage}, output / f"{stage}.pt")
    save("sft")
    after_sft = torch.cat([p.detach().flatten().clone() for p in trainable])
    reference = copy.deepcopy(model).eval()
    for p in reference.parameters():
        p.requires_grad_(False)
    optimizer = torch.optim.Adam(trainable, lr=0.001 if backend == "tiny" else 1e-5)
    history = []
    for iteration in range(rl_steps):
        # Same prompt/task within a group. Distinct training tasks across iterations.
        task = make_task("train", 100 + iteration)
        group = [rollout(model, task) for _ in range(group_size)]
        rewards = torch.tensor([g["reward"] for g in group])
        advantages = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-6)
        # Reuse old-policy rollouts for two clipped updates; frozen SFT KL reference.
        losses = []
        for _ in range(2):
            optimizer.zero_grad()
            loss_value = 0.0
            for g, advantage in zip(group, advantages):
                logits = model(g["states"])
                dist = torch.distributions.Categorical(logits=logits)
                ratio = (dist.log_prob(g["actions"]) - g["old_logs"]).exp()
                objective = torch.minimum(ratio * advantage, ratio.clamp(0.8, 1.2) * advantage).mean()
                ref = torch.distributions.Categorical(logits=reference(g["states"]))
                kl = torch.distributions.kl_divergence(dist, ref).mean()
                loss = (-objective + 0.02 * kl) / group_size
                # Accumulate the group mean without retaining every rollout's
                # activation graph at once (important for the LM backend).
                loss.backward()
                loss_value += loss.detach().item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss_value)
        history.append({"iteration": iteration, "mean_reward": rewards.mean().item(),
                        "success_rate": sum(g["success"] for g in group) / group_size,
                        "loss": losses[-1], "advantage_std": advantages.std(unbiased=False).item()})
    after = torch.cat([p.detach().flatten() for p in trainable])
    save("grpo")
    report = {"seed": seed, "sft_steps": sft_steps, "rl_steps": rl_steps,
              "group_size": group_size, "sft_final_loss": sft_loss,
              "backend": backend, "model_name": model_name,
              "parameter_change_l2": (after - before).norm().item(),
              "rl_parameter_change_l2": (after - after_sft).norm().item(), "history": history}
    (output / "training.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def evaluate(checkpoint=None, baseline="model", split="test", count=100, seed=0):
    if split == "train":
        raise ValueError("evaluation requires validation or test split")
    if count < 1:
        raise ValueError("count must be positive")
    rng = random.Random(seed)
    model = None
    if baseline == "model":
        if checkpoint is None:
            raise ValueError("--checkpoint required for model")
        torch = torch_module()
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model = policy(saved.get("backend", "tiny"), saved.get("model_name"))
        if saved.get("backend") == "hf":
            model.load_adapter_state(saved["state_dict"])
        else:
            model.load_state_dict(saved["state_dict"])
        model.eval()
    rows = []
    for index in range(count):
        task = make_task(split, index)
        if model is not None:
            result = rollout(model, task, sample=False)
            row = {k: result[k] for k in ("success", "steps", "invalid", "reward")}
        else:
            env = RecoveryEnv(task)
            while not env.done:
                env.step(scripted_action(env) if baseline == "scripted" else rng.randrange(len(ACTIONS)))
            row = {"success": env.verify(), "steps": env.steps, "invalid": env.invalid, "reward": env.reward()}
        rows.append({"task": task.identity, "family": task.family, **row})
    return {"baseline": baseline, "checkpoint": str(checkpoint) if checkpoint else None,
            "split": split, "seed": seed, "count": count,
            "success_rate": sum(r["success"] for r in rows) / count,
            "mean_reward": sum(r["reward"] for r in rows) / count,
            "mean_steps": sum(r["steps"] for r in rows) / count,
            "invalid_calls": sum(r["invalid"] for r in rows), "episodes": rows}
