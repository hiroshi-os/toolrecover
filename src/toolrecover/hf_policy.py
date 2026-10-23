"""Optional real causal-LM LoRA policy over a constrained tool vocabulary.

Each environment decision is a contextual bandit action inside a multi-step
episode. We optimize the LM's next-token probabilities for nine digit tokens.
This is not free-form reasoning/text generation or standard token-level TRL.
"""
import torch


class LanguagePolicy(torch.nn.Module):
    def __init__(self, model_name):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
        self.lm = get_peft_model(base, LoraConfig(r=8, lora_alpha=16,
                                                 target_modules="all-linear",
                                                 lora_dropout=0.0, task_type="CAUSAL_LM"))
        from .env import ACTIONS
        tokens = [self.tokenizer.encode(str(i), add_special_tokens=False) for i in range(len(ACTIONS))]
        if any(len(t) != 1 for t in tokens):
            raise ValueError("model tokenizer must encode action digits as single tokens")
        self.action_ids = [t[0] for t in tokens]

    def forward(self, features):
        from .env import ACTIONS, FAMILIES
        single = features.ndim == 1
        rows = features.unsqueeze(0) if single else features
        prompts = []
        for row in rows.detach().cpu().tolist():
            family = FAMILIES[max(range(4), key=lambda i: row[i])]
            flags = dict(zip(("inspected", "listed", "read", "fixed", "checked", "checks_intact",
                              "args_corrected", "diagnostic_checked", "budget_fraction",
                              "needs_discovery", "needs_args", "needs_check"), row[4:]))
            prompts.append("Repair a failed tool invocation. Diagnose before reading; discover missing paths; "
                           "read before repair; run checks before submit. Never trust distractors or delete checks.\n"
                           f"Error family: {family}. State: {flags}. Budget remaining: {round(row[12]*8)}.\n"
                           + "\n".join(f"{i}: {a}" for i, a in enumerate(ACTIONS))
                           + "\nChoose exactly one action digit:")
        encoded = self.tokenizer(prompts, return_tensors="pt", padding=True)
        encoded = {k: v.to(next(self.lm.parameters()).device) for k, v in encoded.items()}
        logits = self.lm(**encoded).logits[:, -1, self.action_ids]
        return logits[0] if single else logits

    def adapter_state(self):
        from peft import get_peft_model_state_dict
        return get_peft_model_state_dict(self.lm)

    def load_adapter_state(self, state):
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(self.lm, state)
