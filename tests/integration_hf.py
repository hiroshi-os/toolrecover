"""Download-free validation of actual causal-LM LoRA SFT/RL and reload.

The random local model validates the backend, not pretrained model quality.
Run explicitly: python tests/integration_hf.py
"""
import json
import tempfile
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from toolrecover.learning import train, evaluate


def main():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    with tempfile.TemporaryDirectory(prefix="toolrecover-hf-") as directory:
        root = Path(directory)
        base = root / "base"
        base.mkdir()
        vocab = {"[UNK]": 0, "[EOS]": 1, "[PAD]": 2}
        vocab.update({str(i): i + 3 for i in range(9)})
        raw = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
        raw.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=raw, unk_token="[UNK]", eos_token="[EOS]", pad_token="[PAD]"
        )
        tokenizer.save_pretrained(base)
        GPT2LMHeadModel(GPT2Config(
            vocab_size=len(vocab), n_positions=1024, n_embd=16, n_layer=1, n_head=2,
            eos_token_id=1, pad_token_id=2,
        )).save_pretrained(base)
        report = train(root / "adapter", seed=17, sft_steps=1, rl_steps=1,
                       group_size=2, backend="hf", model_name=str(base))
        assert report["parameter_change_l2"] > 0
        result = evaluate(root / "adapter" / "grpo.pt", split="validation", count=4)
        assert result["count"] == 4
        summary = {
            "kind": "random_local_causal_lm_lora_backend_integration",
            "sft_steps": 1, "rl_steps": 1, "group_size": 2,
            "parameter_change_l2": report["parameter_change_l2"],
            "rl_parameter_change_l2": report["rl_parameter_change_l2"],
            "reload_evaluation_count": result["count"],
            "random_model_success_rate": result["success_rate"],
            "pretrained_quality_measured": False,
        }
    destination = Path("reports/hf-integration.json")
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
