import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from toolrecover.preferences import validate, assign_splits, parse_judgment, write_rows


def row(i=0, split="train"):
    return {"id": str(i), "prompt": f"task {i}", "chosen": "good", "rejected": "bad",
            "split": split, "provenance": {"label_type": "scripted", "source": "unit test",
            "license": "MIT", "method": "synthetic constant preference, not human"}}


class DataTests(unittest.TestCase):
    def test_validation(self):
        for bad in ({}, dict(row(), chosen="BAD"), dict(row(), provenance={}), dict(row(), split="dev")):
            with self.assertRaises(ValueError):
                validate([bad])
        with self.assertRaises(ValueError):
            validate([row(), dict(row(1, "test"), prompt="TASK   0")])
        with self.assertRaises(ValueError):
            validate([row(), dict(row(1), prompt="task 0", chosen="bad", rejected="good")])
        self.assertEqual(validate([row()])[0]["provenance"]["label_type"], "scripted")

    def test_split_prompt_groups(self):
        rows = assign_splits([row(i) for i in range(20)])
        self.assertEqual({r["split"] for r in rows}, {"train", "validation", "test"})
        self.assertEqual(rows, assign_splits([row(i) for i in range(20)]))
        validate(rows)

    def test_judge_strict_parser(self):
        self.assertEqual(parse_judgment('{"winner":"A"}'), "A")
        for invalid in ('A', '{"winner":"tie"}', '[]', '{"winner":"A","reason":"x"}', '```{"winner":"B"}```'):
            self.assertIsNone(parse_judgment(invalid))


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"), "HF extra required")
class ObjectiveTests(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(1)

    def test_dpo_analytical_sign_reference_and_extremes(self):
        import torch
        from toolrecover.preferences import dpo_loss, reward_loss
        policy = torch.tensor([-2., -4.], requires_grad=True)
        reference = torch.tensor([-3., -3.], requires_grad=True)
        loss, margin = dpo_loss(policy, reference, 0.5)
        self.assertAlmostEqual(margin.item(), 1.)
        self.assertAlmostEqual(loss.item(), 0.3132617, places=6)
        loss.backward()
        self.assertLess(policy.grad[0].item(), 0)
        self.assertGreater(policy.grad[1].item(), 0)
        self.assertIsNone(reference.grad)
        _, improvement = dpo_loss(torch.tensor([-6., -1.]), torch.tensor([-11., -1.]))
        self.assertGreater(improvement.item(), 0)  # raw policy margin is still -5
        for values in (torch.ones(3), torch.ones(0), torch.ones(2, 2)):
            with self.assertRaises(ValueError):
                reward_loss(values)
            with self.assertRaises(ValueError):
                dpo_loss(values, values)
        loss, _ = reward_loss(torch.tensor([-10000., 10000.]))
        self.assertTrue(torch.isfinite(loss))

    def test_completion_mask_matches_individual_token_likelihoods(self):
        import torch
        from transformers import GPT2Config, GPT2LMHeadModel
        from toolrecover.preferences import batch_pairs, completion_logps
        tokenizer = make_tokenizer()
        torch.manual_seed(8)
        model = GPT2LMHeadModel(GPT2Config(vocab_size=len(tokenizer), n_positions=32,
                                         n_embd=16, n_head=2, n_layer=1)).eval()
        rows = [row(), dict(row(1), prompt="task 1 task", chosen="good good")]
        batch = batch_pairs(tokenizer, rows)
        result = completion_logps(model, batch)
        for i in range(4):
            length = batch["attention_mask"][i].sum().item()
            ids = batch["input_ids"][i, :length]
            mask = batch["completion_mask"][i, :length]
            manual = 0.
            for pos in range(1, length):
                if mask[pos]:
                    logits = model(ids[:pos].unsqueeze(0)).logits[0, -1].log_softmax(-1)
                    manual = manual + logits[ids[pos]]
            self.assertTrue(torch.allclose(result[i], manual, atol=1e-5))
        with self.assertRaises(ValueError):
            batch_pairs(tokenizer, rows, max_length=2)

    def test_dpo_improvement_is_not_raw_pair_accuracy(self):
        import torch
        from unittest.mock import patch
        from toolrecover.preferences import evaluate_pairs
        model = torch.nn.Linear(1, 1)
        with patch("toolrecover.preferences.batch_pairs", return_value={}), patch(
            "toolrecover.preferences.completion_logps", side_effect=[torch.tensor([-6., -1.]), torch.tensor([-11., -1.])]):
            result = evaluate_pairs(model, None, [row()], "dpo", model)
        self.assertEqual(result["pair_accuracy"], 0.)
        self.assertEqual(result["dpo_improvement_rate"], 1.)

    def test_real_lm_dpo_reward_train_reload_and_heldout(self):
        import torch
        from transformers import GPT2Config, GPT2LMHeadModel, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
        from toolrecover.preferences import train_preferences, read_rows, evaluate_pairs
        torch.manual_seed(3)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base = root / "base"
            tokenizer = make_tokenizer()
            tokenizer.save_pretrained(base)
            GPT2LMHeadModel(GPT2Config(vocab_size=len(tokenizer), n_positions=32, n_embd=16,
                n_layer=1, n_head=2, pad_token_id=2, eos_token_id=1)).save_pretrained(base)
            data = root / "pairs.jsonl"
            # Held-out prompt identities, same synthetic good-over-bad rule.
            write_rows(data, assign_splits([row(i) for i in range(12)]))
            reports = {}
            for method in ("dpo", "reward"):
                output = root / method
                report = train_preferences(data, output, method, str(base), steps=35,
                    batch_size=4, lr=0.003, beta=0.5, max_length=32, seed=3)
                self.assertGreater(report["history"][0]["gradient_norm"], 0)
                self.assertLess(report["validation"]["loss"], report["validation_before"]["loss"])
                self.assertEqual(report["test"]["pair_accuracy"], 1.)
                cls = AutoModelForCausalLM if method == "dpo" else AutoModelForSequenceClassification
                model = cls.from_pretrained(output / "model").eval()
                ref = AutoModelForCausalLM.from_pretrained(output / "reference").eval() if method == "dpo" else None
                if ref is not None:
                    original = AutoModelForCausalLM.from_pretrained(base)
                    self.assertTrue(all(torch.equal(v, ref.state_dict()[k]) for k, v in original.state_dict().items()))
                loaded = evaluate_pairs(model, AutoTokenizer.from_pretrained(output / "model"),
                    [r for r in read_rows(data) if r["split"] == "test"], method, ref, beta=0.5, max_length=32)
                self.assertAlmostEqual(loaded["loss"], report["test"]["loss"], places=6)
                reports[method] = {k: report[k] for k in ("steps", "label_counts", "validation_before", "validation", "test",
                    "model_revision", "tokenizer_revision", "device", "dtype", "package_versions", "data_sha256")}
                reports[method]["first_gradient_norm"] = report["history"][0]["gradient_norm"]
            destination = Path("reports/preference-integration.json")
            destination.parent.mkdir(exist_ok=True)
            destination.write_text(json.dumps({"kind": "local_random_HF_models_synthetic_scripted_preferences",
                "human_training_completed": False, "AI_judge_run_completed": False,
                "pretrained_GPU_run_completed": False, "results": reports}, indent=2), encoding="utf-8")


def make_tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast
    vocab = {"[UNK]": 0, "[EOS]": 1, "[PAD]": 2, "task": 3, "good": 4, "bad": 5}
    vocab.update({str(i): 6+i for i in range(20)})
    raw = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    raw.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]", eos_token="[EOS]", pad_token="[PAD]")
