import importlib.util
import tempfile
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional torch dependency unavailable")
class LearningTests(unittest.TestCase):
    def test_optimizer_updates_and_checkpoint_reloads(self):
        from toolrecover.learning import train, evaluate
        with tempfile.TemporaryDirectory() as out:
            report = train(out, seed=7, sft_steps=4, rl_steps=3, group_size=4)
            self.assertGreater(report["parameter_change_l2"], 0)
            self.assertGreater(report["rl_parameter_change_l2"], 0)
            result = evaluate(out + "/grpo.pt", count=8)
            self.assertEqual(result["count"], 8)
            self.assertTrue(0 <= result["success_rate"] <= 1)

    def test_train_split_cannot_be_reported_as_eval(self):
        from toolrecover.learning import evaluate
        with self.assertRaises(ValueError):
            evaluate(baseline="scripted", split="train")
