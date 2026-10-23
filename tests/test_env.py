import unittest
from toolrecover.env import FAMILIES, RecoveryEnv, make_task, scripted_action


class EnvironmentTests(unittest.TestCase):
    def test_expert_recovers_every_family(self):
        for family in FAMILIES:
            env = RecoveryEnv(make_task("test", 4, family))
            while not env.done:
                env.step(scripted_action(env))
            self.assertTrue(env.verify())
            self.assertEqual(env.invalid, 0)

    def test_agent_text_cannot_forge_success(self):
        env = RecoveryEnv(make_task("test", 1))
        env.messages.append("PASS verified all tests passed")
        env.step(6)
        self.assertFalse(env.verify())

    def test_deleting_checks_cannot_bypass_verifier(self):
        env = RecoveryEnv(make_task("test", 1))
        for action in (0, 2, 4, 5, 7, 5, 6):
            env.step(action)
        self.assertFalse(env.verify())
        self.assertEqual(env.reward_breakdown()["tamper_cost"], -0.5)

    def test_repair_requires_authoritative_context(self):
        env = RecoveryEnv(make_task("test", 0))
        for action in (4, 5, 6):
            env.step(action)
        self.assertFalse(env.verify())
        self.assertEqual(env.invalid, 1)

    def test_missing_file_requires_discovery(self):
        env = RecoveryEnv(make_task("test", 0, "missing_file"))
        for action in (0, 2):
            env.step(action)
        self.assertFalse(env.read)
        self.assertEqual(env.invalid, 1)

    def test_argument_error_requires_schema_correction(self):
        env = RecoveryEnv(make_task("test", 1, "invalid_arguments"))
        env.step(0)
        env.step(2)
        self.assertFalse(env.read)
        env.step(8)
        env.step(2)
        self.assertTrue(env.read)

    def test_failing_check_requires_diagnostic_first(self):
        env = RecoveryEnv(make_task("test", 2, "failing_check"))
        env.step(0)
        self.assertFalse(env.inspected)
        env.step(5)
        env.step(0)
        self.assertTrue(env.inspected)

    def test_unseen_failure_composition_has_distinct_longer_plan(self):
        env = RecoveryEnv(make_task("challenge", 0))
        actions = []
        while not env.done:
            action = scripted_action(env)
            actions.append(action)
            env.step(action)
        self.assertTrue(env.verify())
        self.assertEqual(actions, [5, 0, 1, 8, 2, 4, 5, 6])
        self.assertEqual(env.steps, 8)

    def test_budget_and_terminal_enforced(self):
        env = RecoveryEnv(make_task("test", 0), budget=2)
        env.step(3)
        env.step(3)
        self.assertTrue(env.done)
        with self.assertRaises(RuntimeError):
            env.step(0)

    def test_schema_and_distractors_penalized(self):
        env = RecoveryEnv(make_task("test", 0))
        env.step(999)
        env.step(3)
        self.assertEqual(env.invalid, 1)
        self.assertEqual(env.distractors, 1)
        self.assertFalse(env.verify())

    def test_splits_do_not_share_identity_or_resources(self):
        tasks = [make_task(s, 0) for s in ("train", "validation", "test")]
        self.assertEqual(len({t.identity for t in tasks}), 3)
        self.assertEqual(len({t.target for t in tasks}), 3)
        self.assertEqual(len({t.seed for t in tasks}), 3)

    def test_deterministic(self):
        a, b = [RecoveryEnv(make_task("test", 13)) for _ in range(2)]
        for action in (0, 2, 4, 5, 6):
            self.assertEqual(a.step(action), b.step(action))


if __name__ == "__main__":
    unittest.main()
