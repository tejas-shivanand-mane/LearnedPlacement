import copy
import json
from pathlib import Path
import tempfile
import unittest

from learnedplacement.metrics import summarize, wilson_lower
from learnedplacement.model import Case, Node, Plan, violations
from learnedplacement.planners import METHODS, add_backups, initial
from learnedplacement.runner import ROOT, export_run, paired_report, read_json, run, status
from learnedplacement.simulator import Event, common_version, generate_trace, simulate


def fixture():
    nodes = [Node(1., 1., 10., 100., .1, i % 2) for i in range(6)]
    c = Case("tiny", nodes, replicas=2, shards=2, state_gib=2., working_gib=1.,
             steps=6, base_step_s=1., collective_factor=0., cross_domain_factor=1.,
             storage_gib_s=100., backup_every=2, checkpoint_every=100,
             backup_budget_gib=2., detection_s=0., reinit_s=0., replacement_s=2.,
             deadline_s=20., cost_budget_usd=1., max_simulated_s=50.)
    return c, Plan(((0, 1), (2, 3)))


class SimulatorTests(unittest.TestCase):
    def test_healthy_closed_form(self):
        c, p = fixture()
        r = simulate(c, p, [])
        self.assertEqual(r["completion_s"], 6.)
        self.assertAlmostEqual(r["cost_usd"], 4*6/3600)
        self.assertEqual(r["phase_seconds"]["recovery"], 0)

    def test_backup_overhead_and_spare_billing(self):
        c, p = fixture()
        q = Plan(p.groups, ((0, 4), (1, 5)))
        a, b = simulate(c, p, []), simulate(c, q, [])
        self.assertGreater(b["completion_s"], a["completion_s"])
        self.assertAlmostEqual(b["cost_usd"], 6*b["completion_s"]/3600)

    def test_surviving_group_no_rollback(self):
        c, p = fixture()
        r = simulate(c, p, [Event(2.5, 0, 0), Event(4.5, 1, 0)], audit=True)
        self.assertTrue(r["completed"])
        self.assertEqual(r["rollbacks"], 0)
        self.assertGreater(r["completion_s"], 6)
        self.assertEqual(r["aborted_steps"], 1)
        self.assertGreater(r["recovered_groups"], 0)
        self.assertAlmostEqual(sum(r["phase_seconds"].values()), r["observed_s"])

    def test_all_primary_state_lost_rolls_back(self):
        c, p = fixture()
        trace = sorted([Event(2.5, 0, n) for n in range(4)] + [Event(4.5, 1, n) for n in range(4)])
        r = simulate(c, p, trace, audit=True)
        self.assertTrue(r["completed"])
        self.assertEqual(r["lost_steps"], 2)
        self.assertEqual([x["version"] for x in r["audit"] if x["event"] == "rollback"], [0])

    def test_scattered_current_shards_are_recoverable(self):
        c, p = fixture()
        trace = sorted([Event(2.5, 0, n) for n in (0, 3)] + [Event(4.5, 1, n) for n in (0, 3)])
        r = simulate(c, p, trace)
        self.assertTrue(r["completed"])
        self.assertEqual(r["rollbacks"], 0)

    def test_backups_restore_common_older_version(self):
        c, p = fixture()
        trace = sorted([Event(3.5, 0, n) for n in range(4)] + [Event(5.5, 1, n) for n in range(4)])
        a = simulate(c, p, trace)
        b = simulate(c, Plan(p.groups, ((0, 4), (1, 5))), trace, audit=True)
        self.assertEqual(a["lost_steps"], 3)
        self.assertEqual(b["lost_steps"], 1)
        self.assertEqual([x["version"] for x in b["audit"] if x["event"] == "rollback"], [2])

    def test_persistent_checkpoint_restore(self):
        c, p = fixture()
        c.checkpoint_every = 2
        trace = sorted([Event(3.5, 0, n) for n in range(4)] + [Event(5.5, 1, n) for n in range(4)])
        r = simulate(c, p, trace, audit=True)
        self.assertEqual(r["lost_steps"], 1)
        self.assertEqual([x["version"] for x in r["audit"] if x["event"] == "rollback"], [2])

    def test_mixed_versions_cannot_be_combined(self):
        self.assertEqual(common_version([{0, 10}, {0, 12}], 12), 0)
        self.assertIsNone(common_version([{10}, {12}], 12))
        self.assertEqual(common_version([{0, 10, 12}, {0, 10}], 12), 10)

    def test_failure_during_backup_does_not_commit_partial_snapshot(self):
        c, p = fixture()
        c.steps = 3
        q = Plan(p.groups, ((0, 4), (1, 5)))
        # Initial snapshot completes at .01; version-2 refresh starts at 2.01.
        r = simulate(c, q, [Event(2.015, 0, 4), Event(10, 1, 4)], audit=True)
        commits = [x for x in r["audit"] if x["event"] == "backup" and x["version"] == 2]
        self.assertEqual(len(commits), 1)
        self.assertEqual(commits[0]["copies"], 1)
        self.assertGreater(commits[0]["time_s"], 2.015)

    def test_failure_during_recovery_retries(self):
        c, p = fixture()
        c.reinit_s = 2.
        trace = [Event(.5, 0, 0), Event(1., 1, 0), Event(3., 0, 2), Event(4., 1, 2)]
        r = simulate(c, p, trace)
        self.assertTrue(r["completed"])
        self.assertEqual(r["failures"], 2)
        self.assertAlmostEqual(sum(r["phase_seconds"].values()), r["observed_s"])

    def test_overlap_outages_do_not_restore_too_early(self):
        c, p = fixture()
        r = simulate(c, p, [Event(.5, 0, 0), Event(1., 0, 0), Event(2., 1, 0), Event(3., 1, 0)], audit=True)
        replacements = [x["time_s"] for x in r["audit"] if x["event"] == "replacement"]
        self.assertEqual(replacements, [3.])

    def test_censored_runs_not_reported_as_finished(self):
        c, p = fixture()
        c.max_simulated_s, c.deadline_s = 5., 4.
        r = simulate(c, p, [])
        self.assertFalse(r["completed"])
        self.assertIsNone(r["completion_s"])
        s = summarize(c, [r, r])
        self.assertIsNone(s["mean_completion_s"])
        self.assertIsNone(s["mean_cost_usd"])
        self.assertFalse(s["statistical_screen_passed"])

    def test_memory_checks_include_staging(self):
        c, p = fixture()
        c.nodes[1] = Node(1, 1, 3., 100, .1, 1)
        self.assertIn("node memory exceeded (includes backup staging buffer)",
                      violations(c, Plan(p.groups, ((0, 1),))))

    def test_duplicate_assignment_and_budget_rejected(self):
        c, p = fixture()
        self.assertTrue(violations(c, Plan(((0, 1), (0, 3)))))
        c.backup_budget_gib = 0
        self.assertTrue(violations(c, Plan(p.groups, ((0, 4),))))

    def test_trace_seed_reproducibility_and_separation(self):
        c, _ = fixture()
        c.domain_failure_per_hour = 1000
        self.assertEqual(generate_trace(c, 1), generate_trace(c, 1))
        self.assertNotEqual(generate_trace(c, 1), generate_trace(c, 2))

    def test_wilson_is_not_perfect_reliability(self):
        self.assertLess(wilson_lower(256, 256), 1)
        self.assertGreater(wilson_lower(256, 256), .95)
        self.assertLess(wilson_lower(32, 32), .95)

    def test_baseline_structure(self):
        c, _ = fixture()
        for kind in METHODS[:4]:
            self.assertEqual(violations(c, add_backups(c, initial(c, kind))), [])


class RunnerTests(unittest.TestCase):
    """Tiny two-step/two-trace fixtures only, not the requested experiment."""
    def config(self):
        config = read_json(ROOT/"configs"/"pilot.json")
        config.update(active_nodes=[4], regimes=["homogeneous"], instances_per_regime=1,
                      planning_traces=2, evaluation_traces=2, greedy_candidates=2,
                      local_candidates=2, greedy_seconds=.2, local_seconds=.2,
                      save_every_traces=1)
        config["workload"].update(shards=2, steps=2, base_step_s=.1)
        return config

    def test_pause_resume_export_and_ablation(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)/"run"
            config = self.config()
            result = run(output=out, config=config, max_minutes=1e-9)
            self.assertEqual(result["state"], "paused_time_limit")
            result = run(output=out, config=config, max_minutes=1)
            self.assertEqual(result["completed_comparisons"], len(METHODS))
            plans = read_json(out/"cases"/"n4-homogeneous-i0"/"plans.json")
            self.assertEqual(plans["joint_local"]["plan"]["groups"], plans["joint_local_no_backup"]["plan"]["groups"])
            self.assertEqual(plans["joint_local_no_backup"]["plan"]["backups"], [])
            before = (out/"cases"/"n4-homogeneous-i0"/"plans.json").read_bytes()
            run(output=out, config=config, max_minutes=1)
            self.assertEqual(before, (out/"cases"/"n4-homogeneous-i0"/"plans.json").read_bytes())
            comparison = paired_report(out, repeats=20)
            self.assertEqual(comparison[0]["time_difference_s"], 0.)
            self.assertTrue(export_run(out).exists())

    def test_changed_configuration_cannot_mix_results(self):
        with tempfile.TemporaryDirectory() as temp:
            config = self.config()
            run(output=temp, config=config, max_minutes=1e-9)
            config["seed"] += 1
            with self.assertRaisesRegex(ValueError, "NEW output"):
                run(output=temp, config=config, max_minutes=1)

    def test_lock_is_not_removed_by_second_runner(self):
        with tempfile.TemporaryDirectory() as temp:
            lock = Path(temp)/".run.lock"
            lock.write_text("123")
            with self.assertRaisesRegex(RuntimeError, "locked"):
                run(output=temp, config=self.config())
            self.assertTrue(lock.exists())


if __name__ == "__main__":
    unittest.main()
