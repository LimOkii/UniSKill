from __future__ import annotations

from pathlib import Path
import unittest

from uniskill.training.anchors import build_proposal_candidates, validate_candidate_roles
from uniskill.training.types import ActionStep, Trajectory
from uniskill.training.webshop_warmup import build_warmup_candidates


ROOT = Path(__file__).resolve().parents[1]


def trajectory(identifier: str, *, success: bool, query: str = "q") -> Trajectory:
    return Trajectory(
        query_id=query,
        trajectory_id=identifier,
        task_type="webshop",
        task_description="Buy a matching product",
        gamefile=None,
        episode_reward=10.0 if success else 0.0,
        success=success,
        end_reason="success" if success else "truncation",
        retrieved_skill=None,
        retrieval={},
        steps=[],
        environment="webshop",
    )


class WebshopHeldOutReferenceTest(unittest.TestCase):
    def build_group(self, successes: int, failures: int):
        return [
            trajectory(f"s{index}", success=True)
            for index in range(successes)
        ] + [
            trajectory(f"f{index}", success=False)
            for index in range(failures)
        ]

    def test_only_groups_with_two_of_each_outcome_generate_proposals(self):
        for successes, failures in ((6, 2), (5, 3), (4, 4), (3, 5), (2, 6)):
            with self.subTest(successes=successes, failures=failures):
                candidates, stats = build_proposal_candidates(
                    self.build_group(successes, failures), seed=13
                )
                self.assertEqual(len(candidates), 8)
                self.assertEqual(stats["held_out_reference_groups"], 1)
                self.assertEqual(
                    stats["insufficient_held_out_reference_groups"], 0
                )
                self.assertEqual(
                    stats[
                        f"outcome_split_{successes}s_{failures}f_groups"
                    ],
                    1,
                )

        for successes, failures in ((8, 0), (7, 1), (1, 7), (0, 8)):
            with self.subTest(successes=successes, failures=failures):
                candidates, stats = build_proposal_candidates(
                    self.build_group(successes, failures), seed=13
                )
                self.assertEqual(candidates, [])
                if successes and failures:
                    self.assertEqual(
                        stats["insufficient_held_out_reference_groups"], 1
                    )
                    self.assertEqual(
                        stats["skipped_sources_missing_held_out_anchors"], 8
                    )
                else:
                    self.assertEqual(stats["homogeneous_groups"], 1)

    def test_prompt_reference_is_excluded_from_both_alignment_anchors(self):
        candidates, _ = build_proposal_candidates(
            self.build_group(4, 4), seed=7
        )

        for candidate in candidates:
            roles = {
                "source": candidate.source,
                "opposite": candidate.opposite_reference,
                "success_anchor": candidate.success_anchor,
                "failure_anchor": candidate.failure_anchor,
            }
            self.assertEqual(
                len({item.trajectory_id for item in roles.values()}), 4
            )
            self.assertNotEqual(
                candidate.source.success,
                candidate.opposite_reference.success,
            )
            self.assertTrue(candidate.success_anchor.success)
            self.assertFalse(candidate.failure_anchor.success)
            record = candidate.to_record()
            self.assertEqual(
                record["opposite_reference_id"],
                candidate.opposite_reference.trajectory_id,
            )

    def test_first_step_format_warmup_does_not_require_four_trajectory_stats(self):
        group = self.build_group(1, 1)
        for source in group:
            source.steps.append(
                ActionStep(
                    step_id=f"{source.trajectory_id}-step-1",
                    query_id=source.query_id,
                    trajectory_id=source.trajectory_id,
                    step_index=1,
                    context={"current_observation": "Search page"},
                    response_ids=[1],
                    response_mask=[1],
                    action_token_mask=[1],
                    response_text="<action>search[shirt]</action>",
                    action_text="search[shirt]",
                    reward=0.0,
                    done=False,
                    is_action_valid=True,
                )
            )
        candidates, stats = build_warmup_candidates(group, seed=1, sample_size=2)
        self.assertEqual(len(candidates), 2)
        self.assertNotIn("four_trajectory_candidates", stats)
        validate_candidate_roles(candidates, stats, format_only_warmup=True)
        with self.assertRaisesRegex(RuntimeError, "four distinct roles"):
            validate_candidate_roles(candidates, stats, format_only_warmup=False)

    def test_main_route_requires_four_distinct_roles(self):
        candidates, stats = build_proposal_candidates(self.build_group(4, 4), seed=7)
        validate_candidate_roles(candidates, stats, format_only_warmup=False)
        stats["four_trajectory_candidates"] -= 1
        with self.assertRaisesRegex(RuntimeError, "four distinct roles"):
            validate_candidate_roles(candidates, stats, format_only_warmup=False)

    def test_training_script_keeps_main_experiment_settings(self):
        script = (ROOT / "scripts/train_webshop.sh").read_text()
        self.assertIn("+uniskill.webshop.strict_actor_credit_start_step=31", script)
        self.assertIn("+uniskill.r_align_warmup_steps=30", script)
        self.assertIn("+uniskill.retrieval.start_step=31", script)
        self.assertIn("trainer.total_epochs=250", script)


if __name__ == "__main__":
    unittest.main()
