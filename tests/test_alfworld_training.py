from __future__ import annotations

from pathlib import Path
import unittest

from uniskill.critic.prompts import build_critic_prompt
from uniskill.environments.history import build_recent_action_history_context
from uniskill.environments.prompts.alfworld import render_action_prompt
from uniskill.training.anchors import build_proposal_candidates
from uniskill.training.types import Trajectory


ROOT = Path(__file__).resolve().parents[1]


def trajectory(identifier: str, *, success: bool, query: str = "q") -> Trajectory:
    return Trajectory(
        query_id=query,
        trajectory_id=identifier,
        task_type="pick_and_place_simple",
        task_description="put an apple on countertop.",
        gamefile="game.tw-pddl",
        episode_reward=10.0 if success else 0.0,
        success=success,
        end_reason="success" if success else "truncation",
        retrieved_skill={"skill_id": "skill-1", "skill_text": "Find the apple."},
        retrieval={},
        steps=[],
        environment="alfworld/AlfredTWEnv",
    )


class AlfworldHeldOutReferenceTest(unittest.TestCase):
    def build_group(self, successes: int, failures: int) -> list[Trajectory]:
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
                    self.build_group(successes, failures),
                    seed=13,
                    expected_environment="alfworld/AlfredTWEnv",
                )
                self.assertEqual(len(candidates), 8)
                self.assertEqual(stats["held_out_reference_groups"], 1)
                self.assertEqual(
                    stats["insufficient_held_out_reference_groups"], 0
                )

        for successes, failures in ((8, 0), (7, 1), (1, 7), (0, 8)):
            with self.subTest(successes=successes, failures=failures):
                candidates, stats = build_proposal_candidates(
                    self.build_group(successes, failures),
                    seed=13,
                    expected_environment="alfworld/AlfredTWEnv",
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

    def test_four_roles_are_distinct_and_prompt_contains_both_inputs(self):
        candidates, _ = build_proposal_candidates(
            self.build_group(4, 4),
            seed=7,
            expected_environment="alfworld/AlfredTWEnv",
        )

        for candidate in candidates:
            roles = {
                candidate.source.trajectory_id,
                candidate.opposite_reference.trajectory_id,
                candidate.success_anchor.trajectory_id,
                candidate.failure_anchor.trajectory_id,
            }
            self.assertEqual(len(roles), 4)
            self.assertNotEqual(
                candidate.source.success,
                candidate.opposite_reference.success,
            )
            self.assertTrue(candidate.success_anchor.success)
            self.assertFalse(candidate.failure_anchor.success)
            self.assertIn("CURRENT_EPISODE:", candidate.prompt)
            self.assertIn("OPPOSITE_OUTCOME_REFERENCE:", candidate.prompt)
            self.assertEqual(
                candidate.to_record()["opposite_reference_id"],
                candidate.opposite_reference.trajectory_id,
            )
            self.assertEqual(
                candidate.to_record()["proposal_reference_mode"],
                "held_out_opposite",
            )
            self.assertTrue(
                candidate.to_record()["four_trajectory_roles_distinct"]
            )

    def test_fails_fast_when_rollout_environment_tag_is_missing(self):
        group = self.build_group(4, 4)
        group[0].environment = None
        with self.assertRaisesRegex(RuntimeError, "environment metadata"):
            build_proposal_candidates(
                group,
                seed=7,
                expected_environment="alfworld/AlfredTWEnv",
            )

    def test_critic_receives_opposite_outcome_reference(self):
        source = trajectory("source", success=True).to_record()
        opposite = trajectory("opposite", success=False).to_record()
        prompt = build_critic_prompt(
            source,
            {"proposal_action": "UPDATE_SKILL", "skill_text": "Search cabinets."},
            comparison_record=opposite,
        )
        self.assertIn("`OPPOSITE_OUTCOME_REFERENCE`", prompt)
        self.assertIn('"success": false', prompt)

    def test_actor_prompt_shows_only_two_recent_turns(self):
        memory = [
            {"text_obs": "at shelf", "action": "examine shelf 1"},
            {"text_obs": "at desk", "action": "go to desk 1"},
            {"text_obs": "apple visible", "action": "take apple 1"},
        ]
        history = build_recent_action_history_context(memory, history_length=2)
        prompt = render_action_prompt(
            {
                "task_description": "put apple in bowl",
                "step_count": 3,
                "current_step": 4,
                "history": history,
                "current_observation": "holding apple",
                "admissible_actions": ["go to bowl 1"],
            },
            None,
        )
        self.assertEqual([item["step"] for item in history], [2, 3])
        self.assertNotIn("examine shelf 1", prompt)
        self.assertIn("Action 2: 'go to desk 1'", prompt)
        self.assertIn("Action 3: 'take apple 1'", prompt)

    def test_training_script_uses_four_trajectory_route(self):
        script = (ROOT / "scripts/train_alfworld.sh").read_text()
        self.assertNotIn("proposal_reference_mode=source_only", script)
        self.assertIn("+algorithm.credit_assignment=true", script)
        self.assertIn("+algorithm.step_gamma=0.95", script)
        self.assertIn("trainer.total_epochs=250", script)


if __name__ == "__main__":
    unittest.main()
