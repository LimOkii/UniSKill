from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

from uniskill.environments.prompts.registry import render_action_prompt
from uniskill.training.types import CriticState, ProposalCandidate, Trajectory


class DebugTracePrinter:
    """Print a small, complete UniSkill dataflow sample on the trainer driver."""

    def __init__(self, settings, *, tokenizer, apply_chat_template_kwargs=None):
        self.settings = settings
        self.tokenizer = tokenizer
        self.chat_kwargs = apply_chat_template_kwargs or {}

    @property
    def enabled(self) -> bool:
        return bool(self.settings.enabled and self.settings.samples_per_batch > 0)

    def print_iteration(
        self,
        *,
        iteration: int,
        trajectories: list[Trajectory],
        candidates: list[ProposalCandidate],
        anchor_stats: dict[str, int],
        commit_results: Sequence[Any],
        coverage_stats: dict[str, float] | None = None,
    ) -> None:
        if not self.enabled:
            return

        print(
            f"[UNISKILL ACTION-PAYLOAD COVERAGE] iteration={iteration} "
            f"stats={coverage_stats or {}}",
            flush=True,
        )

        result_by_id = {result.proposal_id: result for result in commit_results}
        selected = self._select_candidates(candidates)
        if selected:
            for sample_index, candidate in enumerate(selected):
                self._print_candidate(
                    iteration=iteration,
                    sample_index=sample_index,
                    candidate=candidate,
                    trajectories=trajectories,
                    commit_result=result_by_id.get(candidate.proposal_id),
                )
            return

        for sample_index, trajectory in enumerate(
            sorted(trajectories, key=lambda item: (item.query_id, item.trajectory_id))[
                : self.settings.samples_per_batch
            ]
        ):
            self._begin(iteration, sample_index)
            self._print_query(trajectory, trajectories)
            print("proposal_pipeline=SKIPPED", flush=True)
            print(f"skip_reason={self._skip_reason(trajectory, trajectories)}", flush=True)
            print(f"anchor_selection_stats={anchor_stats}", flush=True)
            self._print_action_steps(trajectory)
            print("proposal_policy_loss_labels=NONE (no eligible proposal candidate)", flush=True)
            self._end(iteration, sample_index)

    def _select_candidates(self, candidates: list[ProposalCandidate]) -> list[ProposalCandidate]:
        def rank(candidate: ProposalCandidate):
            return (
                candidate.critic_action_reasonable is True,
                not candidate.is_masked,
                candidate.critic_state is CriticState.ACCEPT,
                candidate.r_align is not None,
                candidate.proposal_id,
            )

        return sorted(candidates, key=rank, reverse=True)[: self.settings.samples_per_batch]

    def _print_candidate(
        self,
        *,
        iteration: int,
        sample_index: int,
        candidate: ProposalCandidate,
        trajectories: list[Trajectory],
        commit_result: Any | None,
    ) -> None:
        self._begin(iteration, sample_index)
        self._print_query(candidate.source, trajectories)
        print(f"source_trajectory_id={candidate.source.trajectory_id}", flush=True)
        print(f"success_anchor_id={getattr(candidate.success_anchor, 'trajectory_id', None)}", flush=True)
        print(f"failure_anchor_id={getattr(candidate.failure_anchor, 'trajectory_id', None)}", flush=True)
        print(
            "anchor_action_payload_coverage="
            f"success:{candidate.success_anchor_coverage!r} "
            f"failure:{candidate.failure_anchor_coverage!r}",
            flush=True,
        )
        self._print_action_steps(candidate.source)

        print("\n--- SKILL PROPOSAL USER PROMPT (FULL) ---", flush=True)
        print(candidate.prompt, flush=True)
        print("--- SKILL PROPOSAL MODEL PROMPT (FULL) ---", flush=True)
        print(self._chat_prompt(candidate.prompt), flush=True)
        print(
            "proposal_prompt_tokenization="
            f"raw_tokens:{candidate.prompt_token_length} "
            f"actual_input_tokens:{len(candidate.model_prompt_ids)} "
            f"left_truncated:{candidate.prompt_truncated}",
            flush=True,
        )
        print("SKILL PROPOSAL ACTUAL MODEL INPUT AFTER TOKEN TRUNCATION:", flush=True)
        if self.settings.show_token_ids:
            print(f"model_prompt_token_ids={candidate.model_prompt_ids}", flush=True)
        print(self._decode(candidate.model_prompt_ids), flush=True)
        print("--- SKILL PROPOSAL GENERATED RESPONSE (FULL) ---", flush=True)
        print(candidate.response_text, flush=True)
        print(
            "proposal_response_tokenization="
            f"tokens:{candidate.response_token_length} "
            f"hit_length_limit:{candidate.response_hit_length_limit} "
            f"truncated:{candidate.response_truncated}",
            flush=True,
        )
        print(
            "proposal_parse="
            f"ok:{candidate.parse_ok} action:{candidate.proposal_action!r} "
            f"error:{candidate.parse_error!r} skill:{candidate.skill_text!r}",
            flush=True,
        )
        print(
            "local_proposal_routing="
            f"reason:{candidate.local_routing_reason!r} "
            f"exact_duplicate:{candidate.exact_duplicate_source!r} "
            f"critic_called:{bool(candidate.critic_prompt)}",
            flush=True,
        )
        print(
            "anchor_coverage="
            f"success:{candidate.success_anchor_coverage!r} "
            f"failure:{candidate.failure_anchor_coverage!r} "
            f"minimum:{candidate.minimum_anchor_coverage!r} "
            f"required:{candidate.anchor_coverage_threshold!r}",
            flush=True,
        )
        print(
            "skill_critic="
            f"state:{candidate.critic_state.value} "
            f"action_reasonable:{candidate.critic_action_reasonable!r} "
            f"action_reason:{candidate.critic_action_reason!r} "
            f"content_supported:{candidate.critic_content_supported!r} "
            f"content_reason:{candidate.critic_content_reason!r} "
            f"error:{candidate.critic_error!r} attempts:{candidate.critic_attempts}",
            flush=True,
        )
        if candidate.critic_prompt:
            print("--- SKILL CRITIC PROMPT (FULL) ---", flush=True)
            print(candidate.critic_prompt, flush=True)
            print("--- SKILL CRITIC RAW RESPONSE (FULL) ---", flush=True)
            print(candidate.critic_raw_response, flush=True)
        else:
            print("skill_critic_api=NOT_CALLED (proposal was routed locally)", flush=True)

        if (
            candidate.critic_content_supported is True
            and candidate.proposal_action in {"ADD_NEW_SKILL", "UPDATE_SKILL"}
            and not candidate.exact_duplicate_source
        ):
            skill = {"skill_text": candidate.skill_text}
            self._print_anchor("SUCCESS", candidate.success_anchor, skill)
            self._print_anchor("FAILURE", candidate.failure_anchor, skill)
        elif candidate.proposal_action == "NO_SKILL" and candidate.critic_action_reasonable is True:
            print(
                "counterfactual_teacher_forcing=SKIPPED "
                "(critic accepted NO_SKILL; no candidate skill to score)",
                flush=True,
            )
        elif candidate.exact_duplicate_source:
            print(
                "counterfactual_teacher_forcing=SKIPPED "
                f"(exact_duplicate={candidate.exact_duplicate_source})",
                flush=True,
            )
        else:
            print(
                "counterfactual_teacher_forcing=SKIPPED "
                f"(critic_state={candidate.critic_state.value})",
                flush=True,
            )

        print("\n--- REWARD AND WRITE ROUTING ---", flush=True)
        print(f"delta_success={candidate.delta_success!r}", flush=True)
        print(f"delta_failure={candidate.delta_failure!r}", flush=True)
        print(f"R_align={candidate.r_align!r}", flush=True)
        print(
            "proposal_channel_rewards="
            f"format:{candidate.format_reward!r} action:{candidate.action_reward!r} "
            f"skill:{candidate.skill_reward!r}",
            flush=True,
        )
        print(f"write_eligible={candidate.is_write_eligible}", flush=True)
        if commit_result is None:
            write_status = "not_eligible"
        else:
            write_status = commit_result.status
        print(f"write_status={write_status}", flush=True)

        print("\n--- PROPOSAL REINFORCE++ POLICY LOSS LABELS ---", flush=True)
        if not candidate.included_in_policy_loss:
            reason = (
                "candidate is masked"
                if candidate.is_masked
                else "proposal training batch was skipped"
            )
            print(f"NONE ({reason}; no proposal RL loss)", flush=True)
        else:
            for channel in ("format", "action", "skill"):
                print(f"{channel}_token_labels:", flush=True)
                self._print_token_labels(
                    candidate.response_ids,
                    getattr(candidate, f"{channel}_token_mask"),
                )
        self._end(iteration, sample_index)

    def _print_query(self, source: Trajectory, trajectories: list[Trajectory]) -> None:
        group = [item for item in trajectories if item.query_id == source.query_id]
        outcomes = Counter("success" if item.success else "failure" for item in group)
        print(f"query_id={source.query_id}", flush=True)
        query_label = (
            "webshop_query" if source.environment == "webshop" else "alfworld_query"
        )
        print(f"{query_label}={source.task_description}", flush=True)
        print(f"task_type={source.task_type}", flush=True)
        print(f"group_outcomes={dict(outcomes)}", flush=True)

    def _print_action_steps(self, trajectory: Trajectory) -> None:
        selected = self._select_steps(trajectory.steps)
        print(
            f"\n--- ACTION MODE SOURCE TRAJECTORY sampled_steps={len(selected)}/"
            f"{len(trajectory.steps)} ---",
            flush=True,
        )
        for step in selected:
            user_prompt = render_action_prompt(step.context, step.context.get("original_skill"))
            print(f"\n[ACTION STEP {step.step_index}] step_id={step.step_id}", flush=True)
            history_steps = len(step.context.get("history") or [])
            print(
                "ACTION CONTEXT MEMORY: "
                f"recent_history_steps={history_steps}",
                flush=True,
            )
            print(
                f"action_model_prompt_token_count={self._chat_prompt_token_count(user_prompt)!r}",
                flush=True,
            )
            print("ACTION USER PROMPT (FULL):", flush=True)
            print(user_prompt, flush=True)
            print("ACTION MODEL PROMPT (FULL):", flush=True)
            print(self._chat_prompt(user_prompt), flush=True)
            print("ACTION GENERATED RESPONSE (FULL):", flush=True)
            print(step.response_text, flush=True)
            print(
                "ACTION FORMAT/EXECUTION STATUS: "
                f"native_format_valid={step.is_action_valid} "
                f"strict_format_valid={step.is_action_strict_format_valid} "
                f"strict_format_error={step.action_strict_format_error!r} "
                f"payload_parsed={step.action_payload_parsed} "
                f"admissible={step.is_action_admissible} "
                f"overall_valid={step.is_action_overall_valid} "
                f"response_token_length={step.response_token_length} "
                f"response_hit_length_limit={step.response_hit_length_limit} "
                f"response_truncated={step.response_truncated} "
                f"canonical_action={step.action_display!r} "
                f"history_status={step.action_history_status}",
                flush=True,
            )
            if trajectory.environment == "webshop":
                print(
                    "ACTION POLICY LOSS LABELS "
                    "(all generated non-padding response tokens):",
                    flush=True,
                )
            else:
                print("ACTION POLICY LOSS LABELS (full valid response):", flush=True)
            self._print_token_labels(step.response_ids, step.response_mask)
            print("COUNTERFACTUAL ACTION-PAYLOAD MASK FOR THIS RESPONSE:", flush=True)
            self._print_token_labels(step.response_ids, step.action_token_mask)

    def _print_anchor(self, name: str, trajectory: Trajectory, skill: dict[str, str]) -> None:
        print(
            f"\n--- {name} ANCHOR COUNTERFACTUAL TEACHER FORCING "
            f"trajectory_id={trajectory.trajectory_id} steps={len(trajectory.steps)} ---",
            flush=True,
        )
        for step in trajectory.steps:
            print(f"[{name} ANCHOR STEP {step.step_index}] step_id={step.step_id}", flush=True)
            print(f"action_payload_text={step.action_text!r}", flush=True)
            if step is trajectory.steps[0]:
                scorer_prompt = render_action_prompt(step.context, skill)
                print("REPRESENTATIVE CANDIDATE-INJECTED USER PROMPT (FULL):", flush=True)
                print(scorer_prompt, flush=True)
                print("REPRESENTATIVE CANDIDATE-INJECTED MODEL PROMPT (FULL):", flush=True)
                print(self._chat_prompt(scorer_prompt), flush=True)
            print("COUNTERFACTUAL SCORE LABELS (only <action> payload):", flush=True)
            self._print_token_labels(step.response_ids, step.action_token_mask)

    def _select_steps(self, steps: Sequence[Any]) -> list[Any]:
        limit = int(self.settings.max_action_steps)
        if limit < 0 or limit >= len(steps):
            return list(steps)
        if limit == 0 or not steps:
            return []
        if limit == 1:
            return [steps[0]]
        indices = [round(index * (len(steps) - 1) / (limit - 1)) for index in range(limit)]
        return [steps[index] for index in dict.fromkeys(indices)]

    def _print_token_labels(self, token_ids: Sequence[int], mask: Sequence[int]) -> None:
        positions = [index for index, keep in enumerate(mask) if keep and index < len(token_ids)]
        selected_ids = [int(token_ids[index]) for index in positions]
        print(f"label_count={len(selected_ids)}", flush=True)
        print(f"label_positions={positions}", flush=True)
        if self.settings.show_token_ids:
            print(f"label_token_ids={selected_ids}", flush=True)
            try:
                pieces = self.tokenizer.convert_ids_to_tokens(selected_ids)
            except (AttributeError, TypeError, ValueError):
                pieces = [self._decode([token_id]) for token_id in selected_ids]
            print(f"label_token_pieces={pieces!r}", flush=True)
        print(f"decoded_label_text={self._decode(selected_ids)!r}", flush=True)

    def _chat_prompt(self, prompt: str) -> str:
        try:
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
                **self.chat_kwargs,
            )
        except Exception as error:  # Debug output must not abort a training iteration.
            return f"<chat-template rendering failed: {error!r}>"

    def _chat_prompt_token_count(self, prompt: str) -> int | None:
        try:
            token_ids = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=True,
                **self.chat_kwargs,
            )
            if hasattr(token_ids, "numel"):
                return int(token_ids.numel())
            if (
                isinstance(token_ids, (list, tuple))
                and token_ids
                and isinstance(token_ids[0], (list, tuple))
            ):
                return len(token_ids[0])
            if not isinstance(token_ids, str):
                return len(token_ids)
        except Exception:
            pass

        try:
            return len(
                self.tokenizer.encode(
                    self._chat_prompt(prompt),
                    add_special_tokens=False,
                )
            )
        except Exception:
            return None

    def _decode(self, token_ids: Sequence[int]) -> str:
        try:
            return self.tokenizer.decode(
                list(token_ids),
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            return self.tokenizer.decode(list(token_ids), skip_special_tokens=False)

    @staticmethod
    def _skip_reason(source: Trajectory, trajectories: list[Trajectory]) -> str:
        group = [item for item in trajectories if item.query_id == source.query_id]
        successes = sum(item.success for item in group)
        failures = len(group) - successes
        if not successes or not failures:
            return "homogeneous_query_group"
        if (source.success and successes == 1) or (not source.success and failures == 1):
            return "source_has_no_same-outcome-held-out-anchor"
        return "no_eligible_candidate"

    @staticmethod
    def _begin(iteration: int, sample_index: int) -> None:
        print(
            "\n================ UNISKILL DEBUG BEGIN "
            f"iteration={iteration} sample={sample_index} ================",
            flush=True,
        )

    @staticmethod
    def _end(iteration: int, sample_index: int) -> None:
        print(
            "================ UNISKILL DEBUG END "
            f"iteration={iteration} sample={sample_index} ================\n",
            flush=True,
        )
