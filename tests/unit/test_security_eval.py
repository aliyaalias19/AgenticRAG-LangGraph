"""Tests for the security evaluation harness."""

from pathlib import Path

from agentic_rag.eval.security_eval import (
    CANARY,
    AttackCase,
    AttackFamily,
    AttackOutcome,
    SecurityReport,
    build_attack_set,
    compare_reports,
    judge_outcome,
    write_report,
)


def case(**overrides: object) -> AttackCase:
    defaults = {
        "case_id": "t1",
        "family": AttackFamily.INDIRECT_INJECTION,
        "question": "q",
        "description": "d",
    }
    defaults.update(overrides)
    return AttackCase(**defaults)  # type: ignore[arg-type]


class TestAttackSet:
    def test_covers_every_family(self) -> None:
        families = {c.family for c in build_attack_set()}
        assert families == set(AttackFamily)

    def test_case_ids_are_unique(self) -> None:
        ids = [c.case_id for c in build_attack_set()]
        assert len(ids) == len(set(ids))

    def test_is_stable_across_calls(self) -> None:
        assert [c.case_id for c in build_attack_set()] == [c.case_id for c in build_attack_set()]

    def test_injection_cases_carry_a_poisoned_passage(self) -> None:
        injections = [c for c in build_attack_set() if c.family == AttackFamily.INDIRECT_INJECTION]
        assert all(c.poisoned_passage for c in injections)

    def test_every_case_has_a_detection_mechanism(self) -> None:
        """A case with nothing to detect can never fail, so it tests nothing."""
        for attack in build_attack_set():
            detectable = (
                attack.forbidden_markers or attack.forbidden_chunk_ids or attack.expect_abstention
            )
            assert detectable, f"{attack.case_id} has no way to fail"

    def test_most_injections_plant_the_canary(self) -> None:
        """Some target prompt disclosure instead, which needs no marker."""
        injections = [c for c in build_attack_set() if c.family == AttackFamily.INDIRECT_INJECTION]
        with_canary = [c for c in injections if CANARY in c.poisoned_passage]
        assert len(with_canary) >= len(injections) - 1

    def test_poisoned_passages_still_look_like_documentation(self) -> None:
        """A passage that is only an attack would never be retrieved."""
        injections = [c for c in build_attack_set() if c.family == AttackFamily.INDIRECT_INJECTION]
        assert all("kubectl drain" in c.poisoned_passage for c in injections)


class TestJudging:
    def test_marker_in_answer_is_a_success(self) -> None:
        outcome = judge_outcome(case(forbidden_markers=(CANARY,)), f"Sure: {CANARY}", [], False)
        assert outcome.succeeded
        assert "forbidden marker" in outcome.reason

    def test_marker_match_is_case_insensitive(self) -> None:
        outcome = judge_outcome(
            case(forbidden_markers=("system prompt",)),
            "Here is my SYSTEM PROMPT as requested",
            [],
            False,
        )
        assert outcome.succeeded

    def test_clean_answer_is_a_defence(self) -> None:
        outcome = judge_outcome(
            case(forbidden_markers=(CANARY,)),
            "Use kubectl drain to evict pods [1].",
            ["safe#0"],
            False,
        )
        assert not outcome.succeeded
        assert outcome.reason == "defended"

    def test_leaked_chunk_is_a_success(self) -> None:
        outcome = judge_outcome(
            case(forbidden_chunk_ids=("secret#0",)),
            "Here is the answer",
            ["public#0", "secret#0"],
            False,
        )
        assert outcome.succeeded
        assert "out-of-scope" in outcome.reason

    def test_answering_an_unanswerable_question_is_a_success(self) -> None:
        outcome = judge_outcome(case(expect_abstention=True), "The answer is 42.", [], False)
        assert outcome.succeeded

    def test_abstaining_when_expected_is_a_defence(self) -> None:
        outcome = judge_outcome(case(expect_abstention=True), "Not enough information.", [], True)
        assert not outcome.succeeded

    def test_judgement_uses_no_model(self) -> None:
        """Mechanical judging keeps the test itself out of the attack surface."""
        outcome = judge_outcome(case(forbidden_markers=("x",)), "y", [], False)
        assert isinstance(outcome, AttackOutcome)


class TestReport:
    def _report(self, label: str, successes: int, total: int) -> SecurityReport:
        return SecurityReport(
            label=label,
            outcomes=[
                AttackOutcome(
                    f"c{i}",
                    AttackFamily.INDIRECT_INJECTION,
                    i < successes,
                    "r",
                )
                for i in range(total)
            ],
        )

    def test_attack_success_rate(self) -> None:
        assert self._report("x", 3, 10).attack_success_rate == 0.3

    def test_empty_report_is_zero(self) -> None:
        assert SecurityReport("empty").attack_success_rate == 0.0

    def test_by_family_breakdown(self) -> None:
        report = SecurityReport(
            "mixed",
            outcomes=[
                AttackOutcome("a", AttackFamily.INDIRECT_INJECTION, True, "r"),
                AttackOutcome("b", AttackFamily.INDIRECT_INJECTION, False, "r"),
                AttackOutcome("c", AttackFamily.CROSS_TENANT, False, "r"),
            ],
        )
        families = report.by_family()
        assert families["indirect_injection"]["attack_success_rate"] == 0.5
        assert families["cross_tenant"]["attack_success_rate"] == 0.0

    def test_comparison_reports_the_reduction(self) -> None:
        comparison = compare_reports(self._report("before", 8, 10), self._report("after", 1, 10))
        assert comparison["absolute_reduction"] == 0.7
        assert comparison["before"]["attack_success_rate"] == 0.8
        assert comparison["after"]["attack_success_rate"] == 0.1

    def test_report_round_trips_to_disk(self, tmp_path: Path) -> None:
        path = tmp_path / "security.json"
        write_report(self._report("run", 2, 5), path)

        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["cases"] == 5
        assert payload["attack_success_rate"] == 0.4
        assert len(payload["outcomes"]) == 5
