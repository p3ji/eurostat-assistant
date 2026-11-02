from __future__ import annotations

import unittest

from app.agent_verification import (
    extract_prose_numerical_claims,
    extract_row_observations,
    verify_answer_grounding,
)


class AgentVerificationTests(unittest.TestCase):
    def test_extract_prose_numerical_claims_filters_metadata(self) -> None:
        text = (
            "In 2024, Spain's unemployment was 11.5%, ranking 1st ahead of France at 7.3%. "
            "See [une_rt_a](https://ec.europa.eu/eurostat/databrowser/view/une_rt_a/default/table?lang=en) "
            "and `nama_10_gdp`. CO2 intensity was low. For activity C10-12:\n"
            "1. First point\n"
            "2. Second point (index 2005=100)."
        )
        claims = extract_prose_numerical_claims(text)
        self.assertEqual(claims, ["11.5%", "7.3%"])

    def test_extract_row_observations_collects_numeric_fields(self) -> None:
        rows = [
            {"geo": "ES", "time": "2024", "value": 11.52, "unit": "PC"},
            {"geo": "FR", "time": "2024", "value": 7.3, "unit": "PC"},
            {"geo": "DE", "time": "2024", "x_value": 24.5, "y_value": 110.2},
        ]
        observations = extract_row_observations(rows)
        self.assertIn(11.52, observations)
        self.assertIn(7.3, observations)
        self.assertIn(24.5, observations)
        self.assertIn(110.2, observations)

    def test_grounded_numbers_pass_without_warning(self) -> None:
        result = {
            "dataset": {"code": "une_rt_a"},
            "rows": [
                {"geo": "ES", "time": "2024", "value": 11.5},
                {"geo": "FR", "time": "2024", "value": 7.3},
            ],
            "warnings": [],
        }
        answer = "Spain's unemployment rate in 2024 was 11.5% compared to France at 7.3%."
        verified_result, confidence = verify_answer_grounding(
            answer,
            result,
            confidence="high",
            tool_truncated=False,
        )
        self.assertEqual(confidence, "high")
        self.assertEqual(verified_result.get("warnings"), [])

    def test_ungrounded_numbers_trigger_warning_and_downgrade_confidence(self) -> None:
        result = {
            "dataset": {"code": "une_rt_a"},
            "rows": [
                {"geo": "ES", "time": "2024", "value": 11.5},
                {"geo": "FR", "time": "2024", "value": 7.3},
            ],
            "warnings": [],
        }
        # The model claims 28.4%, which is nowhere in the data
        answer = "Spain's youth unemployment reached 28.4% in 2024."
        verified_result, confidence = verify_answer_grounding(
            answer,
            result,
            confidence="high",
            tool_truncated=False,
        )
        self.assertEqual(confidence, "medium")
        warnings = verified_result.get("warnings", [])
        self.assertEqual(len(warnings), 1)
        self.assertIn("ungrounded numerical values (28.4%)", warnings[0])
        self.assertIn("authoritative source", warnings[0])

    def test_truncation_warning_included_when_tool_truncated(self) -> None:
        result = {
            "dataset": {"code": "une_rt_a"},
            "rows": [{"geo": "ES", "value": 11.5}],
            "warnings": [],
        }
        answer = "The rate for Poland was 3.2%."
        verified_result, confidence = verify_answer_grounding(
            answer,
            result,
            confidence="high",
            tool_truncated=True,
        )
        self.assertEqual(confidence, "medium")
        warnings = verified_result.get("warnings", [])
        self.assertEqual(len(warnings), 2)
        self.assertIn("truncated in the model's working context", warnings[1])

    def test_metadata_counts_and_filters_are_accepted(self) -> None:
        result = {
            "dataset": {"code": "une_rt_a"},
            "filters": {"lastTimePeriod": ["8"]},
            "rows": [
                {"geo": "ES", "value": 11.5},
                {"geo": "PT", "value": 6.5},
            ],
            "warnings": [],
        }
        answer = "Across the 8 years and 2 countries, Spain stayed above Portugal."
        verified_result, confidence = verify_answer_grounding(
            answer,
            result,
            confidence="high",
            tool_truncated=False,
        )
        self.assertEqual(confidence, "high")
        self.assertEqual(verified_result.get("warnings"), [])


if __name__ == "__main__":
    unittest.main()
