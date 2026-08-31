from pathlib import Path
import re
import unittest


class WorkflowContractTest(unittest.TestCase):
    @staticmethod
    def _production_workflow() -> str:
        return (Path(__file__).parents[1] / ".github" / "workflows" / "post.yml").read_text(encoding="utf-8")

    def test_production_workflow_uses_utf16_caption_limit(self):
        workflow = self._production_workflow()
        self.assertIn('TG_CAPTION_MAX_UTF16_UNITS: "1000"', workflow)
        self.assertNotIn("TG_CAPTION_MAX_BYTES:", workflow)

    def test_tip_of_day_scan_capacity_covers_incident_candidate_universe(self):
        workflow = self._production_workflow()
        match = re.search(r'^\s*MAX_CANDIDATES_PER_RUBRIC:\s*"(\d+)"\s*$', workflow, re.MULTILINE)
        self.assertIsNotNone(match)
        scan_cap = int(match.group(1))

        # Production incident #456 exposed 36 ordered tip_of_day candidates.
        current_tip_of_day_candidate_universe = 36
        self.assertEqual(scan_cap, 40)
        self.assertGreaterEqual(scan_cap, current_tip_of_day_candidate_universe)
        self.assertLess(25, current_tip_of_day_candidate_universe)

    def test_full_incident_universe_keeps_late_tier1_candidates_reachable(self):
        workflow = self._production_workflow()
        match = re.search(r'^\s*MAX_CANDIDATES_PER_RUBRIC:\s*"(\d+)"\s*$', workflow, re.MULTILINE)
        self.assertIsNotNone(match)
        scan_cap = int(match.group(1))

        # When the full 36-item universe is within the scan cap, no candidate can
        # be dropped solely for occupying positions 26-36, including late
        # scientific/Tier-1 ASHA and HealthyChildren candidates from incident #456.
        candidate_positions = list(range(1, 37))
        scanned_positions = candidate_positions[:scan_cap]
        self.assertEqual(scanned_positions, candidate_positions)
        self.assertIn(26, scanned_positions)
        self.assertIn(36, scanned_positions)

    def test_scan_capacity_change_preserves_existing_run_and_hard_skip_budgets(self):
        workflow = self._production_workflow()
        self.assertIn('MAX_RUN_SECONDS: "1500"', workflow)
        self.assertIn('MAX_SKIPS_PER_RUBRIC: "12"', workflow)
        self.assertIn('MAX_LLM_SECONDS_PER_CANDIDATE: "180"', workflow)
        self.assertIn('IMAGE_PROMPT_TIMEOUT_SECONDS: "60"', workflow)


if __name__ == "__main__":
    unittest.main()
