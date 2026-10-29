"""No historical source receives implicit production admission in R2."""
from __future__ import annotations
import json
import unittest
from pathlib import Path
from research.r3a_r2_final_post_contract import SourceSnapshot, assess_finalized_post, REJECTION_REASON

EVIDENCE_PATH = Path(__file__).resolve().parents[1] / "evidence" / "source_admission_20261027_historical_UNVERIFIED.json"


class R3AR2HistoricalAdmissionInventoryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ledger = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))

    def test_exact_recorded_11_sources_and_no_live_admission(self):
        entries = self.ledger["sources"]
        self.assertEqual(len(entries), 11)
        self.assertEqual(len({x["source_id"] for x in entries}), 11)
        self.assertEqual(sum(x["historical_capture"] == "HTTP_403_REPORTED" for x in entries), 2)
        self.assertEqual(sum(x["historical_capture"] == "HTTP_200_REPORTED" for x in entries), 9)
        for case in entries:
            with self.subTest(source_id=case["source_id"]):
                self.assertEqual(case["admission"], "NO_POST")
                self.assertEqual(case["authenticated_independent_clinical_receipt"], "NOT_VERIFIED")
                snapshot = SourceSnapshot(case["source_id"], "fixture://unverified", b"test-only", "test-only",
                                          "bilingualism", "children", "unspecified")
                decision = assess_finalized_post(snapshot, "fixture", simulation_only=False)
                self.assertFalse(decision.allowed)
                self.assertEqual(decision.reason, REJECTION_REASON)


if __name__ == "__main__":
    unittest.main()
