import unittest

from src.publisher.dedup_policy import (
    SEMANTIC_THRESHOLD_POST_METHOD_PIGGYBANK,
    semantic_post_threshold_for_rubric,
    should_allow_evergreen_source_reuse,
)


class DedupPolicyTest(unittest.TestCase):
    def test_method_piggybank_allows_evergreen_source_reuse(self):
        self.assertTrue(should_allow_evergreen_source_reuse("method_piggybank"))

    def test_method_piggybank_final_semantic_threshold_unchanged(self):
        self.assertEqual(SEMANTIC_THRESHOLD_POST_METHOD_PIGGYBANK, 0.985)
        self.assertEqual(semantic_post_threshold_for_rubric("method_piggybank"), 0.985)


if __name__ == "__main__":
    unittest.main()
