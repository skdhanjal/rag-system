import unittest

from src.retrieval.query_router import QueryRouter


class QueryRouterTests(unittest.TestCase):
    def setUp(self):
        self.router = QueryRouter()
        self.history = [
            {"role": "user", "content": "Explain the BERT architecture in 3-4 lines."},
            {"role": "assistant", "content": "BERT is a bidirectional Transformer encoder..."},
        ]

    def test_detects_response_refinement_request(self):
        self.assertTrue(
            self.router.is_response_refinement_request(
                "Can you give the response in 2-3 lines?",
                self.history,
            )
        )

    def test_does_not_refine_without_previous_answer(self):
        self.assertFalse(
            self.router.is_response_refinement_request(
                "Can you give the response in 2-3 lines?",
                [{"role": "user", "content": "Explain BERT."}],
            )
        )

    def test_normal_research_question_is_not_refinement(self):
        self.assertFalse(
            self.router.is_response_refinement_request(
                "What is the main training objective of BERT?",
                self.history,
            )
        )


if __name__ == "__main__":
    unittest.main()
