import unittest

from src.helpers.conversation_memory import retain_recent_completed_turns


class ConversationMemoryTests(unittest.TestCase):
    def test_retains_only_the_latest_four_complete_exchanges(self):
        history = []
        for number in range(1, 7):
            history.extend(
                [
                    {"role": "user", "content": f"question {number}"},
                    {"role": "assistant", "content": f"answer {number}"},
                ]
            )

        retained = retain_recent_completed_turns(history)

        self.assertEqual(len(retained), 8)
        self.assertEqual(retained[0]["content"], "question 3")
        self.assertEqual(retained[-1]["content"], "answer 6")

    def test_returns_empty_history_for_zero_turns(self):
        history = [
            {"role": "user", "content": "completed question"},
            {"role": "assistant", "content": "completed answer"},
        ]

        retained = retain_recent_completed_turns(history, max_turns=0)

        self.assertEqual(retained, [])


if __name__ == "__main__":
    unittest.main()
