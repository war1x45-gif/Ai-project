import unittest

from simple_ai import ChatBot, NaiveBayesClassifier, tokenize
from simple_ai.bot import FALLBACK_RESPONSES


class TokenizeTest(unittest.TestCase):
    def test_lowercases_and_strips_punctuation(self):
        self.assertEqual(tokenize("Hello, World! What's up?"), ["hello", "world", "what's", "up"])


class NaiveBayesTest(unittest.TestCase):
    def setUp(self):
        self.clf = NaiveBayesClassifier()
        self.clf.train([
            ("i love this movie", "pos"),
            ("great film, loved it", "pos"),
            ("terrible movie", "neg"),
            ("i hated this awful film", "neg"),
        ])

    def test_predicts_correct_label(self):
        self.assertEqual(self.clf.predict("loved it, great")[0], "pos")
        self.assertEqual(self.clf.predict("awful and terrible")[0], "neg")

    def test_probabilities_sum_to_one(self):
        self.assertAlmostEqual(sum(self.clf.predict_proba("movie").values()), 1.0)

    def test_untrained_raises(self):
        with self.assertRaises(ValueError):
            NaiveBayesClassifier().predict("hi")


class ChatBotTest(unittest.TestCase):
    def setUp(self):
        self.bot = ChatBot(seed=0)

    def test_classifies_common_messages(self):
        cases = {
            "hello there!": "greeting",
            "ok bye, see you": "goodbye",
            "thank you very much": "thanks",
            "what's your name?": "name",
            "tell me a funny joke": "joke",
            "what can you do for me": "capabilities",
        }
        for message, expected in cases.items():
            with self.subTest(message=message):
                self.assertEqual(self.bot.classify(message)[0], expected)

    def test_unknown_words_fall_back(self):
        self.assertEqual(self.bot.classify("quantum zebra pancakes"), (None, 0.0))
        self.assertIn(self.bot.respond("quantum zebra pancakes"), FALLBACK_RESPONSES)

    def test_response_comes_from_intent(self):
        self.assertIn(self.bot.respond("hi"), self.bot.responses["greeting"])


if __name__ == "__main__":
    unittest.main()
