import unittest
import sys
import os
from unittest.mock import patch, MagicMock
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.base import ABSAAnalyzer, AspectSentiment
from src.lexicon_absa import LexiconABSA
from src.transformer_absa import TransformerABSA
from src.llm_absa import LLMABSA
from src.utils import calculate_accuracy, calculate_precision_recall_f1, create_sample_data


class TestAspectSentiment(unittest.TestCase):
    """Test cases for AspectSentiment data class"""

    def test_aspect_sentiment_creation(self):
        """Test creating AspectSentiment objects"""
        aspect = AspectSentiment(
            aspect="food",
            sentiment="positive",
            confidence=0.8,
            text_span=(0, 4)
        )

        self.assertEqual(aspect.aspect, "food")
        self.assertEqual(aspect.sentiment, "positive")
        self.assertEqual(aspect.confidence, 0.8)
        self.assertEqual(aspect.text_span, (0, 4))

    def test_aspect_sentiment_without_span(self):
        """Test creating AspectSentiment without text_span"""
        aspect = AspectSentiment(
            aspect="service",
            sentiment="negative",
            confidence=0.9
        )

        self.assertEqual(aspect.aspect, "service")
        self.assertEqual(aspect.sentiment, "negative")
        self.assertEqual(aspect.confidence, 0.9)
        self.assertIsNone(aspect.text_span)


class TestLexiconABSA(unittest.TestCase):
    """Test cases for LexiconABSA implementation"""

    def setUp(self):
        """Set up test fixtures"""
        try:
            self.analyzer = LexiconABSA()
        except Exception as e:
            self.skipTest(f"Could not initialize LexiconABSA: {e}")

    def test_analyze_simple_text(self):
        """Test analysis of simple text"""
        text = "The food was good."
        results = self.analyzer.analyze(text)

        self.assertIsInstance(results, list)
        for result in results:
            self.assertIsInstance(result, AspectSentiment)
            self.assertIn(result.sentiment, ['positive', 'negative', 'neutral'])
            self.assertGreaterEqual(result.confidence, 0.0)
            self.assertLessEqual(result.confidence, 1.0)

    def test_analyze_complex_text(self):
        """Test analysis of complex text with multiple aspects"""
        text = "The pizza was delicious but the service was terrible."
        results = self.analyzer.analyze(text)

        self.assertIsInstance(results, list)
        aspects = [r.aspect for r in results]

        self.assertTrue(len(aspects) > 0)

    def test_analyze_empty_text(self):
        """Test analysis of empty text"""
        text = ""
        results = self.analyzer.analyze(text)

        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 0)

    def test_sentiment_score_calculation(self):
        """Test sentiment score calculation"""
        positive_score = self.analyzer._get_sentiment_score("excellent")
        negative_score = self.analyzer._get_sentiment_score("terrible")

        self.assertIsInstance(positive_score, float)
        self.assertIsInstance(negative_score, float)

    def test_score_to_label_conversion(self):
        """Test conversion of scores to labels"""
        self.assertEqual(self.analyzer._score_to_label(0.5), "positive")
        self.assertEqual(self.analyzer._score_to_label(-0.5), "negative")
        self.assertEqual(self.analyzer._score_to_label(0.05), "neutral")


class TestTransformerABSA(unittest.TestCase):
    """Test cases for TransformerABSA implementation"""

    def setUp(self):
        """Set up test fixtures"""
        try:
            self.analyzer = TransformerABSA()
        except Exception as e:
            self.skipTest(f"Could not initialize TransformerABSA: {e}")

    def test_analyze_simple_text(self):
        """Test analysis of simple text"""
        text = "The food was good."
        results = self.analyzer.analyze(text)

        self.assertIsInstance(results, list)
        for result in results:
            self.assertIsInstance(result, AspectSentiment)
            self.assertIn(result.sentiment, ['positive', 'negative', 'neutral'])
            self.assertGreaterEqual(result.confidence, 0.0)
            self.assertLessEqual(result.confidence, 1.0)

    def test_extract_aspects(self):
        """Test aspect extraction"""
        text = "The pizza was delicious but the service was terrible."
        aspects = self.analyzer._extract_aspects(text)

        self.assertIsInstance(aspects, list)
        self.assertTrue(len(aspects) > 0)

    def test_get_aspect_context(self):
        """Test getting context around aspects"""
        text = "The pizza was delicious but the service was terrible."
        context = self.analyzer._get_aspect_context(text, "pizza")

        self.assertIsInstance(context, str)
        self.assertIn("pizza", context.lower())

    def test_find_aspect_span(self):
        """Test finding aspect span in text"""
        text = "The pizza was delicious."
        span = self.analyzer._find_aspect_span(text, "pizza")

        self.assertIsInstance(span, tuple)
        self.assertEqual(len(span), 2)
        self.assertLessEqual(span[0], span[1])


class TestLLMABSA(unittest.TestCase):
    """Test cases for LLMABSA implementation"""

    def setUp(self):
        """Set up test fixtures"""
        try:
            self.analyzer = LLMABSA()
        except Exception as e:
            self.skipTest(f"Could not initialize LLMABSA: {e}")

    def test_create_prompt(self):
        """Test prompt creation"""
        text = "The food was good."
        prompt = self.analyzer._create_prompt(text)

        self.assertIsInstance(prompt, str)
        self.assertIn(text, prompt)
        self.assertIn("aspects", prompt.lower())

    @patch('ollama.chat')
    def test_get_llm_response(self, mock_chat):
        """Test getting LLM response"""
        mock_response = {
            'message': {
                'content': '{"aspects": [{"aspect": "food", "sentiment": "positive", "confidence": 0.8}]}'
            }
        }
        mock_chat.return_value = mock_response

        prompt = "Test prompt"
        response = self.analyzer._get_llm_response(prompt)

        self.assertIsInstance(response, str)
        mock_chat.assert_called_once()

    def test_parse_response_valid_json(self):
        """Test parsing valid JSON response"""
        response = '{"aspects": [{"aspect": "food", "sentiment": "positive", "confidence": 0.8}]}'
        results = self.analyzer._parse_response(response)

        self.assertIsInstance(results, list)
        if results:
            self.assertIsInstance(results[0], AspectSentiment)

    def test_parse_response_invalid_json(self):
        """Test parsing invalid JSON response"""
        response = "Invalid JSON response"
        results = self.analyzer._parse_response(response)

        self.assertIsInstance(results, list)

    def test_validate_sentiment(self):
        """Test sentiment validation"""
        self.assertEqual(self.analyzer._validate_sentiment("positive"), "positive")
        self.assertEqual(self.analyzer._validate_sentiment("POS"), "positive")
        self.assertEqual(self.analyzer._validate_sentiment("negative"), "negative")
        self.assertEqual(self.analyzer._validate_sentiment("NEG"), "negative")
        self.assertEqual(self.analyzer._validate_sentiment("unknown"), "neutral")


class TestUtils(unittest.TestCase):
    """Test cases for utility functions"""

    def test_calculate_accuracy(self):
        """Test accuracy calculation"""
        predictions = [
            AspectSentiment("food", "positive", 0.8),
            AspectSentiment("service", "negative", 0.9)
        ]
        ground_truth = [
            AspectSentiment("food", "positive", 0.8),
            AspectSentiment("service", "negative", 0.9)
        ]

        accuracy = calculate_accuracy(predictions, ground_truth)
        self.assertEqual(accuracy, 1.0)

    def test_calculate_accuracy_partial_match(self):
        """Test accuracy calculation with partial matches"""
        predictions = [
            AspectSentiment("food", "positive", 0.8),
            AspectSentiment("service", "positive", 0.9)
        ]
        ground_truth = [
            AspectSentiment("food", "positive", 0.8),
            AspectSentiment("service", "negative", 0.9)
        ]

        accuracy = calculate_accuracy(predictions, ground_truth)
        self.assertEqual(accuracy, 0.5)

    def test_calculate_precision_recall_f1(self):
        """Test precision, recall, and F1 calculation"""
        predictions = [
            AspectSentiment("food", "positive", 0.8),
            AspectSentiment("service", "negative", 0.9)
        ]
        ground_truth = [
            AspectSentiment("food", "positive", 0.8),
            AspectSentiment("service", "negative", 0.9)
        ]

        metrics = calculate_precision_recall_f1(predictions, ground_truth)

        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1', metrics)
        self.assertEqual(metrics['precision'], 1.0)
        self.assertEqual(metrics['recall'], 1.0)
        self.assertEqual(metrics['f1'], 1.0)

    def test_create_sample_data(self):
        """Test sample data creation"""
        sample_data = create_sample_data()

        self.assertIsInstance(sample_data, list)
        self.assertTrue(len(sample_data) > 0)

        for item in sample_data:
            self.assertIn('text', item)
            self.assertIn('ground_truth', item)
            self.assertIsInstance(item['ground_truth'], list)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""

    def test_all_analyzers_implement_interface(self):
        """Test that all analyzers implement the ABSAAnalyzer interface"""
        analyzers = []

        try:
            analyzers.append(LexiconABSA())
        except Exception:
            pass

        try:
            analyzers.append(TransformerABSA())
        except Exception:
            pass

        try:
            analyzers.append(LLMABSA())
        except Exception:
            pass

        for analyzer in analyzers:
            self.assertIsInstance(analyzer, ABSAAnalyzer)
            self.assertTrue(hasattr(analyzer, 'analyze'))
            self.assertTrue(callable(getattr(analyzer, 'analyze')))

    def test_analyzer_consistency(self):
        """Test that analyzers return consistent data types"""
        text = "The food was good."

        analyzers = []
        try:
            analyzers.append(("LexiconABSA", LexiconABSA()))
        except Exception:
            pass

        try:
            analyzers.append(("TransformerABSA", TransformerABSA()))
        except Exception:
            pass

        try:
            analyzers.append(("LLMABSA", LLMABSA()))
        except Exception:
            pass

        for name, analyzer in analyzers:
            with self.subTest(analyzer=name):
                results = analyzer.analyze(text)
                self.assertIsInstance(results, list)

                for result in results:
                    self.assertIsInstance(result, AspectSentiment)
                    self.assertIsInstance(result.aspect, str)
                    self.assertIn(result.sentiment, ['positive', 'negative', 'neutral'])
                    self.assertIsInstance(result.confidence, float)
                    self.assertGreaterEqual(result.confidence, 0.0)
                    self.assertLessEqual(result.confidence, 1.0)


if __name__ == '__main__':
    unittest.main()
