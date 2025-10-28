import json
import time
from typing import List, Dict, Any, Optional
import pandas as pd
from pathlib import Path

from .base import AspectSentiment


def load_test_data(file_path: str) -> List[Dict[str, Any]]:
    """Load test data from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Test data file not found: {file_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file: {e}")
        return []


def save_results(results: List[Dict[str, Any]], file_path: str):
    """Save analysis results to JSON file"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {file_path}")
    except Exception as e:
        print(f"Error saving results: {e}")


def _aspects_match(pred_aspect: str, gt_aspect: str) -> bool:
    """Check if two aspects match using flexible matching rules"""
    pred = pred_aspect.lower().strip()
    gt = gt_aspect.lower().strip()
    
    # Exact match
    if pred == gt:
        return True
    
    # Remove common articles
    pred = pred.replace("the ", "").replace("a ", "").replace("an ", "")
    gt = gt.replace("the ", "").replace("a ", "").replace("an ", "")
    
    if pred == gt:
        return True
    
    # Substring match (handles qualifiers like "food quality" -> "food")
    if pred in gt or gt in pred:
        return True
    
    # Word overlap for multi-word aspects
    pred_words = set(pred.split())
    gt_words = set(gt.split())
    overlap = pred_words & gt_words
    
    # Match if there's significant word overlap
    if overlap and len(overlap) >= max(1, len(gt_words) * 0.5):
        return True
    
    return False


def calculate_accuracy(predictions: List[AspectSentiment], ground_truth: List) -> float:
    """Calculate accuracy between predictions and ground truth
    
    Args:
        predictions: List of AspectSentiment objects
        ground_truth: List of dicts with 'aspect', 'sentiment', 'confidence' keys
    """
    if not ground_truth:
        return 0.0
    
    correct = 0
    total = len(ground_truth)
    
    for gt in ground_truth:
        gt_aspect = gt['aspect'] if isinstance(gt, dict) else gt.aspect
        gt_sentiment = gt['sentiment'] if isinstance(gt, dict) else gt.sentiment
        
        for pred in predictions:
            if _aspects_match(pred.aspect, gt_aspect) and pred.sentiment == gt_sentiment:
                correct += 1
                break
    
    return correct / total if total > 0 else 0.0


def calculate_precision_recall_f1(predictions: List[AspectSentiment], ground_truth: List) -> Dict[str, float]:
    """Calculate precision, recall, and F1 score
    
    Args:
        predictions: List of AspectSentiment objects
        ground_truth: List of dicts with 'aspect', 'sentiment', 'confidence' keys
    """
    if not ground_truth:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for pred in predictions:
        found = False
        for gt in ground_truth:
            gt_aspect = gt['aspect'] if isinstance(gt, dict) else gt.aspect
            gt_sentiment = gt['sentiment'] if isinstance(gt, dict) else gt.sentiment
            
            if _aspects_match(pred.aspect, gt_aspect) and pred.sentiment == gt_sentiment:
                true_positives += 1
                found = True
                break
        if not found:
            false_positives += 1
    
    false_negatives = len(ground_truth) - true_positives
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def format_results_table(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Format results as a pandas DataFrame for easy viewing"""
    data = []
    for result in results:
        row = {
            'text': result['text'],
            'method': result['method'],
            'aspects_found': len(result['aspects']),
            'processing_time': result.get('processing_time', 0),
            'accuracy': result.get('accuracy', 0.0)
        }
        data.append(row)
    
    return pd.DataFrame(data)


def benchmark_analyzer(analyzer, test_data: List[Dict[str, Any]], method_name: str) -> List[Dict[str, Any]]:
    """Benchmark an analyzer on test data"""
    results = []
    
    for item in test_data:
        text = item['text']
        ground_truth = item.get('ground_truth', [])
        
        start_time = time.time()
        try:
            predictions = analyzer.analyze(text)
            processing_time = time.time() - start_time
            
            accuracy = calculate_accuracy(predictions, ground_truth)
            metrics = calculate_precision_recall_f1(predictions, ground_truth)
            
            result = {
                'text': text,
                'method': method_name,
                'aspects': predictions,
                'ground_truth': ground_truth,
                'processing_time': processing_time,
                'accuracy': accuracy,
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1']
            }
            results.append(result)
            
        except Exception as e:
            print(f"Error analyzing text with {method_name}: {e}")
            result = {
                'text': text,
                'method': method_name,
                'aspects': [],
                'ground_truth': ground_truth,
                'processing_time': 0,
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'error': str(e)
            }
            results.append(result)
    
    return results


def create_sample_data() -> List[Dict[str, Any]]:
    """Create sample test data for demonstration"""
    sample_data = [
        {
            "text": "The pizza was delicious but the service was terrible.",
            "ground_truth": [
                {"aspect": "pizza", "sentiment": "positive", "confidence": 0.9},
                {"aspect": "service", "sentiment": "negative", "confidence": 0.9}
            ]
        },
        {
            "text": "The laptop has great performance and excellent battery life.",
            "ground_truth": [
                {"aspect": "performance", "sentiment": "positive", "confidence": 0.8},
                {"aspect": "battery life", "sentiment": "positive", "confidence": 0.8}
            ]
        },
        {
            "text": "The hotel room was clean and comfortable, but the WiFi was slow.",
            "ground_truth": [
                {"aspect": "hotel room", "sentiment": "positive", "confidence": 0.8},
                {"aspect": "WiFi", "sentiment": "negative", "confidence": 0.7}
            ]
        },
        {
            "text": "The food quality is average, nothing special.",
            "ground_truth": [
                {"aspect": "food quality", "sentiment": "neutral", "confidence": 0.6}
            ]
        },
        {
            "text": "Amazing customer support and fast delivery!",
            "ground_truth": [
                {"aspect": "customer support", "sentiment": "positive", "confidence": 0.9},
                {"aspect": "delivery", "sentiment": "positive", "confidence": 0.8}
            ]
        }
    ]
    return sample_data


def print_analysis_results(results: List[Dict[str, Any]]):
    """Print analysis results in a formatted way"""
    for result in results:
        print(f"\nText: {result['text']}")
        print(f"Method: {result['method']}")
        print(f"Processing Time: {result.get('processing_time', 0):.3f}s")
        print(f"Accuracy: {result.get('accuracy', 0.0):.3f}")
        print("Aspects Found:")
        for aspect in result['aspects']:
            print(f"  - {aspect.aspect}: {aspect.sentiment} (confidence: {aspect.confidence:.3f})")
        if result.get('error'):
            print(f"Error: {result['error']}")


def ensure_data_directory():
    """Ensure data directory exists and create sample data if needed"""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    sample_file = data_dir / "test_samples.json"
    if not sample_file.exists():
        sample_data = create_sample_data()
        save_results(sample_data, str(sample_file))
        print(f"Created sample data file: {sample_file}")
    
    return str(sample_file)
