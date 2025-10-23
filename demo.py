#!/usr/bin/env python3
"""
Demo script for Aspect-Based Sentiment Analysis project.
This script demonstrates all three ABSA implementations.
"""

import sys
import os
import time
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from lexicon_absa import LexiconABSA
from transformer_absa import TransformerABSA
from llm_absa import LLMABSA
from utils import create_sample_data, print_analysis_results


def main():
    """Main demo function"""
    print("=" * 60)
    print("Aspect-Based Sentiment Analysis Demo")
    print("=" * 60)
    
    analyzers = {}
    
    print("\nInitializing analyzers...")
    
    try:
        lexicon_analyzer = LexiconABSA()
        analyzers['Lexicon'] = lexicon_analyzer
        print("✓ LexiconABSA initialized")
    except Exception as e:
        print(f"✗ LexiconABSA failed: {e}")
    
    try:
        transformer_analyzer = TransformerABSA()
        analyzers['Transformer'] = transformer_analyzer
        print("✓ TransformerABSA initialized")
    except Exception as e:
        print(f"✗ TransformerABSA failed: {e}")
    
    try:
        llm_analyzer = LLMABSA()
        analyzers['LLM'] = llm_analyzer
        print("✓ LLMABSA initialized")
    except Exception as e:
        print(f"✗ LLMABSA failed: {e}")
    
    if not analyzers:
        print("No analyzers could be initialized. Please check your setup.")
        return
    
    print(f"\nSuccessfully initialized {len(analyzers)} analyzers")
    
    sample_texts = [
        "The pizza was delicious but the service was terrible.",
        "The laptop has great performance and excellent battery life.",
        "The hotel room was clean and comfortable, but the WiFi was slow."
    ]
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION")
    print("=" * 60)
    
    for i, text in enumerate(sample_texts, 1):
        print(f"\n{i}. Text: {text}")
        print("-" * 50)
        
        for name, analyzer in analyzers.items():
            try:
                start_time = time.time()
                results = analyzer.analyze(text)
                processing_time = time.time() - start_time
                
                print(f"\n{name} Results ({processing_time:.3f}s):")
                if results:
                    for result in results:
                        print(f"  • {result.aspect}: {result.sentiment} "
                              f"(confidence: {result.confidence:.3f})")
                else:
                    print("  No aspects found")
                    
            except Exception as e:
                print(f"\n{name} Error: {e}")
    
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)
    
    test_data = create_sample_data()
    total_results = []
    
    for name, analyzer in analyzers.items():
        print(f"\nBenchmarking {name}...")
        method_results = []
        
        for item in test_data[:3]:
            try:
                start_time = time.time()
                results = analyzer.analyze(item['text'])
                processing_time = time.time() - start_time
                
                method_results.append({
                    'text': item['text'],
                    'method': name,
                    'aspects': results,
                    'processing_time': processing_time
                })
                
            except Exception as e:
                print(f"Error processing '{item['text']}': {e}")
        
        total_results.extend(method_results)
    
    print("\nSummary:")
    for name in analyzers.keys():
        method_results = [r for r in total_results if r['method'] == name]
        if method_results:
            avg_time = sum(r['processing_time'] for r in method_results) / len(method_results)
            avg_aspects = sum(len(r['aspects']) for r in method_results) / len(method_results)
            print(f"  {name}: {avg_time:.3f}s avg, {avg_aspects:.1f} aspects avg")
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
