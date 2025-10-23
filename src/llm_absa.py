import ollama
import json
import re
from typing import List, Dict, Optional
import time

from .base import ABSAAnalyzer, AspectSentiment


class LLMABSA(ABSAAnalyzer):
    """LLM-based Aspect-Based Sentiment Analysis using Ollama"""
    
    def __init__(self, model_name: str = "llama3.1:8b"):
        self.model_name = model_name
        self.max_retries = 3
        self.retry_delay = 1
        
    def analyze(self, text: str) -> List[AspectSentiment]:
        """Analyze text and extract aspect-sentiment pairs using LLM"""
        prompt = self._create_prompt(text)
        
        for attempt in range(self.max_retries):
            try:
                response = self._get_llm_response(prompt)
                aspect_sentiments = self._parse_response(response)
                return aspect_sentiments
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    return []
    
    def _create_prompt(self, text: str) -> str:
        """Create a structured prompt for ABSA analysis"""
        prompt = f"""You are an expert in Aspect-Based Sentiment Analysis. Your task is to identify aspects (features, attributes, or topics) mentioned in the given text and determine the sentiment expressed toward each aspect.

Text to analyze: "{text}"

Please analyze this text and extract all aspects with their corresponding sentiments. Return your analysis in the following JSON format:

{{
  "aspects": [
    {{
      "aspect": "aspect_name",
      "sentiment": "positive|negative|neutral",
      "confidence": 0.0-1.0,
      "reasoning": "brief explanation"
    }}
  ]
}}

Guidelines:
1. Identify all meaningful aspects mentioned in the text
2. Determine sentiment for each aspect (positive, negative, or neutral)
3. Provide confidence scores (0.0 to 1.0)
4. Include brief reasoning for each sentiment decision
5. If no clear aspects are found, return an empty aspects array
6. Focus on specific features, attributes, or topics rather than general statements

Examples:
- "The pizza was delicious but the service was terrible" → aspects: "pizza" (positive), "service" (negative)
- "The laptop has great performance and battery life" → aspects: "performance" (positive), "battery life" (positive)
- "The hotel room was clean" → aspects: "hotel room" (positive)

Now analyze the given text:"""
        
        return prompt
    
    def _get_llm_response(self, prompt: str) -> str:
        """Get response from Ollama LLM"""
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                options={
                    'temperature': 0.1,
                    'top_p': 0.9,
                    'num_predict': 1000
                }
            )
            return response['message']['content']
        except Exception as e:
            raise RuntimeError(f"Failed to get LLM response: {e}")
    
    def _parse_response(self, response: str) -> List[AspectSentiment]:
        """Parse LLM response and extract aspect-sentiment pairs"""
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                return self._fallback_parsing(response)
            
            json_str = json_match.group()
            data = json.loads(json_str)
            
            aspect_sentiments = []
            for item in data.get('aspects', []):
                aspect = item.get('aspect', '').strip()
                sentiment = item.get('sentiment', 'neutral').lower()
                confidence = float(item.get('confidence', 0.5))
                
                if aspect and sentiment in ['positive', 'negative', 'neutral']:
                    aspect_sentiment = AspectSentiment(
                        aspect=aspect,
                        sentiment=sentiment,
                        confidence=confidence
                    )
                    aspect_sentiments.append(aspect_sentiment)
            
            return aspect_sentiments
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Error parsing LLM response: {e}")
            return self._fallback_parsing(response)
    
    def _fallback_parsing(self, response: str) -> List[AspectSentiment]:
        """Fallback parsing when JSON parsing fails"""
        aspect_sentiments = []
        
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            aspect_match = re.search(r'aspect[:\s]+([^,]+)', line, re.IGNORECASE)
            sentiment_match = re.search(r'(positive|negative|neutral)', line, re.IGNORECASE)
            
            if aspect_match and sentiment_match:
                aspect = aspect_match.group(1).strip()
                sentiment = sentiment_match.group(1).lower()
                
                aspect_sentiment = AspectSentiment(
                    aspect=aspect,
                    sentiment=sentiment,
                    confidence=0.7
                )
                aspect_sentiments.append(aspect_sentiment)
        
        return aspect_sentiments
    
    def _validate_sentiment(self, sentiment: str) -> str:
        """Validate and normalize sentiment labels"""
        sentiment = sentiment.lower().strip()
        if sentiment in ['positive', 'pos', '+']:
            return 'positive'
        elif sentiment in ['negative', 'neg', '-']:
            return 'negative'
        else:
            return 'neutral'
    
    def _extract_confidence(self, text: str) -> float:
        """Extract confidence score from text"""
        confidence_match = re.search(r'confidence[:\s]+([0-9.]+)', text, re.IGNORECASE)
        if confidence_match:
            try:
                return float(confidence_match.group(1))
            except ValueError:
                pass
        
        return 0.7
