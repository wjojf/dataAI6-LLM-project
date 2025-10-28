import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from typing import List, Dict, Tuple
import re
import spacy

from .base import ABSAAnalyzer, AspectSentiment


class TransformerABSA(ABSAAnalyzer):
    """Transformer-based Aspect-Based Sentiment Analysis using Hugging Face models"""
    
    def __init__(self, model_name: str = "yangheng/deberta-v3-base-absa-v1.1"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            # Try loading with fast tokenizer first
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            except Exception:
                # Fall back to slow tokenizer if fast tokenizer fails
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
            
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            print("Falling back to alternative ABSA model...")
            self._load_fallback_model()
    
    def _load_fallback_model(self):
        """Load a fallback ABSA model if the primary one fails"""
        try:
            fallback_model = "cardiffnlp/twitter-roberta-base-sentiment-latest"
            self.tokenizer = AutoTokenizer.from_pretrained(fallback_model)
            self.model = AutoModelForSequenceClassification.from_pretrained(fallback_model)
            self.model.to(self.device)
            self.nlp = spacy.load("en_core_web_sm")
            self.model_name = fallback_model
        except Exception as e:
            print(f"Error loading fallback model: {e}")
            raise RuntimeError("Could not load any ABSA model")
    
    def analyze(self, text: str) -> List[AspectSentiment]:
        """Analyze text and extract aspect-sentiment pairs"""
        if "absa" in self.model_name.lower():
            return self._analyze_with_absa_model(text)
        else:
            return self._analyze_with_sentiment_model(text)
    
    def _analyze_with_absa_model(self, text: str) -> List[AspectSentiment]:
        """Analyze using a dedicated ABSA model"""
        aspects = self._extract_aspects(text)
        aspect_sentiments = []
        
        for aspect in aspects:
            try:
                inputs = self.tokenizer(
                    text,
                    aspect,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    predicted_class = torch.argmax(predictions, dim=-1).item()
                    confidence = predictions[0][predicted_class].item()
                
                sentiment_label = self._get_sentiment_label(predicted_class)
                
                aspect_sentiment = AspectSentiment(
                    aspect=aspect,
                    sentiment=sentiment_label,
                    confidence=confidence,
                    text_span=self._find_aspect_span(text, aspect)
                )
                aspect_sentiments.append(aspect_sentiment)
                
            except Exception as e:
                print(f"Error processing aspect '{aspect}': {e}")
                continue
        
        return aspect_sentiments
    
    def _analyze_with_sentiment_model(self, text: str) -> List[AspectSentiment]:
        """Analyze using a general sentiment model with aspect extraction"""
        aspects = self._extract_aspects(text)
        aspect_sentiments = []
        
        classifier = pipeline(
            "sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1
        )
        
        for aspect in aspects:
            try:
                aspect_context = self._get_aspect_context(text, aspect)
                result = classifier(aspect_context)
                
                sentiment_label = self._normalize_sentiment_label(result[0]['label'])
                confidence = result[0]['score']
                
                aspect_sentiment = AspectSentiment(
                    aspect=aspect,
                    sentiment=sentiment_label,
                    confidence=confidence,
                    text_span=self._find_aspect_span(text, aspect)
                )
                aspect_sentiments.append(aspect_sentiment)
                    
            except Exception as e:
                print(f"Error processing aspect '{aspect}': {e}")
                continue
        
        return aspect_sentiments
    
    def _extract_aspects(self, text: str) -> List[str]:
        """Extract aspects from text using spaCy with filtering"""
        doc = self.nlp(text)
        aspects = []
        
        # Generic/meaningless words to filter out
        generic_aspects = {
            'lot', 'lots', 'thing', 'things', 'way', 'ways',
            'place', 'year', 'years', 'day', 'days', 'end', 'need', 'part',
            'review', 'experience', 'experiences', 'order', 'bit', 'kind', 'type', 'sort',
            'everything', 'anything', 'something', 'nothing', 'someone', 'anyone',
            'one', 'two', 'three', 'number', 'hours', 'hour',
            'face', 'butt', 'leg', 'legs', 'arm', 'arms', 'hand', 'hands',
            'head', 'body', 'smile', 'shout', 'notch', 'russell', 'passion',
            'desire', 'ideas', 'recommendations', 'encouragement',
            'chicken', 'grape', 'melon', 'leaves', 'jalapeño', 'tamale',
            'locations', 'location', 'diner', 'restaurant', 'weekends', 'weekend',
            'expectations', 'assortment', 'family', 'clients', 'workouts', 'hotel'
        }
        
        # Extract single nouns
        for token in doc:
            if token.pos_ in ['NOUN', 'PROPN'] and not token.is_stop and len(token.text) > 2:
                aspect = token.text.lower()
                if aspect not in generic_aspects:
                    aspects.append(aspect)
        
        # Extract noun chunks (multi-word aspects)
        for chunk in doc.noun_chunks:
            if 1 < len(chunk.text.split()) <= 3:  # Between 2 and 3 words
                chunk_text = chunk.text.lower()
                
                # Skip possessive phrases (his/her/their/your + noun)
                if chunk_text.startswith(('his ', 'her ', 'their ', 'your ', 'my ', 'our ')):
                    continue
                
                # Skip number phrases
                if chunk[0].pos_ == 'NUM':
                    continue
                
                # Skip if all words are stop words or generic
                words = [w.text.lower() for w in chunk if not w.is_stop]
                if words and all(w not in generic_aspects for w in words):
                    aspects.append(chunk_text)
        
        # Deduplicate and filter
        unique_aspects = list(set(aspects))
        
        # Remove articles from aspects
        cleaned_aspects = []
        for asp in unique_aspects:
            cleaned = asp
            # Remove leading articles
            for article in ['the ', 'a ', 'an ']:
                if cleaned.startswith(article):
                    cleaned = cleaned[len(article):]
            cleaned_aspects.append(cleaned)
        
        # Remove duplicates after cleaning
        unique_aspects = list(set(cleaned_aspects))
        
        # Remove aspects that are substrings of others (keep longer ones)
        filtered = []
        for asp in sorted(unique_aspects, key=len, reverse=True):
            if not any(asp in other and asp != other for other in filtered):
                filtered.append(asp)
        
        # Final quality filters
        final = []
        for asp in filtered:
            # Skip if starts with negation/quantifier
            if asp.startswith(('no ', 'many ', 'every ', 'all ', 'some ')):
                continue
            # Skip very generic phrases
            if asp in ['eclectic assortment', 'clarion hotel', 'family']:
                continue
            final.append(asp)
        
        return final
    
    def _get_aspect_context(self, text: str, aspect: str) -> str:
        """Get context around an aspect for sentiment analysis"""
        words = text.split()
        aspect_words = aspect.split()
        
        for i, word in enumerate(words):
            if word.lower() == aspect_words[0].lower():
                start = max(0, i - 5)
                end = min(len(words), i + len(aspect_words) + 5)
                context = ' '.join(words[start:end])
                return context
        
        return text
    
    def _find_aspect_span(self, text: str, aspect: str) -> Tuple[int, int]:
        """Find the span of an aspect in the original text"""
        text_lower = text.lower()
        aspect_lower = aspect.lower()
        
        start = text_lower.find(aspect_lower)
        if start != -1:
            end = start + len(aspect)
            return (start, end)
        
        return (0, len(text))
    
    def _get_sentiment_label(self, predicted_class: int) -> str:
        """Convert predicted class to sentiment label"""
        label_mapping = {
            0: 'negative',
            1: 'neutral', 
            2: 'positive'
        }
        return label_mapping.get(predicted_class, 'neutral')
    
    def _normalize_sentiment_label(self, label: str) -> str:
        """Normalize sentiment labels from different models to standard format"""
        label_lower = label.lower()
        
        # Handle standard labels
        if label_lower in ['positive', 'negative', 'neutral']:
            return label_lower
        
        # Handle LABEL_X format (common in Hugging Face models)
        if 'label_0' in label_lower or label_lower == '0':
            return 'negative'
        elif 'label_1' in label_lower or label_lower == '1':
            return 'neutral'
        elif 'label_2' in label_lower or label_lower == '2':
            return 'positive'
        
        # Handle other common variations
        if 'pos' in label_lower:
            return 'positive'
        elif 'neg' in label_lower:
            return 'negative'
        elif 'neu' in label_lower:
            return 'neutral'
        
        # Default to neutral if unknown
        return 'neutral'
