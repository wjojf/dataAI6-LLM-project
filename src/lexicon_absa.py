import spacy
import re
from typing import List, Dict, Set, Tuple
from collections import defaultdict
import nltk
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .base import ABSAAnalyzer, AspectSentiment


class LexiconABSA(ABSAAnalyzer):
    """Lexicon-based Aspect-Based Sentiment Analysis using spaCy and sentiment lexicons"""
    
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.vader = SentimentIntensityAnalyzer()
        self.aspect_patterns = self._load_aspect_patterns()
        self.negation_words = {
            'not', 'no', 'never', 'none', 'nothing', 'nowhere', 'neither', 'nor',
            'hardly', 'scarcely', 'barely', 'rarely', 'seldom', 'barely'
        }
        self.intensifiers = {
            'very', 'extremely', 'highly', 'quite', 'rather', 'somewhat', 'fairly',
            'pretty', 'really', 'totally', 'completely', 'absolutely', 'utterly'
        }
        
    def _load_aspect_patterns(self) -> List[str]:
        """Load POS patterns for aspect extraction"""
        return [
            'NOUN', 'PROPN', 'ADJ NOUN', 'NOUN NOUN', 'ADJ ADJ NOUN',
            'DET ADJ NOUN', 'NOUN PREP NOUN'
        ]
    
    def analyze(self, text: str) -> List[AspectSentiment]:
        """Analyze text and extract aspect-sentiment pairs"""
        doc = self.nlp(text)
        aspects = self._extract_aspects(doc)
        sentiments = self._extract_sentiments(doc)
        
        aspect_sentiments = self._match_aspects_sentiments(aspects, sentiments, doc)
        return aspect_sentiments
    
    def _extract_aspects(self, doc) -> List[Dict]:
        """Extract potential aspects using linguistic patterns"""
        aspects = []
        
        for token in doc:
            if token.pos_ in ['NOUN', 'PROPN'] and not token.is_stop:
                aspect_info = {
                    'text': token.text.lower(),
                    'start': token.idx,
                    'end': token.idx + len(token.text),
                    'token': token,
                    'confidence': 0.8
                }
                aspects.append(aspect_info)
        
        for chunk in doc.noun_chunks:
            if len(chunk) > 1:
                # Use the root/head token of the noun chunk for dependency parsing
                aspect_info = {
                    'text': chunk.text.lower(),
                    'start': chunk.start_char,
                    'end': chunk.end_char,
                    'token': chunk.root,
                    'confidence': 0.9
                }
                aspects.append(aspect_info)
        
        return self._deduplicate_aspects(aspects)
    
    def _extract_sentiments(self, doc) -> List[Dict]:
        """Extract sentiment words and their polarities"""
        sentiments = []
        
        for token in doc:
            if token.pos_ in ['ADJ', 'ADV'] and not token.is_stop:
                sentiment_score = self._get_sentiment_score(token.text)
                if sentiment_score != 0:
                    sentiment_info = {
                        'text': token.text.lower(),
                        'start': token.idx,
                        'end': token.idx + len(token.text),
                        'token': token,
                        'score': sentiment_score,
                        'confidence': abs(sentiment_score)
                    }
                    sentiments.append(sentiment_info)
        
        return sentiments
    
    def _get_sentiment_score(self, word: str) -> float:
        """Get sentiment score for a word using multiple lexicons"""
        scores = []
        
        vader_scores = self.vader.polarity_scores(word)
        scores.append(vader_scores['compound'])
        
        try:
            synsets = list(swn.senti_synsets(word))
            if synsets:
                pos_score = synsets[0].pos_score()
                neg_score = synsets[0].neg_score()
                swn_score = pos_score - neg_score
                scores.append(swn_score)
        except:
            pass
        
        if scores:
            return sum(scores) / len(scores)
        return 0.0
    
    def _match_aspects_sentiments(self, aspects: List[Dict], sentiments: List[Dict], doc) -> List[AspectSentiment]:
        """Match aspects with their corresponding sentiments using dependency parsing"""
        aspect_sentiments = []
        
        for aspect in aspects:
            aspect_sentiment = self._find_sentiment_for_aspect(aspect, sentiments, doc)
            if aspect_sentiment:
                aspect_sentiments.append(aspect_sentiment)
        
        return aspect_sentiments
    
    def _find_sentiment_for_aspect(self, aspect: Dict, sentiments: List[Dict], doc) -> AspectSentiment:
        """Find sentiment for a specific aspect using dependency relations"""
        aspect_token = aspect['token']
        best_sentiment = None
        best_score = 0
        
        for sentiment in sentiments:
            sentiment_token = sentiment['token']
            
            if self._are_related(aspect_token, sentiment_token, doc):
                score = sentiment['confidence']
                if score > best_score:
                    best_score = score
                    best_sentiment = sentiment
        
        if best_sentiment:
            sentiment_label = self._score_to_label(best_sentiment['score'])
            return AspectSentiment(
                aspect=aspect['text'],
                sentiment=sentiment_label,
                confidence=best_score,
                text_span=(aspect['start'], aspect['end'])
            )
        
        return None
    
    def _are_related(self, aspect_token, sentiment_token, doc) -> bool:
        """Check if aspect and sentiment tokens are related using dependency parsing"""
        if aspect_token == sentiment_token:
            return False
        
        # Check if sentiment is a direct modifier of the aspect
        if sentiment_token in aspect_token.children:
            return True
        
        # Check if aspect is a child of the sentiment
        if aspect_token in sentiment_token.children:
            return True
        
        # Check if they share a common head (e.g., both modify the same verb)
        if aspect_token.head == sentiment_token.head and aspect_token.head != aspect_token:
            return True
        
        # Check if they're connected through a copula (e.g., "food is good")
        if aspect_token.head == sentiment_token or sentiment_token.head == aspect_token:
            return True
        
        # Check if they're in the same sentence and within proximity (window of 5 tokens)
        aspect_idx = aspect_token.i
        sentiment_idx = sentiment_token.i
        if abs(aspect_idx - sentiment_idx) <= 5:
            return True
        
        return False
    
    def _score_to_label(self, score: float) -> str:
        """Convert sentiment score to label"""
        if score > 0.1:
            return 'positive'
        elif score < -0.1:
            return 'negative'
        else:
            return 'neutral'
    
    def _deduplicate_aspects(self, aspects: List[Dict]) -> List[Dict]:
        """Remove duplicate aspects"""
        seen = set()
        unique_aspects = []
        
        for aspect in aspects:
            if aspect['text'] not in seen:
                seen.add(aspect['text'])
                unique_aspects.append(aspect)
        
        return unique_aspects
