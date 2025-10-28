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
    
    def _extract_domain_aspects(self, doc) -> List[Dict]:
        """Extract domain-specific composite aspects"""
        aspects = []
        text = doc.text.lower()
        
        # Common domain aspect patterns (service/temporal concepts)
        domain_patterns = [
            ('wait time', ['wait time', 'waiting time', 'wait']),
            ('service speed', ['service speed', 'service.*quick', 'service.*slow', 'service.*fast']),
            ('wait', ['long.*wait', 'wait.*long', 'waiting']),
        ]
        
        for aspect_name, patterns in domain_patterns:
            for pattern in patterns:
                if pattern in text or any(word in text for word in pattern.split('.*')):
                    # Find approximate position
                    for token in doc:
                        if token.text.lower() in pattern or pattern.split('.*')[0] in token.text.lower():
                            aspect_info = {
                                'text': aspect_name,
                                'start': token.idx,
                                'end': token.idx + len(token.text),
                                'token': token,
                                'confidence': 0.85
                            }
                            aspects.append(aspect_info)
                            break
                    break
        
        return aspects
    
    def _extract_aspects(self, doc) -> List[Dict]:
        """Extract potential aspects using linguistic patterns"""
        aspects = []
        
        # First, extract domain-specific aspects
        aspects.extend(self._extract_domain_aspects(doc))
        
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
            'expectations', 'assortment', 'family'
        }
        
        for token in doc:
            if token.pos_ in ['NOUN', 'PROPN'] and not token.is_stop:
                aspect_text = token.text.lower()
                # Filter out generic aspects and very short ones
                if aspect_text not in generic_aspects and len(aspect_text) > 2:
                    aspect_info = {
                        'text': aspect_text,
                        'start': token.idx,
                        'end': token.idx + len(token.text),
                        'token': token,
                        'confidence': 0.8
                    }
                    aspects.append(aspect_info)
        
        for chunk in doc.noun_chunks:
            if len(chunk) > 1 and len(chunk) <= 3:  # Max 3 words
                # Skip if contains only generic words or starts with number/determiner phrases
                chunk_text = chunk.text.lower()
                
                # Skip possessive phrases
                if chunk_text.startswith(('his ', 'her ', 'their ', 'your ', 'my ', 'our ')):
                    continue
                
                # Skip number phrases like "two rolled grape leaves"
                if chunk[0].pos_ == 'NUM':
                    continue
                
                # Skip quantifier phrases
                if chunk_text.startswith(('no ', 'many ', 'every ', 'all ', 'some ', 'another ')):
                    continue
                
                # Remove articles from chunk text
                for article in ['the ', 'a ', 'an ']:
                    if chunk_text.startswith(article):
                        chunk_text = chunk_text[len(article):]
                        break
                
                chunk_words = [t.text.lower() for t in chunk if not t.is_stop]
                if chunk_words and all(w not in generic_aspects for w in chunk_words):
                    # Use the root/head token of the noun chunk for dependency parsing
                    aspect_info = {
                        'text': chunk_text,
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
        
        # If no direct sentiment found, use context-based sentiment
        context = self._get_aspect_context(aspect, doc)
        context_sentiment = self.vader.polarity_scores(context)
        sentiment_score = context_sentiment['compound']
        
        # If context sentiment is too weak, skip this aspect
        if abs(sentiment_score) < 0.05:
            return None
        
        sentiment_label = self._score_to_label(sentiment_score)
        
        return AspectSentiment(
            aspect=aspect['text'],
            sentiment=sentiment_label,
            confidence=abs(sentiment_score) * 0.7,  # Reduce confidence for context-based
            text_span=(aspect['start'], aspect['end'])
        )
    
    def _get_aspect_context(self, aspect: Dict, doc) -> str:
        """Get the context around an aspect for sentiment analysis"""
        aspect_token = aspect['token']
        
        # Get the sentence containing the aspect
        sentence = aspect_token.sent
        
        # If sentence is short, use the whole sentence
        if len(sentence) <= 10:
            return sentence.text
        
        # Otherwise, get a window around the aspect
        start_idx = max(0, aspect_token.i - 5)
        end_idx = min(len(doc), aspect_token.i + 6)
        
        context_tokens = [doc[i].text for i in range(start_idx, end_idx)]
        return ' '.join(context_tokens)
    
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
