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
        
        # Expanded domain aspect patterns
        domain_patterns = [
            # Service-related
            ('wait time', ['wait time', 'waiting time', 'wait']),
            ('service speed', ['service speed', 'service quick', 'service slow', 'service fast', 'quick service', 'slow service', 'fast service']),
            ('service', ['service', 'customer service', 'service quality']),
            ('staff', ['staff', 'waitstaff', 'wait staff', 'waiters', 'waitresses', 'server', 'servers', 'bartender', 'bartenders']),
            ('customer service', ['customer service', 'service quality']),
            # Food-related
            ('food', ['food', 'cuisine', 'meal', 'meals', 'dish', 'dishes']),
            ('food quality', ['food quality', 'quality of food']),
            ('food variety', ['food variety', 'menu variety', 'variety', 'selection']),
            ('menu', ['menu', 'menu options', 'menu variety']),
            # Atmosphere/Ambience
            ('atmosphere', ['atmosphere', 'ambience', 'ambiance', 'vibe', 'vibe']),
            ('decor', ['decor', 'decoration', 'interior', 'exterior']),
            ('location', ['location', 'place', 'spot']),
            ('cleanliness', ['cleanliness', 'clean', 'dirty', 'dirt']),
            # Price/Value
            ('price', ['price', 'pricing', 'cost', 'costs']),
            ('value', ['value', 'worth', 'bang for buck']),
            # Experience
            ('experience', ['experience', 'visit', 'visit']),
            ('overall experience', ['overall experience', 'overall', 'in general']),
        ]
        
        # Use regex for better pattern matching
        for aspect_name, patterns in domain_patterns:
            for pattern in patterns:
                # Simple word-based matching
                pattern_words = pattern.split()
                if len(pattern_words) == 1:
                    # Single word pattern
                    if pattern in text:
                        # Find all occurrences
                        for match in re.finditer(r'\b' + re.escape(pattern) + r'\b', text):
                            # Find corresponding token
                            for token in doc:
                                if match.start() <= token.idx <= match.end():
                                    aspect_info = {
                                        'text': aspect_name,
                                        'start': match.start(),
                                        'end': match.end(),
                                        'token': token,
                                        'confidence': 0.9
                                    }
                                    aspects.append(aspect_info)
                                    break
                else:
                    # Multi-word pattern
                    pattern_lower = pattern.lower()
                    if pattern_lower in text:
                        match_start = text.find(pattern_lower)
                        match_end = match_start + len(pattern_lower)
                        # Find corresponding token
                        for token in doc:
                            if match_start <= token.idx < match_end:
                                aspect_info = {
                                    'text': aspect_name,
                                    'start': match_start,
                                    'end': match_end,
                                    'token': token,
                                    'confidence': 0.9
                                }
                                aspects.append(aspect_info)
                                break
        
        return aspects
    
    def _extract_aspects(self, doc) -> List[Dict]:
        """Extract potential aspects using linguistic patterns"""
        aspects = []
        
        # First, extract domain-specific aspects
        aspects.extend(self._extract_domain_aspects(doc))
        
        # Generic/meaningless words to filter out (expanded list)
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
            'expectations', 'assortment', 'family', 'time', 'times',
            'minute', 'minutes', 'second', 'seconds', 'moment', 'moments',
            'person', 'people', 'guy', 'guys', 'girl', 'girls', 'woman', 'women',
            'man', 'men', 'someone', 'anyone', 'everyone', 'nobody'
        }
        
        # Extract single noun aspects
        for token in doc:
            if token.pos_ in ['NOUN', 'PROPN'] and not token.is_stop:
                aspect_text = token.text.lower()
                # Filter out generic aspects and very short ones
                if aspect_text not in generic_aspects and len(aspect_text) > 2:
                    # Check if it's part of a compound aspect already found
                    is_compound = False
                    for existing_aspect in aspects:
                        if aspect_text in existing_aspect['text'] and existing_aspect['text'] != aspect_text:
                            is_compound = True
                            break
                    if not is_compound:
                        aspect_info = {
                            'text': aspect_text,
                            'start': token.idx,
                            'end': token.idx + len(token.text),
                            'token': token,
                            'confidence': 0.8
                        }
                        aspects.append(aspect_info)
        
        # Extract noun chunks (compound aspects)
        for chunk in doc.noun_chunks:
            if len(chunk) >= 1 and len(chunk) <= 4:  # Allow up to 4 words
                chunk_text = chunk.text.lower()
                
                # Skip possessive phrases
                if chunk_text.startswith(('his ', 'her ', 'their ', 'your ', 'my ', 'our ', 'its ')):
                    continue
                
                # Skip number phrases like "two rolled grape leaves"
                if chunk[0].pos_ == 'NUM':
                    continue
                
                # Skip quantifier phrases
                if chunk_text.startswith(('no ', 'many ', 'every ', 'all ', 'some ', 'another ', 'few ', 'several ')):
                    continue
                
                # Remove articles from chunk text
                for article in ['the ', 'a ', 'an ']:
                    if chunk_text.startswith(article):
                        chunk_text = chunk_text[len(article):]
                        break
                
                # Skip if too short or generic
                if len(chunk_text) < 3:
                    continue
                
                chunk_words = [t.text.lower() for t in chunk if not t.is_stop]
                if chunk_words and all(w not in generic_aspects for w in chunk_words):
                    # Check for meaningful compound aspects
                    meaningful_compound = False
                    if len(chunk_words) > 1:
                        # Multi-word compounds are more likely to be meaningful
                        meaningful_compound = True
                    elif len(chunk_words) == 1:
                        # Single word should not be generic
                        meaningful_compound = chunk_words[0] not in generic_aspects
                    
                    if meaningful_compound:
                        aspect_info = {
                            'text': chunk_text,
                            'start': chunk.start_char,
                            'end': chunk.end_char,
                            'token': chunk.root,
                            'confidence': 0.9 if len(chunk_words) > 1 else 0.8
                        }
                        aspects.append(aspect_info)
        
        # Extract ADJ + NOUN patterns (e.g., "good food", "excellent service")
        for i, token in enumerate(doc):
            if token.pos_ == 'ADJ' and i + 1 < len(doc):
                next_token = doc[i + 1]
                if next_token.pos_ in ['NOUN', 'PROPN']:
                    aspect_text = next_token.text.lower()
                    if aspect_text not in generic_aspects and len(aspect_text) > 2:
                        aspect_info = {
                            'text': aspect_text,
                            'start': next_token.idx,
                            'end': next_token.idx + len(next_token.text),
                            'token': next_token,
                            'confidence': 0.85
                        }
                        aspects.append(aspect_info)
        
        return self._deduplicate_aspects(aspects)
    
    def _extract_sentiments(self, doc) -> List[Dict]:
        """Extract sentiment words and their polarities"""
        sentiments = []
        
        # Sentiment verbs (e.g., "love", "hate", "enjoy", "disappoint")
        sentiment_verbs = {
            'love', 'hate', 'enjoy', 'disappoint', 'disappointed', 'disappointing',
            'like', 'dislike', 'prefer', 'appreciate', 'admire', 'despise',
            'satisfy', 'satisfied', 'satisfying', 'frustrate', 'frustrated', 'frustrating',
            'impress', 'impressed', 'impressing', 'amaze', 'amazed', 'amazing',
            'disgust', 'disgusted', 'disgusting', 'please', 'pleased', 'pleasing',
            'shock', 'shocked', 'shocking', 'surprise', 'surprised', 'surprising',
            'thrill', 'thrilled', 'thrilling', 'bore', 'bored', 'boring',
            'excite', 'excited', 'exciting', 'terrify', 'terrified', 'terrifying'
        }
        
        for i, token in enumerate(doc):
            sentiment_score = 0.0
            sentiment_text = token.text.lower()
            
            # Check adjectives and adverbs
            if token.pos_ in ['ADJ', 'ADV'] and not token.is_stop:
                sentiment_score = self._get_sentiment_score(token.text)
            
            # Check verbs with sentiment
            elif token.pos_ == 'VERB' and sentiment_text in sentiment_verbs:
                sentiment_score = self._get_sentiment_score(token.text)
            
            # Check for sentiment in compound words or phrases
            if sentiment_score == 0 and token.pos_ in ['ADJ', 'ADV']:
                # Try with context (e.g., "very good", "really bad")
                if i > 0:
                    prev_token = doc[i - 1]
                    if prev_token.text.lower() in self.intensifiers:
                        sentiment_score = self._get_sentiment_score(token.text)
                        if sentiment_score != 0:
                            sentiment_score *= 1.2  # Boost for intensifiers
            
            if sentiment_score != 0:
                # Check for negation
                is_negated = self._is_negated(token, doc)
                if is_negated:
                    sentiment_score = -sentiment_score
                
                # Check for intensifiers
                intensifier_strength = self._get_intensifier_strength(token, doc)
                sentiment_score *= intensifier_strength
                
                sentiment_info = {
                    'text': sentiment_text,
                    'start': token.idx,
                    'end': token.idx + len(token.text),
                    'token': token,
                    'score': sentiment_score,
                    'confidence': min(abs(sentiment_score), 1.0)
                }
                sentiments.append(sentiment_info)
        
        return sentiments
    
    def _get_sentiment_score(self, word: str) -> float:
        """Get sentiment score for a word using multiple lexicons"""
        scores = []
        weights = []
        
        # VADER sentiment
        vader_scores = self.vader.polarity_scores(word)
        vader_score = vader_scores['compound']
        if abs(vader_score) > 0.05:  # Only use if significant
            scores.append(vader_score)
            weights.append(1.0)
        
        # SentiWordNet
        try:
            synsets = list(swn.senti_synsets(word))
            if synsets:
                # Use the first synset (most common)
                pos_score = synsets[0].pos_score()
                neg_score = synsets[0].neg_score()
                swn_score = pos_score - neg_score
                # Scale to [-1, 1] range
                if swn_score != 0:
                    swn_score = max(-1.0, min(1.0, swn_score * 2))
                    scores.append(swn_score)
                    weights.append(0.8)
        except:
            pass
        
        # Manual sentiment lexicon for common words
        manual_lexicon = {
            # Positive
            'excellent': 0.9, 'amazing': 0.9, 'wonderful': 0.9, 'fantastic': 0.9,
            'great': 0.8, 'good': 0.7, 'nice': 0.6, 'fine': 0.5,
            'delicious': 0.9, 'tasty': 0.8, 'yummy': 0.8, 'perfect': 0.9,
            'lovely': 0.8, 'beautiful': 0.8, 'gorgeous': 0.9, 'stunning': 0.9,
            'outstanding': 0.9, 'superb': 0.9, 'brilliant': 0.9, 'exceptional': 0.9,
            'friendly': 0.7, 'helpful': 0.7, 'attentive': 0.7, 'professional': 0.7,
            'fast': 0.6, 'quick': 0.6, 'efficient': 0.7, 'speedy': 0.6,
            'clean': 0.6, 'comfortable': 0.7, 'cozy': 0.7, 'relaxing': 0.7,
            # Negative
            'terrible': -0.9, 'awful': -0.9, 'horrible': -0.9, 'disgusting': -0.9,
            'bad': -0.7, 'poor': -0.7, 'worse': -0.8, 'worst': -0.9,
            'slow': -0.6, 'late': -0.6, 'delayed': -0.6,
            'rude': -0.8, 'unfriendly': -0.7, 'unhelpful': -0.7, 'unprofessional': -0.8,
            'dirty': -0.8, 'messy': -0.7, 'filthy': -0.9,
            'expensive': -0.6, 'overpriced': -0.7, 'costly': -0.6,
            'disappointing': -0.7, 'frustrating': -0.7, 'annoying': -0.7,
            'mediocre': -0.5, 'average': -0.3, 'okay': -0.2, 'meh': -0.3,
        }
        
        word_lower = word.lower()
        if word_lower in manual_lexicon:
            scores.append(manual_lexicon[word_lower])
            weights.append(1.2)  # Higher weight for manual lexicon
        
        # Weighted average
        if scores and weights:
            weighted_sum = sum(s * w for s, w in zip(scores, weights))
            total_weight = sum(weights)
            return weighted_sum / total_weight
        
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
        best_distance = float('inf')
        
        # Find all related sentiments and pick the best one
        related_sentiments = []
        for sentiment in sentiments:
            sentiment_token = sentiment['token']
            
            if self._are_related(aspect_token, sentiment_token, doc):
                distance = abs(aspect_token.i - sentiment_token.i)
                related_sentiments.append((sentiment, distance, sentiment['confidence']))
        
        # Sort by distance (closer is better), then by confidence
        related_sentiments.sort(key=lambda x: (x[1], -x[2]))
        
        if related_sentiments:
            best_sentiment, best_distance, best_score = related_sentiments[0]
            # Boost confidence if very close
            if best_distance <= 2:
                best_score = min(1.0, best_score * 1.1)
        
        if best_sentiment:
            sentiment_label = self._score_to_label(best_sentiment['score'])
            return AspectSentiment(
                aspect=aspect['text'],
                sentiment=sentiment_label,
                confidence=min(1.0, best_score),
                text_span=(aspect['start'], aspect['end'])
            )
        
        # If no direct sentiment found, use context-based sentiment
        context = self._get_aspect_context(aspect, doc)
        context_sentiment = self.vader.polarity_scores(context)
        sentiment_score = context_sentiment['compound']
        
        # Lower threshold for context-based sentiment (0.03 instead of 0.05)
        if abs(sentiment_score) < 0.03:
            return None
        
        sentiment_label = self._score_to_label(sentiment_score)
        
        return AspectSentiment(
            aspect=aspect['text'],
            sentiment=sentiment_label,
            confidence=abs(sentiment_score) * 0.75,  # Slightly higher confidence for context-based
            text_span=(aspect['start'], aspect['end'])
        )
    
    def _get_aspect_context(self, aspect: Dict, doc) -> str:
        """Get the context around an aspect for sentiment analysis"""
        aspect_token = aspect['token']
        
        # Get the sentence containing the aspect
        sentence = aspect_token.sent
        
        # If sentence is short, use the whole sentence
        if len(sentence) <= 15:
            return sentence.text
        
        # Otherwise, get a wider window around the aspect (expanded from 5 to 7 tokens each side)
        start_idx = max(sentence.start, aspect_token.i - 7)
        end_idx = min(sentence.end, aspect_token.i + 8)
        
        context_tokens = [doc[i].text for i in range(start_idx, end_idx)]
        return ' '.join(context_tokens)
    
    def _is_negated(self, token, doc) -> bool:
        """Check if a token is negated"""
        # Check direct children for negation words
        for child in token.children:
            if child.text.lower() in self.negation_words:
                return True
        
        # Check if negation is in the dependency path
        if token.head.text.lower() in self.negation_words:
            return True
        
        # Check nearby tokens (within 3 tokens before)
        token_idx = token.i
        for i in range(max(0, token_idx - 3), token_idx):
            if doc[i].text.lower() in self.negation_words:
                return True
        
        return False
    
    def _get_intensifier_strength(self, token, doc) -> float:
        """Get intensifier strength modifier"""
        # Check direct children
        for child in token.children:
            if child.text.lower() in self.intensifiers:
                intensifier = child.text.lower()
                if intensifier in ['very', 'extremely', 'highly', 'really', 'totally', 'completely', 'absolutely', 'utterly']:
                    return 1.3
                elif intensifier in ['quite', 'rather', 'somewhat', 'fairly', 'pretty']:
                    return 1.1
                else:
                    return 1.2
        
        # Check nearby tokens (within 2 tokens before)
        token_idx = token.i
        for i in range(max(0, token_idx - 2), token_idx):
            if doc[i].text.lower() in self.intensifiers:
                intensifier = doc[i].text.lower()
                if intensifier in ['very', 'extremely', 'highly', 'really', 'totally', 'completely', 'absolutely', 'utterly']:
                    return 1.3
                elif intensifier in ['quite', 'rather', 'somewhat', 'fairly', 'pretty']:
                    return 1.1
                else:
                    return 1.2
        
        return 1.0
    
    def _are_related(self, aspect_token, sentiment_token, doc) -> bool:
        """Check if aspect and sentiment tokens are related using dependency parsing"""
        if aspect_token == sentiment_token:
            return False
        
        # Same sentence check
        if aspect_token.sent != sentiment_token.sent:
            return False
        
        aspect_idx = aspect_token.i
        sentiment_idx = sentiment_token.i
        
        # Check if sentiment is a direct modifier of the aspect
        if sentiment_token in aspect_token.children:
            return True
        
        # Check if aspect is a child of the sentiment
        if aspect_token in sentiment_token.children:
            return True
        
        # Check if they share a common head (e.g., both modify the same verb/noun)
        if aspect_token.head == sentiment_token.head and aspect_token.head != aspect_token:
            return True
        
        # Check if they're connected through a copula (e.g., "food is good")
        if aspect_token.head == sentiment_token or sentiment_token.head == aspect_token:
            return True
        
        # Check dependency path (aspect -> head -> sentiment or vice versa)
        if aspect_token.head == sentiment_token.head:
            return True
        
        # Check if aspect is connected to sentiment through dependency chain (max 2 hops)
        aspect_head = aspect_token.head
        if aspect_head == sentiment_token or sentiment_token.head == aspect_head:
            return True
        
        sentiment_head = sentiment_token.head
        if sentiment_head == aspect_token or aspect_token.head == sentiment_head:
            return True
        
        # Check if they're in the same sentence and within proximity (expanded window of 8 tokens)
        if abs(aspect_idx - sentiment_idx) <= 8:
            # Additional check: are they in the same noun phrase or verb phrase?
            # Check if there's a path between them
            if abs(aspect_idx - sentiment_idx) <= 3:
                return True
            
            # Check for common patterns: ADJ NOUN, NOUN is ADJ, etc.
            if aspect_token.pos_ in ['NOUN', 'PROPN'] and sentiment_token.pos_ in ['ADJ', 'ADV']:
                if abs(aspect_idx - sentiment_idx) <= 5:
                    return True
        
        return False
    
    def _score_to_label(self, score: float) -> str:
        """Convert sentiment score to label"""
        # Lower threshold for better sensitivity (0.05 instead of 0.1)
        if score > 0.05:
            return 'positive'
        elif score < -0.05:
            return 'negative'
        else:
            return 'neutral'
    
    def _deduplicate_aspects(self, aspects: List[Dict]) -> List[Dict]:
        """Remove duplicate aspects"""
        seen = set()
        unique_aspects = []
        
        # Sort by confidence (higher first) to keep the best aspects
        aspects_sorted = sorted(aspects, key=lambda x: x['confidence'], reverse=True)
        
        for aspect in aspects_sorted:
            aspect_text = aspect['text'].lower()
            
            # Check for exact matches
            if aspect_text in seen:
                continue
            
            # Check for substring matches (e.g., "food" and "food quality")
            is_duplicate = False
            for seen_text in seen:
                if aspect_text in seen_text or seen_text in aspect_text:
                    # Keep the longer/more specific one
                    if len(aspect_text) > len(seen_text):
                        # Remove the shorter one and add this one
                        unique_aspects = [a for a in unique_aspects if a['text'].lower() != seen_text]
                        seen.remove(seen_text)
                        seen.add(aspect_text)
                        unique_aspects.append(aspect)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                seen.add(aspect_text)
                unique_aspects.append(aspect)
        
        return unique_aspects
