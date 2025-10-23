from dataclasses import dataclass
from typing import List, Optional, Tuple
from abc import ABC, abstractmethod


@dataclass
class AspectSentiment:
    """Data class for aspect-sentiment pairs"""
    aspect: str
    sentiment: str
    confidence: float
    text_span: Optional[Tuple[int, int]] = None


class ABSAAnalyzer(ABC):
    """Base class/interface for all ABSA implementations"""
    
    @abstractmethod
    def analyze(self, text: str) -> List[AspectSentiment]:
        """
        Analyze text and extract aspect-sentiment pairs
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of AspectSentiment objects
        """
        raise NotImplementedError
