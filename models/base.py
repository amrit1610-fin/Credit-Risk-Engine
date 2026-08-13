from abc import ABC, abstractmethod
from typing import List, Dict, Any
from core.portfolio import Portfolio

class RiskEngine(ABC):
    """
    Abstract base class for all risk calculations.
    Ensures that any new risk model we build (like Copulas later) 
    follows the same standard interface.
    """
    def __init__(self, portfolio: Portfolio):
        self.portfolio = portfolio

    @abstractmethod
    def calculate_risk(self) -> Dict[str, Any]:
        """
        Core method to execute the risk calculations.
        Must return a dictionary of metrics to be displayed on the Dash UI.
        """
        pass