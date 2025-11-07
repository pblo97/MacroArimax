"""
Predictive models for liquidity stress detection.

Models that actually work:
- Crisis probability prediction
- Regime classification
- Spread direction
- Feature importance
"""

from .crisis_classifier import CrisisPredictor

__all__ = ['CrisisPredictor']
