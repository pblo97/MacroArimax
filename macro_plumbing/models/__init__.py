"""
Predictive models for liquidity stress detection.

Models that actually work:
- Crisis probability prediction
- Regime classification
- Spread direction
- Feature importance
"""

# Lazy import - only import CrisisPredictor when explicitly used
# This prevents import errors when sklearn is not available
def __getattr__(name):
    if name == 'CrisisPredictor':
        from .crisis_classifier import CrisisPredictor
        return CrisisPredictor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ['CrisisPredictor']
