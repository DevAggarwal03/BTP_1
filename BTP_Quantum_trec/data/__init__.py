"""
data/__init__.py
Expose the core data loading and preprocessing classes.
"""
from .trec_loader import TRECLoader
from .preprocessor import TRECPreprocessor

__all__ = ["TRECLoader", "TRECPreprocessor"]
