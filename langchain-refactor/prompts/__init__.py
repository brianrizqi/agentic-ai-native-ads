"""
__init__.py for prompts package
"""

from .classification_prompts import (
    CLASSIFICATION_EXAMPLES,
    few_shot_classification_prompt,
    simple_classification_prompt,
    explanation_prompt
)

__all__ = [
    'CLASSIFICATION_EXAMPLES',
    'few_shot_classification_prompt',
    'simple_classification_prompt',
    'explanation_prompt',
]
