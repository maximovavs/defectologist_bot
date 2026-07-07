"""Service package bootstrap.

The reliability overrides keep narrowly scoped validation safeguards isolated from
legacy generation code while preserving the existing llm_generator API.
"""

from . import llm_generator as _llm_generator
from .reliability_overrides import apply_overrides as _apply_reliability_overrides

_apply_reliability_overrides(_llm_generator)

del _apply_reliability_overrides
del _llm_generator
