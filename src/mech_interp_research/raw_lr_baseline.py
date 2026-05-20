"""Logistic-regression baseline on raw centered residual-stream activations.

Baseline 3 in the ICD-9 grounding comparison. Companion to
``tfidf_lr_baseline.py`` (Baseline 1) and the SAE probe column produced
by the same run. Pools centered Gemma-2-2B layer-16 activations to note
level (default ``max``) and runs the same StratifiedKFold LR protocol
the other baselines use, then compares head-to-head against a frozen
``sae_cv_results.csv`` from a prior TF-IDF baseline run.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def pool_raw_activations(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError


def run_raw_lr_baseline(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError
