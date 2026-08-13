"""
stats.py
Statistical testing helpers for model comparisons.

Ported from BTP_Quantum_few_rel/qpn/eval/stats.py.
"""
from __future__ import annotations

import numpy as np
from typing import Tuple
from scipy.stats import wilcoxon


def paired_test(a_scores: list[float], b_scores: list[float]) -> Tuple[float, float]:
    """
    Performs a paired Wilcoxon signed-rank test on two sets of episode scores.

    Args:
        a_scores: Scores from model A.
        b_scores: Scores from model B.

    Returns:
        (p_value, effect_size)
        Effect size is approximated as Z / sqrt(N).
    """
    a = np.array(a_scores)
    b = np.array(b_scores)

    # If all differences are zero, p-value is 1.0 (no significant difference)
    if np.allclose(a, b):
        return 1.0, 0.0

    try:
        res = wilcoxon(a, b, zero_method="pratt")
        p_val = float(res.pvalue)

        # Approximate effect size r = Z / sqrt(N)
        from scipy.stats import norm

        z = norm.ppf(1 - p_val / 2)
        n = len(a)
        effect_size = float(z / np.sqrt(n))

        return p_val, effect_size
    except ValueError:
        # Happens if all differences are exactly 0 in wilcoxon computation
        return 1.0, 0.0
