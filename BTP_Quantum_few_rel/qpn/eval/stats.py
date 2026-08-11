# This module provides statistical helpers for comparing model score distributions.
# It wraps hypothesis tests such as the Wilcoxon signed-rank test for paired evaluation results.
from scipy.stats import wilcoxon
import numpy as np
from typing import Tuple

def paired_test(a_scores: list, b_scores: list) -> Tuple[float, float]:
    """
    Performs a paired Wilcoxon signed-rank test on two sets of scores.
    Returns (p_value, effect_size)
    Effect size is approximated as Z / sqrt(N), where Z is the normal approximation.
    """
    a = np.array(a_scores)
    b = np.array(b_scores)
    
    # If all differences are zero, p-value is 1.0
    if np.allclose(a, b):
        return 1.0, 0.0
        
    try:
        res = wilcoxon(a, b, zero_method='pratt')
        p_val = res.pvalue
        
        # Approximate effect size r = Z / sqrt(N)
        # We can reconstruct Z from p_val for a two-sided test
        from scipy.stats import norm
        z = norm.ppf(1 - p_val / 2)
        n = len(a)
        effect_size = z / np.sqrt(n)
        
        return p_val, effect_size
    except ValueError:
        # Happens if all differences are exactly 0 in wilcoxon
        return 1.0, 0.0
