import numpy as np
from scripts.q3_lcmrsg_plus_analyze import get_bootstrap_ci, holm_correction, get_practical_effect

def test_bootstrap_ci():
    """
    Test bootstrap confidence interval function with mock diffs.
    """
    diffs = np.array([0.01, 0.02, 0.015, -0.005, 0.03, 0.012])
    ci_low, ci_high = get_bootstrap_ci(diffs, num_resamples=1000)
    assert ci_low <= np.mean(diffs) <= ci_high
    
def test_holm_correction():
    """
    Test Holm multiple comparison correction.
    """
    p_vals = [0.01, 0.04, 0.15, 0.002, 0.8]
    corrected = holm_correction(p_vals)
    
    assert len(corrected) == len(p_vals)
    assert corrected[3] < corrected[0] < corrected[1] < corrected[2] < corrected[4]
    
def test_practical_effect():
    """
    Test practical significance labels.
    """
    assert get_practical_effect(0.0005) == 'negligible'
    assert get_practical_effect(0.002) == 'small'
    assert get_practical_effect(0.005) == 'moderate'
    assert get_practical_effect(0.01) == 'practically meaningful'
