import numpy as np

from qpn_hybrid.data import validate_disjoint_relation_sets, validate_relation_pools


def test_relation_pool_validation_accepts_usable_pools():
    pools = {f"r{i}": np.zeros((16, 384), dtype=np.float32) for i in range(5)}
    validate_relation_pools(pools)


def test_relation_split_validator_rejects_leakage():
    pools = {"shared": np.zeros((16, 384), dtype=np.float32)}
    try:
        validate_disjoint_relation_sets(pools, pools)
    except ValueError:
        return
    raise AssertionError("overlapping relation sets should be rejected")
