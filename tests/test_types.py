import numpy as np
import pytest

from pbcgraph import (
    add_tvec,
    neg_tvec,
    sub_tvec,
    validate_tvec,
    zero_tvec,
)


def test_zero_tvec():
    assert zero_tvec(3) == (0, 0, 0)
    with pytest.raises(ValueError):
        zero_tvec(0)


def test_validate_tvec_dim_and_int():
    validate_tvec((0, 1, -2), 3)
    with pytest.raises(ValueError):
        validate_tvec((0, 1), 3)
    with pytest.raises(ValueError):
        validate_tvec((0, 1, 2.0), 3)


def test_validate_tvec_numpy_and_rejects_bool():
    validate_tvec(np.array([0, 1, -2], dtype=np.int64), 3)
    validate_tvec((np.int64(1), np.int32(0), np.int64(-1)), 3)

    # bool is an int subclass, but should be rejected
    with pytest.raises(ValueError):
        validate_tvec((True, 0, 0), 3)


def test_tvec_arithmetic():
    a = (1, -2, 3)
    b = (0, 5, -7)
    assert add_tvec(a, b) == (1, 3, -4)
    assert sub_tvec(a, b) == (1, -7, 10)
    assert neg_tvec(a) == (-1, 2, -3)
