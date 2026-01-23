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


def test_tvec_arithmetic():
    a = (1, -2, 3)
    b = (0, 5, -7)
    assert add_tvec(a, b) == (1, 3, -4)
    assert sub_tvec(a, b) == (1, -7, 10)
    assert neg_tvec(a) == (-1, 2, -3)
