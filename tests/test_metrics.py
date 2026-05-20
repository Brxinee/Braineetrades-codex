from math import isclose


def test_math_sanity() -> None:
    assert isclose(0.1 + 0.2, 0.3, rel_tol=1e-9)
