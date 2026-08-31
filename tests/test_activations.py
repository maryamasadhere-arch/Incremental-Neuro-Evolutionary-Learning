import numpy as np

from inel.activations import relu, sig


def test_sig_range():
    x = np.array([-1000.0, -1.0, 0.0, 1.0, 1000.0])
    out = sig(x)
    assert np.all((out >= 0) & (out <= 1))
    assert np.isclose(out[2], 0.5)


def test_sig_no_overflow():
    # the clip exists precisely to avoid exp() overflow
    out = sig(np.array([-1e6, 1e6]))
    assert np.isfinite(out).all()


def test_sig_no_overflow_warning_for_float32():
    # float32's exp() overflows around 88, well inside the old float64-only
    # clip bound of 500 - this must not raise a RuntimeWarning.
    import warnings
    x = np.array([-1e6, 1e6], dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = sig(x)
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out, [0.0, 1.0], atol=1e-30)


def test_relu():
    x = np.array([-2.0, -0.0001, 0.0, 0.5, 3.0])
    np.testing.assert_array_equal(relu(x), [0, 0, 0, 0.5, 3.0])
