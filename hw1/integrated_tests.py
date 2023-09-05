# /bin/env python3.8
import pytest


@pytest.mark.parametrize("num_outputs", [1, 16, 128])
def test_dimensionality(num_outputs):
    import tensorflow as tf

    from hw1 import BasisExpansion
    from hw1 import Linear

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 10
    m = 8

    linear = Linear(num_inputs, num_outputs)
    basis = BasisExpansion(m, num_inputs, num_outputs)

    a = rng.normal(shape=[num_inputs, 1])
    x = basis(a)
    z = linear(tf.transpose(x))
    tf.assert_equal(tf.shape(z)[-1], num_outputs)
