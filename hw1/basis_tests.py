# /bin/env python3.8

import pytest
import numpy as np


def test_nonlinearity_additivity():
    # This test tests the non-linearity of the basis functions
    import tensorflow as tf

    from hw1 import BasisExpansion

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)
    num_inputs = 10
    num_outputs = 1
    m = 8

    basis = BasisExpansion(m, num_inputs, num_outputs)
    a = rng.normal(shape=[1, num_inputs])
    b = rng.normal(shape=[1, num_inputs])

    tf.debugging.assert_none_equal(basis(a + b), basis(a) + basis(b), summarize=2)


def test_nonlinearity_homogeneity():
    # This test tests the non-linearity of the basis functions
    import tensorflow as tf

    from hw1 import BasisExpansion

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 10
    num_outputs = 1
    m = 8
    num_test_cases = 100

    basis = BasisExpansion(m, num_inputs, num_outputs)

    a = rng.normal(shape=[1, num_inputs])
    b = rng.normal(shape=[num_test_cases, 1])

    tf.debugging.assert_none_equal(basis(a * b), basis(a) * b, summarize=2)


@pytest.mark.parametrize("num_inputs", [1, 16, 128])
def test_dimensionality(num_inputs):
    # Tests dimmensionality of basis functions against different numbers of
    # inputs to check that number of basis functions remains unchanges
    import tensorflow as tf

    from hw1 import BasisExpansion

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_outputs = 1
    m = 8

    basis = BasisExpansion(m, num_inputs, num_outputs)

    a = rng.normal(shape=[1, num_inputs])
    z = basis(a)
    tf.print(tf.shape(z)[-1])
    tf.print(tf.shape(a)[-1])
    tf.assert_equal(tf.shape(z)[-1], m)


@pytest.mark.parametrize(
    "a_shape, b_shape",
    [([1000, 1000], [100, 100]), ([1000, 100], [100, 100]), ([100, 1000], [100, 100])],
)
def test_init_properties_mu(a_shape, b_shape):
    import tensorflow as tf

    from hw1 import BasisExpansion

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs_a, num_outputs_a = a_shape
    num_inputs_b, num_outputs_b = b_shape
    m = 8

    basis_a = BasisExpansion(m, num_inputs_a, num_outputs_a)
    basis_b = BasisExpansion(m, num_inputs_b, num_outputs_b)

    std_a = tf.math.reduce_std(basis_a.mu)
    std_b = tf.math.reduce_std(basis_b.mu)

    tf.debugging.assert_greater(std_a, std_b)
