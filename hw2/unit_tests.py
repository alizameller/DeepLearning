# /bin/env python3.8

import pytest


# testing linearity given that the activation functions are identity
def test_additivity():
    import tensorflow as tf
    from hw2 import MLP

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 2
    num_outputs = 1
    num_hidden_layers = 2
    hidden_layer_width = 50
    batch_size = 100

    mlp = MLP(
        num_inputs,
        num_outputs,
        num_hidden_layers,
        hidden_layer_width,
        hidden_activation=tf.identity,
        output_activation=tf.identity,
    )

    a = rng.normal(shape=[batch_size, num_inputs])
    b = rng.normal(shape=[batch_size, num_inputs])

    tf.debugging.assert_near(mlp(a + b), mlp(a) + mlp(b), summarize=2)


# testing linearity given that the activation functions are identity
def test_homogeneity():
    import tensorflow as tf
    from hw2 import MLP

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 2
    num_outputs = 1
    num_hidden_layers = 2
    hidden_layer_width = 50
    batch_size = 100

    mlp = MLP(
        num_inputs,
        num_outputs,
        num_hidden_layers,
        hidden_layer_width,
        hidden_activation=tf.identity,
        output_activation=tf.identity,
    )

    a = rng.normal(shape=[batch_size, num_inputs])
    b = 5

    tf.debugging.assert_near(mlp(a * b), mlp(a) * b, summarize=2)


def test_output_shape():
    import tensorflow as tf
    from hw2 import MLP

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 2
    num_outputs = 1
    num_hidden_layers = 2
    hidden_layer_width = 50

    batch_size = 100

    mlp = MLP(
        num_inputs,
        num_outputs,
        num_hidden_layers,
        hidden_layer_width,
        hidden_activation=tf.nn.relu,
        output_activation=tf.nn.sigmoid,
    )

    a = rng.normal(shape=[batch_size, num_inputs])

    tf.debugging.assert_equal(mlp(a).shape, (batch_size, num_outputs), summarize=2)


def test_num_trainable():
    import tensorflow as tf
    from hw2 import MLP

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 2
    num_outputs = 1
    num_hidden_layers = 2
    hidden_layer_width = 50
    batch_size = 100

    mlp = MLP(
        num_inputs,
        num_outputs,
        num_hidden_layers,
        hidden_layer_width,
        hidden_activation=tf.nn.relu,
        output_activation=tf.nn.sigmoid,
    )

    a = rng.normal(shape=[batch_size, num_inputs])

    num_trainable = (num_hidden_layers + 2) * len(mlp.input_linear.trainable_variables)

    tf.debugging.assert_equal(len(mlp.trainable_variables), num_trainable, summarize=2)
