#!/bin/env python
import tensorflow as tf
import numpy as np

# global variables
upper_bound = 1
lower_bound = 0


class Linear(tf.Module):
    def __init__(self, num_inputs, num_outputs, bias=True):
        rng = tf.random.get_global_generator()

        stddev = tf.math.sqrt(2 / (num_inputs + num_outputs))

        self.w = tf.Variable(
            rng.normal(shape=[num_inputs, num_outputs], stddev=stddev),
            trainable=True,
            name="Linear/w",
        )

        self.bias = bias

        if self.bias:
            self.b = tf.Variable(
                tf.zeros(
                    shape=[1, num_outputs],
                ),
                trainable=True,
                name="Linear/b",
            )

    def __call__(self, x):
        z = x @ self.w

        if self.bias:
            z += self.b

        return z


class BasisExpansion(tf.Module):
    def __init__(self, m, num_inputs, num_outputs):
        rng = tf.random.get_global_generator()
        stddev = tf.math.sqrt(2 / num_inputs + num_outputs)
        self.mu = tf.Variable(
            rng.normal(shape=[m, num_inputs], stddev=stddev),
            trainable=True,
            name="BasisExpansion/mu",
        )
        self.sigma = tf.Variable(
            rng.normal(shape=[m, num_inputs], stddev=stddev),
            trainable=True,
            name="BasisExpansion/sig",
        )

    def __call__(self, x):
        gaussian = tf.math.exp(
            -tf.square(x[:, tf.newaxis] - self.mu) / (tf.square(self.sigma))
        )
        return tf.reduce_sum(gaussian, axis=2)


def grad_update(step_size, variables, grads):
    for var, grad in zip(variables, grads):
        var.assign_sub(step_size * grad)


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    import matplotlib.pyplot as plt
    import yaml
    from tqdm import trange

    parser = argparse.ArgumentParser(
        prog="Linear",
        description="Fits a linear model to some data, given a config",
    )

    parser.add_argument("-c", "--config", type=Path, default=Path("config.yaml"))
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(0x43966E87BD57227011B5B03B58785EC1)

    num_samples = config["data"]["num_samples"]
    num_inputs = 1
    num_outputs = 1

    x = rng.uniform(
        shape=(num_samples, num_inputs), minval=lower_bound, maxval=upper_bound
    )
    w = rng.normal(shape=(num_inputs, num_outputs))
    # b = rng.normal(shape=(1, num_outputs))
    y_noise = rng.normal(
        shape=(num_samples, 1), mean=0, stddev=config["data"]["noise_stddev"]
    )
    y_clean = np.sin(2 * np.pi * x)
    y = y_clean + y_noise

    linear = Linear(config["data"]["m"], num_outputs)
    basis_expansion = BasisExpansion(config["data"]["m"], num_inputs, num_outputs)

    num_iters = config["learning"]["num_iters"]
    step_size = config["learning"]["step_size"]
    decay_rate = config["learning"]["decay_rate"]
    batch_size = config["learning"]["batch_size"]

    refresh_rate = config["display"]["refresh_rate"]

    bar = trange(num_iters)

    for i in bar:
        batch_indices = rng.uniform(
            shape=[batch_size], maxval=num_samples, dtype=tf.int32
        )
        with tf.GradientTape() as tape:
            x_batch = tf.gather(x, batch_indices)
            y_batch = tf.gather(y, batch_indices)

            y_hat = linear(basis_expansion(x_batch))
            loss = tf.math.reduce_mean((y_batch - y_hat) ** 2)

        grads = tape.gradient(
            loss, linear.trainable_variables + basis_expansion.trainable_variables
        )
        grad_update(
            step_size,
            linear.trainable_variables + basis_expansion.trainable_variables,
            grads,
        )

        step_size *= decay_rate

        if i % refresh_rate == (refresh_rate - 1):
            bar.set_description(
                f"Step {i}; Loss => {loss.numpy():0.4f}, step_size => {step_size:0.4f}"
            )
            bar.refresh()

    fig, ax = plt.subplots(1, 2, figsize=(10.25, 4), dpi=200)

    ax[0].plot(x.numpy().squeeze(), y.numpy().squeeze(), "bx", label="raw data points")
    a = tf.linspace(tf.reduce_min(x), tf.reduce_max(x), 100)[:, tf.newaxis]
    ax[0].plot(
        a.numpy().squeeze(),
        linear(basis_expansion(a)).numpy().squeeze(),
        "r--",
        label="linear regression model",
    )
    tf.print(tf.shape(a))
    tf.print(tf.shape(basis_expansion(a)))
    tf.print(tf.shape(linear(basis_expansion(a))))
    ax[0].plot(a, np.sin(2 * np.pi * a), "g-", label="clean sine wave")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[0].set_title("Linear fit using SGD")
    ax[0].legend()

    h = ax[0].set_ylabel("y", labelpad=10)
    h.set_rotation(0)

    a2 = tf.linspace(tf.reduce_min(x), tf.reduce_max(x), 100)[:, tf.newaxis]
    ax[1].plot(
        a2.numpy().squeeze(),
        np.array(basis_expansion(a2).numpy().squeeze()),
        "-",
        label=[
            "Basis Fn 1",
            "Basis Fn 2",
            "Basis Fn 3",
            "Basis Fn 4",
            "Basis Fn 5",
            "Basis Fn 6",
            "Basis Fn 7",
            "Basis Fn 8",
        ],
    )
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("y")
    ax[1].set_title("Basis Functions")
    ax[1].legend()

    h2 = ax[1].set_ylabel("y", labelpad=10)
    h2.set_rotation(0)

    fig.savefig("fit.pdf")
