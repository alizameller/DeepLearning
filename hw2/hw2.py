#!/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn.inspection
from sklearn.inspection import DecisionBoundaryDisplay
import math

# import sys module
import sys

sys.path.insert(0, "..")
from hw1 import Linear, grad_update

"""
I noticed after increasing the number of iterations to above ~1000, that the loss dropped off tremendously. As a result, I decided 
to increase the num_iters to a very large number (4000) to guarantee that the loss could reach that significant dropoff point. I made 
I kept increasing hidden_layer_width, num_hidden_layers and num_samples until they were large enough to seem reasonable and also allow for
the loss to converge realtively quickly while also being effective as a good perceptron. I made step size to be 0.5 because I only had 
4000 iterations and a smaller step size would require more iterations to converge to the same point. I picked lambda to be small so 
that it wouldn't affect the loss significantly, but something big enough to ensure the penalty would have an effect. 
"""


class MLP(Linear, tf.Module):
    def __init__(
        self,
        num_inputs,
        num_outputs,
        num_hidden_layers,
        hidden_layer_width,
        hidden_activation=tf.identity,
        output_activation=tf.identity,
    ):
        self.M = hidden_layer_width
        self.K = num_hidden_layers
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.input_linear = Linear(num_inputs, self.M)
        self.hidden_linear = [Linear(self.M, self.M) for i in range(self.K)]
        self.output_linear = Linear(self.M, num_outputs)

    def __call__(self, x):
        p = self.input_linear(x)
        for i in range(self.K):
            p = self.hidden_activation(p)
            p = self.hidden_linear[i](p)

        p = self.output_linear(p)
        return self.output_activation(p)  # perform sigmoid to ensure good values


def twospirals(rng=np.random.default_rng(), stddev=0.1, num_samples=100):
    theta_a = rng.uniform(low=math.pi, high=4 * math.pi, size=(num_samples,))
    theta_b = rng.uniform(low=2 * math.pi, high=5 * math.pi, size=(num_samples,))

    r_a = rng.normal(size=(num_samples,), loc=theta_a, scale=stddev)
    r_b = rng.normal(size=(num_samples,), loc=theta_b - math.pi, scale=stddev)

    a_0 = np.vstack([r_a * np.cos(theta_a), r_a * np.sin(theta_a)]).astype("float32")
    a_1 = np.vstack([r_b * np.cos(theta_b), r_b * np.sin(theta_b)]).astype("float32")

    data = np.concatenate((a_0, a_1), axis=1)
    classes = np.concatenate(
        (
            np.zeros(shape=(1, num_samples)).astype("float32"),
            np.ones(shape=(1, num_samples)).astype("float32"),
        ),
        axis=1,
    )

    return (data, classes, a_0, a_1)


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

    hidden_layer_width = 256
    num_hidden_layers = 4
    num_inputs = 2
    num_outputs = 1
    num_samples = 500

    mlp = MLP(
        num_inputs,
        num_outputs,
        num_hidden_layers,
        hidden_layer_width,
        hidden_activation=tf.nn.relu,
        output_activation=tf.nn.sigmoid,
    )
    data, classes, spiral1, spiral2 = twospirals(
        stddev=config["data"]["noise_stddev"], num_samples=num_samples
    )
    num_iters = config["learning"]["num_iters"]
    step_size = config["learning"]["step_size"]
    decay_rate = config["learning"]["decay_rate"]
    batch_size = config["learning"]["batch_size"]
    refresh_rate = config["display"]["refresh_rate"]
    lambda_param = config["learning"]["lambda_param"]

    bar = trange(num_iters)

    for i in bar:
        batch_indices = rng.uniform(
            shape=[batch_size], maxval=2 * num_samples, dtype=tf.int32
        )
        with tf.GradientTape() as tape:
            x_batch = tf.transpose(tf.gather(data, batch_indices, axis=1))
            y_batch = tf.transpose(tf.gather(classes, batch_indices, axis=1))
            y_hat = mlp(x_batch)

            l2 = lambda_param * tf.norm(
                tf.concat(
                    [
                        tf.reshape(variable, -1)
                        for variable in mlp.trainable_variables
                        if "w:0" in variable.name
                    ],
                    0,
                )
            )
            loss = (
                tf.math.reduce_mean(
                    -y_batch * tf.math.log(y_hat)
                    - (1 - y_batch) * tf.math.log(1 - y_hat)
                )
                + l2
            )

        grads = tape.gradient(
            loss,
            mlp.trainable_variables,
        )
        grad_update(
            step_size,
            mlp.trainable_variables,
            grads,
        )

        step_size *= decay_rate

        if i % refresh_rate == (refresh_rate - 1):
            bar.set_description(
                f"Step {i}; Loss => {loss.numpy():0.4f}, step_size => {step_size:0.4f}"
            )
            bar.refresh()

    xx0, xx1 = np.meshgrid(np.linspace(-12, 12), np.linspace(-12, 12))
    grid = np.vstack([xx0.ravel(), xx1.ravel()]).T
    y_pred = np.reshape(mlp(grid), xx0.shape)
    display = DecisionBoundaryDisplay(xx0=xx0, xx1=xx1, response=y_pred)
    display.plot()

    display.ax_.scatter(
        spiral1[0, :], spiral1[1, :], label="class 1", edgecolor="black"
    )
    display.ax_.scatter(
        spiral2[0, :], spiral2[1, :], label="class 2", edgecolor="black"
    )

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Scatter Plot of Generated Data and Decision Boundary")
    plt.legend()
    plt.savefig("Decision_Boundary_plot.pdf")
    plt.show()
