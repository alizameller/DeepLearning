#!/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.inspection import DecisionBoundaryDisplay
import math

# import sys module
import sys
sys.path.insert(0,"..")
from hw1.hw1 import Linear, grad_update

def twospirals(n_points, noise=.9):
    rng = np.random.default_rng()
    n = np.sqrt(rng.random(size = (n_points,1))) * (2*np.pi) * 2.25 # range = -2*pi*2.25, 2*pi*2.25
    d1x = -np.cos(n)*n + rng.random(size = (n_points,1)) * noise
    d1y = np.sin(n)*n + rng.random(size = (n_points,1)) * noise
    # creates (x, y) pairs where the set of (d1x, d1y) is class 1 and the set of (-d1x, -d1y) is class 2
    return (np.vstack((np.hstack((d1x,d1y)),np.hstack((-d1x,-d1y)))), 
            np.hstack((np.zeros(n_points),np.ones(n_points))))

class MLP(Linear, tf.Module):
    def __init__(self, num_inputs, num_outputs, 
                num_hidden_layers, hidden_layer_width, 
                hidden_activation = tf.identity, output_activation = tf.identity):
        self.hidden_linear = Linear(hidden_layer_width, hidden_layer_width)
        self.output_linear = Linear(hidden_layer_width, num_outputs)
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.num_hidden_layers = num_hidden_layers
        
    def __call__(self, p):
        for i in self.num_hidden_layers - 1: 
            a = self.hidden_activation@self.hidden_linear(a)
        return self.output_activation@self.output_linear(a)


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

    hidden_layer_width = 5
    num_hidden_layers = 2
    num_inputs = 2
    num_outputs = 1
    num_samples = 400

    linear = Linear(num_inputs, hidden_layer_width)
    mlp = MLP(num_inputs, num_outputs, num_hidden_layers, hidden_layer_width, 
              hidden_activation = tf.mn.relu, output_activation = tf.mn.relu)
    data, classes = twospirals(num_samples)
    num_iters = config["learning"]["num_iters"]
    step_size = config["learning"]["step_size"]
    decay_rate = config["learning"]["decay_rate"]
    batch_size = config["learning"]["batch_size"]

    refresh_rate = config["display"]["refresh_rate"]

    bar = trange(num_iters)

    for i in bar:
        batch_indices = rng.uniform(
            shape=[batch_size], maxval=2*num_samples, dtype=tf.int32
        )
        with tf.GradientTape() as tape:
            x_batch = tf.gather(data, batch_indices)
            y_batch = tf.gather(classes, batch_indices)

            y_hat = mlp(linear(x_batch)) + math.pow(10, -13)
            # fix this
            loss = tf.math.reduce_mean((y_batch - y_hat) ** 2)

        grads = tape.gradient(
            loss, linear.trainable_variables, 
        )
        grad_update(
            step_size,
            linear.trainable_variables,
            grads,
        )

        step_size *= decay_rate

        if i % refresh_rate == (refresh_rate - 1):
            bar.set_description(
                f"Step {i}; Loss => {loss.numpy():0.4f}, step_size => {step_size:0.4f}"
            )
            bar.refresh()

    fig, ax = plt.subplots(1, 2, figsize=(10.25, 4), dpi=200)

    ax[0].title('Spirals')
    ax[0].plot(data[classes==0,0], data[classes==0,1], 'o', label='class 1', color = 'black', mfc = 'red')
    ax[0].plot(data[classes==1,0], data[classes==1,1], 'o', label='class 2', color = 'black', mfc = 'blue')
    ax[0].legend()
    ax[0].show()

    h = ax[0].set_ylabel("y", labelpad=10)
    h.set_rotation(0)