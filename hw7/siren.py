#!/bin/env python
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "../hw3")
from cnn import Adam

class Linear(tf.Module):
    def __init__(self, num_inputs, num_outputs, bias=True):
        rng = tf.random.get_global_generator()

        stddev = tf.math.sqrt(2 / (num_inputs + num_outputs))

        self.w = tf.Variable(
            rng.uniform(shape=[num_inputs, num_outputs], 
            minval = -1* tf.math.sqrt(6/num_inputs), maxval = tf.math.sqrt(6/num_inputs)),
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

class Siren(Linear, tf.Module):
    def __init__(
        self,
        num_inputs,
        num_outputs,
        num_hidden_layers,
        hidden_layer_width,
        hidden_activation,
        output_activation,
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
        # return self.output_activation(p)
        return tf.nn.sigmoid(p)

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    import matplotlib.pyplot as plt
    import yaml
    from tqdm import trange
    import cv2

    hidden_layer_width = 256
    num_hidden_layers = 4
    num_inputs = 2
    num_outputs = 3
    num_samples = 500

    step_size = 0.5
    batch_size = 128
    num_iters = 1000
    decay_rate = 0.999
    refresh_rate = 10

    siren = Siren(
        num_inputs,
        num_outputs,
        num_hidden_layers,
        hidden_layer_width,
        hidden_activation=tf.math.sin,
        output_activation=tf.math.sin,
    )

    img = cv2.resize(cv2.imread('Testcard_f.jpg'), (180, 180))/255
    print(img.shape)
    # breakpoint()

    # Target
    pixel_values = img.reshape(-1, 3)

    # Input
    resolution = img.shape[0]
    tmp = np.linspace(-1, 1, resolution)
    x, y = np.meshgrid(tmp, tmp)
    pixel_coordinates = tf.concat((x.reshape(-1, 1), y.reshape(-1, 1)), 1)
 
    # Ground Truth
    fig, axes = plt.subplots(1, 2, figsize=(15, 3))
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Ground Truth', fontsize=13)
    #plt.show()
    
    bar = trange(num_iters)
    adam = Adam(siren.trainable_variables)

    for i in bar:
        with tf.GradientTape() as tape:
            model_output = siren(tf.cast(pixel_coordinates, dtype=tf.float32))
            loss = tf.math.reduce_mean((model_output - pixel_values)**2)
            # breakpoint()
        grads = tape.gradient(
            loss,
            siren.trainable_variables,
        )

        adam(i + 1, grads, siren.trainable_variables)

        step_size *= decay_rate

        if i % refresh_rate == (refresh_rate - 1):
            bar.set_description(
                f"Step {i}; Loss => {loss.numpy():0.4f}, step_size => {step_size:0.4f}, "
            )
            bar.refresh()
            axes[1].imshow(model_output.numpy().reshape(180, 180, 3), cmap='gray')

    for i in range(2):
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    plt.show()
    plt.savefig("fig.png")
    plt.close() 
