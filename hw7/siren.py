#!/bin/env python
import tensorflow as tf
import numpy as np


class Adam:
    def __init__(
        self,
        trainable_vars,
        alpha=0.001,
        beta_1=0.9,
        beta_2=0.999,
        eps=1e-8,
    ):
        self.m_list = [
            tf.zeros(shape=tf.shape(variable).numpy())
            for variable in trainable_vars
        ]
        self.v_list = [
            tf.zeros(shape=tf.shape(variable).numpy())
            for variable in trainable_vars
        ]
        self.alpha = alpha
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps

    def __call__(self, t, grads, vars):
        new_m_list = []
        new_v_list = []
        for m, v, var, grad in zip(self.m_list, self.v_list, vars, grads):
            m = self.beta_1 * m + (1 - self.beta_1) * grad
            v = self.beta_2 * v + (1 - self.beta_2) * (grad**2)
            new_m_list.append(m)
            new_v_list.append(v)
            m_hat = m / (1 - self.beta_1**t)
            v_hat = v / (1 - self.beta_2**t)
            var.assign_sub(
                (self.alpha * m_hat) / ((v_hat**0.5) + self.eps)
            )
        self.m_list = new_m_list
        self.v_list = new_v_list


class Linear(tf.Module):
    def __init__(self, num_inputs, num_outputs, bias=True, is_first=False):
        rng = tf.random.get_global_generator()

        self.w = tf.Variable(
            rng.uniform(
                shape=[num_inputs, num_outputs],
                minval=-1 * tf.math.sqrt(6 / num_inputs),
                maxval=tf.math.sqrt(6 / num_inputs),
            ),
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

        self.is_first = is_first

    def __call__(self, x):
        z = x @ self.w

        if self.bias:
            z += self.b

        if self.is_first:
            z *= 30

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
        self.input_linear = Linear(num_inputs, self.M, is_first=True)
        self.hidden_linear = [
            Linear(self.M, self.M) for i in range(self.K - 1)
        ]
        self.hidden_linear.append(Linear(self.M, self.M))
        self.output_linear = Linear(self.M, num_outputs)

    def __call__(self, x):
        p = self.input_linear(x)
        for i in range(self.K):
            p = self.hidden_activation(p)
            p = self.hidden_linear[i](p)

        p = self.output_linear(p)
        return self.output_activation(p)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib
    from tqdm import trange
    import cv2

    hidden_layer_width = 256
    num_hidden_layers = 8
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

    # Reshape input to 180 x 180
    img = cv2.resize(cv2.imread("TestcardF.jpeg"), (180, 180)) / 255

    # Target
    pixel_values = img.reshape(-1, 3)

    # Input
    resolution = img.shape[0]
    tmp = np.linspace(-1, 1, resolution)
    x, y = np.meshgrid(tmp, tmp)
    pixel_coordinates = tf.concat((x.reshape(-1, 1), y.reshape(-1, 1)), 1)

    # Ground Truth
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 3))
    ax1.imshow(img)
    ax1.set_title("Ground Truth", fontsize=13)
    # plt.show()

    bar = trange(num_iters)
    adam = Adam(siren.trainable_variables)

    for i in bar:
        with tf.GradientTape() as tape:
            model_output = siren(
                tf.cast(pixel_coordinates, dtype=tf.float32)
            )
            loss = tf.math.reduce_mean((model_output - pixel_values) ** 2)
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
            output = model_output.numpy().reshape(180, 180, 3)
            ax2.imshow(output)

    ax2.set_title("Trained Image", fontsize=13)

    x_test, y_test = np.meshgrid(
        np.linspace(-1, 1, 720), np.linspace(-1, 1, 720)
    )
    pixel_coordinates_test = tf.concat(
        (x_test.reshape(-1, 1), y_test.reshape(-1, 1)), 1
    )
    test = siren(tf.cast(pixel_coordinates_test, dtype=tf.float32))
    test_output = test.numpy().reshape(720, 720, 3)
    ax3.imshow(test_output)
    ax3.set_title("Interesting", fontsize=13)

    plt.axis("scaled")
    plt.savefig("fig.png")
    matplotlib.image.imsave("trained_image.png", np.clip(output, 0, 1))
    matplotlib.image.imsave(
       "super_resolution.png", np.clip(test_output, 0, 1)
    )
    plt.close()
