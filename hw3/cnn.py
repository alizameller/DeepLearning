#!/bin/env python
import tensorflow as tf
import math
from mnist import MNIST

import sys

sys.path.insert(0, "..")


def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding="SAME")


def dropout2d(x, rate=0.1, seed=4567897):
    return tf.nn.dropout(x, rate, seed=seed)


# strides is the number of pixels by which the filter is shifted during each step
class Conv2d(tf.Module):
    def __call__(self, x, W, strides=1):
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding="SAME")
        return tf.nn.relu(x)


class Classifier(tf.Module):
    def __init__(
        self,
        input_height: int,
        input_width: int,
        input_depth: int,
        layer_depths: list[int],
        layer_kernel_sizes: list[tuple[int, int]],
        num_classes: int,
    ):
        self.num_classes = num_classes
        self.layer_depths = layer_depths
        self.input_height = input_height
        self.input_width = input_width

        self.weights = {
            "wc1": tf.Variable(
                tf.random.normal(
                    shape=[
                        layer_kernel_sizes[0][0],
                        layer_kernel_sizes[0][1],
                        input_depth,
                        layer_depths[0],
                    ],
                    stddev=0.1,
                )
            ),
            "wc2": tf.Variable(
                tf.random.normal(
                    shape=[
                        layer_kernel_sizes[1][0],
                        layer_kernel_sizes[1][1],
                        layer_depths[0],
                        layer_depths[1],
                    ],
                    stddev=0.1,
                )
            ),
            "wc3": tf.Variable(
                tf.random.normal(
                    shape=[
                        layer_kernel_sizes[2][0],
                        layer_kernel_sizes[2][1],
                        layer_depths[1],
                        layer_depths[2],
                    ],
                    stddev=0.1,
                )
            ),
            # max pooling divides by 2x2 for each layer because k = 2,
            # take ceiling of division by 2 for each dim in each layer
            "wd1": tf.Variable(
                tf.random.normal(
                    shape=(
                        layer_depths[2]
                        * (math.ceil(math.ceil(math.ceil(input_height / 2) / 2) / 2))
                        * (math.ceil(math.ceil(math.ceil(input_width / 2) / 2) / 2)),
                        128,
                    ),
                    stddev=0.1,
                )
            ),
            "out": tf.Variable(
                tf.random.normal(shape=(128, self.num_classes), stddev=0.1)
            ),
        }
        self.conv = Conv2d()

    def __call__(self, x):
        # layer 1
        conv1 = self.conv(x, self.weights["wc1"])
        conv1 = maxpool2d(conv1, k=2)
        conv1 = dropout2d(conv1)
        # layer 2
        conv2 = self.conv(conv1, self.weights["wc2"])
        conv2 = maxpool2d(conv2, k=2)
        conv2 = dropout2d(conv2)
        # layer 3
        conv3 = self.conv(conv2, self.weights["wc3"])
        conv3 = maxpool2d(conv3, k=2)
        conv3 = dropout2d(conv3)
        # fully connected layer -> reshaping, matmul, relu and matmul
        fc1 = tf.reshape(
            conv3,
            [
                -1,
                self.layer_depths[2]
                * (math.ceil(math.ceil(math.ceil(self.input_height / 2) / 2) / 2))
                * (math.ceil(math.ceil(math.ceil(self.input_width / 2) / 2) / 2)),
            ],
        )
        fc1 = tf.matmul(fc1, self.weights["wd1"])
        fc1 = tf.nn.relu(fc1)
        out = tf.matmul(fc1, self.weights["out"])

        return out


class Adam:
    def __init__(self, trainable_vars, alpha=0.001, beta_1=0.9, beta_2=0.999, eps=1e-8):
        self.m_list = [
            tf.zeros(shape=tf.shape(variable).numpy()) for variable in trainable_vars
        ]
        self.v_list = [
            tf.zeros(shape=tf.shape(variable).numpy()) for variable in trainable_vars
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
            var.assign_sub((self.alpha * m_hat) / ((v_hat**0.5) + self.eps))
        self.m_list = new_m_list
        self.v_list = new_v_list


if __name__ == "__main__":
    from tqdm import trange

    # make sure mnist data files are in current directory
    mndata = MNIST("./")
    train_images, train_labels = mndata.load_training()
    train_images = tf.reshape(train_images, (60000, 28, 28, 1))

    test_images, test_labels = mndata.load_testing()
    test_images = tf.reshape(test_images, (10000, 28, 28, 1))

    # hyper parameters
    num_iters = 2000
    step_size = 0.5
    decay_rate = 0.999
    batch_size = 128
    validation_size = 10000
    lambda_param = 0.1
    refresh_rate = 10

    input_height = 28
    input_width = 28
    n_classes = 10  # 10 digits [0, 9]
    layer_depths = [32, 64, 128]
    layer_kernel_sizes = [(3, 3), (3, 3), (3, 3)]
    input_depth = 1

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(0x43966E87BD57227011B5B03B58785EC1)
    bar = trange(num_iters)

    classifier = Classifier(
        input_height,
        input_width,
        input_depth,
        layer_depths,
        layer_kernel_sizes,
        n_classes,
    )
    adam = Adam(classifier.trainable_variables)

    for i in bar:
        batch_indices = rng.uniform(
            shape=[batch_size],
            maxval=tf.shape(train_images)[0] - validation_size,
            dtype=tf.int32,
        )

        with tf.GradientTape() as tape:
            input_batch = tf.cast(tf.gather(train_images, batch_indices), tf.float32)
            label_batch = tf.gather(train_labels, batch_indices)
            label_hat = classifier(input_batch)

            # no biases all params are weights
            l2 = lambda_param * tf.norm(
                tf.concat(
                    [
                        tf.reshape(variable, -1)
                        for variable in classifier.trainable_variables
                    ],
                    0,
                )
            )

            loss = (
                tf.math.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=label_batch, logits=label_hat
                    )
                )
            ) + l2

        grads = tape.gradient(
            loss,
            classifier.trainable_variables,
        )

        adam(i + 1, grads, classifier.trainable_variables)

        step_size *= decay_rate

        if i % refresh_rate == (refresh_rate - 1):
            bar.set_description(
                f"Step {i}; Loss => {loss.numpy():0.4f}, step_size => {step_size:0.4f}, "
            )
            bar.refresh()

    # used to adjust hyper parameters after training
    # validation_set_data = test_images[(tf.shape(test_images)[0] - validation_size):]
    # validation_set_labels = test_labels[(tf.shape(test_images)[0] - validation_size):]

    test_label_hat = classifier(tf.cast(test_images, tf.float32))
    sum = 0
    for label, pred in zip(test_labels, tf.math.argmax(test_label_hat, axis=1)):
        if label == pred:
            sum += 1
    accuracy = sum / len(test_labels)
    print("Testing Accuracy is " + str(accuracy * 100) + "%")
