# /bin/env python3.8

import pytest

def test_output_shape():
    import tensorflow as tf
    from cnn import Classifier

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    n_classes = 10 
    batch_size = 128
    input_depth = 1

    input_shape = tf.cast(rng.normal(shape=[batch_size, 28, 28, input_depth]), tf.float32)

    layers = [32, 64, 128]
    layer_kernel_sizes = [(3, 3), (3, 3), (3, 3)]
    input_depth = 1
    input_height = 28
    input_width = 28
    
    classifier = Classifier(input_height, input_width, input_depth, layers, layer_kernel_sizes, n_classes)
    tf.debugging.assert_equal(classifier(input_shape).shape, (batch_size, n_classes), summarize=2)

def test_gradient():
    import tensorflow as tf
    from cnn import Classifier
    from mnist import MNIST

    mndata = MNIST('./')
    train_images, train_labels = mndata.load_training()
    train_images = tf.reshape(train_images, (60000, 28, 28, 1))

    test_images, test_labels = mndata.load_testing()
    test_images = tf.reshape(test_images, (10000, 28, 28, 1))

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    layers = [32,64,128]
    layer_kernel_sizes = [[3,3],[3,3],[3,3]]
    n_classes = 10
    lambda_param = 0.1
    input_depth = 1
    input_height = 28
    input_width = 28

    classifier = Classifier(input_height, input_width, input_depth, layers, layer_kernel_sizes, n_classes)
    
    batch_indices = rng.uniform(
            shape=[128], maxval=tf.shape(train_images)[0], dtype=tf.int32
        )
    with tf.GradientTape() as tape:
            input_batch = tf.cast(tf.gather(train_images, batch_indices), tf.float32)
            label_batch = tf.gather(train_labels, batch_indices)
            
            label_hat = classifier(input_batch)
            label_hat = tf.cast(label_hat, dtype=tf.float32)
            label_batch = tf.cast(label_batch, dtype=tf.int32)
            l2 = lambda_param * tf.norm(tf.concat([tf.reshape(variable,-1) for variable in classifier.trainable_variables],0))
            loss = tf.math.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(label_batch, label_hat)) + l2

    grads = tape.gradient(
            loss, classifier.trainable_variables
        )
    gradient_sum = tf.reduce_sum([tf.reduce_sum(i) for i in grads])

    tf.debugging.assert_none_equal(gradient_sum, tf.zeros(shape=tf.shape(gradient_sum)))

def test_adam():
    import tensorflow as tf
    from cnn import Classifier, Adam
    from mnist import MNIST

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    mndata = MNIST('./')
    train_images, train_labels = mndata.load_training()
    train_images = tf.reshape(train_images, (60000, 28, 28, 1))

    test_images, test_labels = mndata.load_testing()
    test_images = tf.reshape(test_images, (10000, 28, 28, 1))

    layers = [32,64,128]
    layer_kernel_sizes = [[3,3],[3,3],[3,3]]
    n_classes = 10
    lambda_param = 0.1
    input_depth = 1
    input_height = 28
    input_width = 28

    classifier = Classifier(input_height, input_width, input_depth, layers, layer_kernel_sizes, n_classes)
    adam = Adam(classifier.trainable_variables)
    batch_indices = rng.uniform(
            shape=[128], maxval=tf.shape(train_images)[0], dtype=tf.int32
        )
    with tf.GradientTape() as tape:
            input_batch = tf.cast(tf.gather(train_images, batch_indices), tf.float32)
            label_batch = tf.gather(train_labels, batch_indices)
            
            label_hat = classifier(input_batch)
            label_hat = tf.cast(label_hat, dtype=tf.float32)
            label_batch = tf.cast(label_batch, dtype=tf.int32)
            l2 = lambda_param * tf.norm(tf.concat([tf.reshape(variable,-1) for variable in classifier.trainable_variables],0))
            loss = tf.math.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(label_batch, label_hat)) + l2
    
    grads = tape.gradient(loss, classifier.trainable_variables)

    old_variable_sum = tf.reduce_sum([tf.reduce_sum(i) for i in classifier.trainable_variables])
    adam(1, grads, classifier.trainable_variables)
    new_variable_sum = tf.reduce_sum([tf.reduce_sum(i) for i in classifier.trainable_variables])

    tf.debugging.assert_none_equal(new_variable_sum, old_variable_sum)
