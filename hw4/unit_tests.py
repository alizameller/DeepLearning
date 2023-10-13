# /bin/env python3.8
import pytest

# Testing Resnet Class
def test_output_shape():
    import tensorflow as tf
    from cifar10 import ResNet

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)
    
    n_classes = 10
    batch_size = 128
    channels = 3

    input_shape = tf.cast(
        rng.normal(shape=[batch_size, 32, 32, channels]), tf.float32
    )

    layers = [64,64,128,128,256,256,512,512]

    resnet = ResNet(channels, layers, n_classes)
    output_shape = resnet(input_shape).shape
    # output tensor is 4d but we want to check first and last dims
    tf.debugging.assert_equal(
        (output_shape, output_shape), (batch_size, n_classes), summarize=2
    )

def test_gradient():
    import tensorflow as tf
    from cifar10 import ResNet, loadData, unpickle, image_augment
    import os
    import numpy as np

    # load data
    directory = 'cifar-10-batches-py'
    images = []
    labels = []
    files = [i for i in os.listdir(directory) if os.path.isfile(os.path.join(directory,i)) and 'data_batch' in i]
    images, labels = loadData(files, images, labels, directory)

    meta_file = r'./cifar-10-batches-py/batches.meta'
    meta_data = unpickle(meta_file)

    train_images = np.concatenate((images[0],images[1],images[2],images[3],images[4]))
    train_labels = np.concatenate((labels[0],labels[1],labels[2],labels[3],labels[4]))
    
    data_batch = unpickle(directory + '/test_batch')
    test_images = data_batch['data']
    test_images = test_images.reshape(len(test_images),3,32,32).transpose(0,2,3,1)
    test_labels = data_batch['labels']

    test_images = test_images/255
    train_images = train_images/255

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    layers = [64,64,128,128,256,256,512,512]
    n_classes = 10
    channels = 3

    resnet = ResNet(channels, layers, n_classes)

    batch_indices = rng.uniform(
            shape=[128], maxval= len(train_images), dtype=tf.int32
        )
    with tf.GradientTape() as tape:
            input_batch = image_augment(tf.gather(train_images, batch_indices), rng)
            label_batch = tf.gather(train_labels, batch_indices)

            label_hat = resnet(input_batch)
            label_hat = tf.cast(label_hat, dtype=tf.float32)
            label_batch = tf.cast(label_batch, dtype=tf.int32)
            
            loss = (
                tf.math.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=label_batch, logits=label_hat
                    )
                )
            )
            

    grads = tape.gradient(loss, resnet.trainable_variables)
    gradient_sum = tf.reduce_sum([tf.reduce_sum(i) for i in grads])
    
    tf.debugging.assert_none_equal(gradient_sum, tf.zeros(shape=tf.shape(gradient_sum)))

# Testing Linear Class
def test_additivity():
    import tensorflow as tf

    from cifar10 import Linear

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 10
    num_outputs = 1

    linear = Linear(num_inputs, num_outputs)

    a = rng.normal(shape=[1, num_inputs])
    b = rng.normal(shape=[1, num_inputs])

    tf.debugging.assert_near(linear(a + b), linear(a) + linear(b), summarize=2)

def test_homogeneity():
    import tensorflow as tf

    from cifar10 import Linear

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 10
    num_outputs = 1
    num_test_cases = 100

    linear = Linear(num_inputs, num_outputs)

    a = rng.normal(shape=[1, num_inputs])
    b = rng.normal(shape=[num_test_cases, 1])

    tf.debugging.assert_near(linear(a * b), linear(a) * b, summarize=2)

@pytest.mark.parametrize("num_outputs", [1, 16, 128])
def test_dimensionality(num_outputs):
    import tensorflow as tf

    from cifar10 import Linear

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 10

    linear = Linear(num_inputs, num_outputs)

    a = rng.normal(shape=[1, num_inputs])
    z = linear(a)

    tf.assert_equal(tf.shape(z)[-1], num_outputs)

# Testing Adam Class
def test_adam():
    import tensorflow as tf
    from cifar10 import ResNet, loadData, unpickle, image_augment, Adam
    import os
    import numpy as np

    # load data
    directory = 'cifar-10-batches-py'
    images = []
    labels = []
    files = [i for i in os.listdir(directory) if os.path.isfile(os.path.join(directory,i)) and 'data_batch' in i]
    images, labels = loadData(files, images, labels, directory)

    meta_file = r'./cifar-10-batches-py/batches.meta'
    meta_data = unpickle(meta_file)

    train_images = np.concatenate((images[0],images[1],images[2],images[3],images[4]))
    train_labels = np.concatenate((labels[0],labels[1],labels[2],labels[3],labels[4]))
    
    data_batch = unpickle(directory + '/test_batch')
    test_images = data_batch['data']
    test_images = test_images.reshape(len(test_images),3,32,32).transpose(0,2,3,1)
    test_labels = data_batch['labels']

    test_images = test_images/255
    train_images = train_images/255

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    layers = [64,64,128,128,256,256,512,512]
    n_classes = 10
    channels = 3

    resnet = ResNet(channels, layers, n_classes)
    adam = Adam(resnet.trainable_variables)

    batch_indices = rng.uniform(
        shape=[128], maxval=tf.shape(train_images)[0], dtype=tf.int32
    )
    with tf.GradientTape() as tape:
        input_batch = tf.cast(tf.gather(train_images, batch_indices), tf.float32)
        label_batch = tf.gather(train_labels, batch_indices)

        label_hat = resnet(input_batch)
        label_hat = tf.cast(label_hat, dtype=tf.float32)
        label_batch = tf.cast(label_batch, dtype=tf.int32)
        
        loss = (
            tf.math.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(label_batch, label_hat)
            )
        )

    grads = tape.gradient(loss, resnet.trainable_variables)

    old_variable_sum = tf.reduce_sum(
        [tf.reduce_sum(i) for i in resnet.trainable_variables]
    )
    adam(1, resnet.trainable_variables, grads)
    new_variable_sum = tf.reduce_sum(
        [tf.reduce_sum(i) for i in resnet.trainable_variables]
    )

    tf.debugging.assert_none_equal(new_variable_sum, old_variable_sum)

# Testing ResBlock Class
def test_output_shape():
    import tensorflow as tf
    from cifar10 import ResBlock, Conv2d

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)
    
    batch_size = 128
    channels = 3
    layers = [64, 64, 64]
    input_conv = Conv2d(tf.Variable(
                rng.normal(
                    shape = [7, 7, channels, layers[0], ], ), 
                    trainable = True, 
                    name = "Conv2d/w", 
                ))

    input_shape = tf.cast(
        rng.normal(shape=[batch_size, 32, 32, channels]), tf.float32
    )
    input = input_conv(input_shape)

    resblock = ResBlock(layers[1], layers[2])
    output_shape = resblock(input).shape

    # output tensor is 4d but we want to check first and last dims
    tf.debugging.assert_equal(
        output_shape, input.shape, summarize=2
    )