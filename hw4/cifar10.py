import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import os
import math
import platform
import pickle

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

def maxpool2d(x, kernel_size, stride, padding):
    return tf.nn.max_pool(x, ksize=(kernel_size, kernel_size), strides=(stride, stride), padding=padding)

def dropout2d(x, rate=0.1, seed=4567897):
    return tf.nn.dropout(x, rate, seed=seed)

def groupnorm(x, gamma, G, beta, eps = 1e-5):
    # x: input features with shape [N,C,H,W]
    # gamma, beta: learnable scale and offset, with shape [1,C,1,1] # G: number of groups for GN
    N, H, W, C = x.shape
    #breakpoint()
    x = tf.reshape(x, [N, G, C // G, H, W])

    mean, var = tf.nn.moments(x, [2, 3, 4], keepdims=True) 
    x = (x - mean) / tf.sqrt(var + eps)
    x = tf.reshape(x, [N, H, W, C]) 

    return x * gamma + beta

class Linear(tf.Module):
    def __init__(self, num_inputs, num_outputs, bias=True):

        self.w = tf.Variable(
            tf.zeros(shape=[num_inputs, num_outputs]),
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

# strides is the number of pixels by which the filter is shifted during each step
class Conv2d(tf.Module):
    def __init__(self,kernel):
        self.kernel = kernel

    def __call__(self, input, stride=1, spacing="VALID"):
        x = tf.nn.conv2d(input, self.kernel, stride, spacing)
        return x
    
# modified the classifier class from hw3
class ResNet(tf.Module):
    def __init__(
        self,
        input_depth,
        layer_depths,
        num_classes,
    ):
        self.num_classes = num_classes
        self.layer_depths = layer_depths
        self.input_depth = input_depth

        self.rng = tf.random.get_global_generator()
        # 7x7 convolution happens first according to paper's diagram
        self.input_conv_layer = Conv2d(tf.Variable(
            self.rng.normal(
                shape = [7, 7, input_depth, layer_depths[0], ], ), 
                trainable = True, 
                name = "Conv2d/w", 
            ))
        # make number of residual blocks for each layer depth inputted
        self.resnet = [ResBlock(layer_depths[i - 1], layer_depths[i]) for i in range(1, len(layer_depths))]
        # fully connected layer
        self.fc = Linear(layer_depths[-1],num_classes)

    def __call__(self, x):
        # layer 1
        output = self.input_conv_layer(tf.cast(x, dtype=tf.float32), stride=1, spacing='SAME')
        # iterate over resblocks to perform operations in those blocks
        for block_num in range(len(self.resnet)):
            output = self.resnet[block_num](output)
        output = tf.squeeze(output)
        output = self.fc(output)

        return output 
    
class ResBlock(tf.Module):
    def __init__(self, in_channels, out_channels, stride = 1, identity_downsample = False):
        self.rng = tf.random.get_global_generator()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride 
        self.trainable_size = self.out_channels
        self.identity_downsample = identity_downsample

        if self.in_channels != self.trainable_size:
            self.identity_downsample = True
            self.stride = 2
        
        # initialize trainable variables 
        # 2 for each group norm, initiate gamma to ones (identity) and beta to zeros (no offset)
        self.gamma_1 = tf.Variable(tf.ones(shape=[1, 1, 1, self.in_channels, ], ),
                trainable=True,
                name="ResBlock/gamma_1",
            )
        self.beta_1 = tf.Variable(tf.zeros(shape=[1, 1, 1, self.in_channels, ], ),
            trainable=True,
            name="ResBlock/beta_1",
            )
        # out channels for shape here because gamma 2 and beta 2 are used in the group norm at the beginning of the next iteration of calling 
        self.gamma_2 = tf.Variable(tf.ones(shape=[1, 1, 1, self.out_channels, ], ),
                trainable=True,
                name="ResBlock/gamma_2",
            )
        self.beta_2 = tf.Variable(tf.zeros(shape=[1, 1, 1, self.out_channels, ], ),
            trainable=True,
            name="ResBlock/beta_2",
            )
        
        # conv1 and conv2 are building blocks 
        self.conv1 = Conv2d(tf.Variable(
            self.rng.normal(shape = [3, 3, in_channels, out_channels, ], 
            stddev = tf.math.sqrt(2/(in_channels * out_channels))
                ),
                trainable=True,
                name="Conv2d/w1",
        ))
        self.conv2 = Conv2d(tf.Variable(
            self.rng.normal(shape = [3, 3, out_channels, out_channels, ], 
            stddev = tf.math.sqrt(2/(out_channels * out_channels))
                ),
                trainable=True,
                name="Conv2d/w2",
        ))
        # if changing dimensions, downsample dimensions to correct size for the skip connection to work
        if self.identity_downsample: 
            self.conv3 = Conv2d(tf.Variable(
                self.rng.normal(shape = [1, 1, in_channels, out_channels, ], 
                stddev = tf.math.sqrt(2/(in_channels * out_channels))
                    ),
                    trainable=True,
                    name="Conv2d/w_skip",
            ))

    def __call__(self, x):
        # first convolution (7x7) was performed in ResNet, continue here with skip connection
        if self.identity_downsample: 
            x = maxpool2d(x, 2, 2, padding="SAME")
        skip = x
        x = groupnorm(x, self.gamma_1, 32, self.beta_1)
        x = tf.nn.relu(x)

        # basic block - first convolution
        x = self.conv1(x, stride=self.stride, spacing = "SAME")
        x = groupnorm(x, self.gamma_2, 32, self.beta_2)
        x = tf.nn.relu(x)

        # basic block - second convolution
        x = self.conv2(x, spacing = "SAME")

        # check if downsample is necessary
        if self.identity_downsample:
            skip = self.conv3(skip, stride=self.stride, spacing="SAME")

        x = x + skip 

        return x

class Adam:
    def __init__(self, trainable_vars, eta=0.001, alpha=0.001, beta_1=0.9, beta_2=0.999, eps=1e-8):
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
        self.eta = eta

    def __call__(self, t, variables, grads):
        new_m_list = []
        new_v_list = []

        for m, v, var, grad in zip(self.m_list, self.v_list, variables, grads):
            m = self.beta_1 * m + (1 - self.beta_1) * grad
            v = self.beta_2 * v + (1 - self.beta_2) * (grad**2)
            new_m_list.append(m)
            new_v_list.append(v)
            m_hat = m / (1 - self.beta_1**t)
            v_hat = v / (1 - self.beta_2**t)
            var.assign_sub((self.alpha * m_hat) / ((v_hat**0.5) + self.eps))
        self.m_list = new_m_list
        self.v_list = new_v_list

def loadData(files, images, labels):
    for filename in files:
        # for some reason, there is a list variable in here, so skip over that
        if isinstance(filename, list):
            continue   
        data_batch = unpickle(directory + '/' + filename)
        image_data = data_batch['data']
        images.append(image_data.reshape(len(image_data),3,32,32).transpose(0,2,3,1))
        labels.append(data_batch['labels'])
    return images, labels

def image_augment(input_images,rng):
    new_images = []
    for input_image in input_images:
        crop_num = rng.uniform(shape=[1], minval=0,maxval=9)
        if crop_num.numpy() <=2:
            input_image = tf.image.central_crop(input_image,0.75)
            input_image = tf.image.resize_with_pad(input_image,32,32)

        flip_num  = rng.uniform(shape=[1], minval=0,maxval=9)
        if flip_num.numpy() <=2:
            input_image = tf.image.flip_left_right(input_image)
        
        saturate_num  = rng.uniform(shape=[1], minval=0,maxval=9)
        if saturate_num.numpy() <=2:
            input_image = tf.image.adjust_saturation(input_image,3)

        brightness_num = rng.uniform(shape=[1], minval=0,maxval=9)
        if brightness_num.numpy() <=2:
            input_image = tf.image.adjust_brightness(input_image,0.4)

        rot_num = rng.uniform(shape=[1], minval=0,maxval=9)
        if rot_num.numpy() <=2:
            input_image = tf.image.rot90(input_image)
        
        new_images.append(input_image)
    return input_images

if __name__ == "__main__":
    import numpy as np
    from tqdm import trange

    # make sure mnist data files are in current directory
    # import required module

    # assign directory
    directory = 'cifar-10-batches-py'
    
    # iterate over files in
    # that directory
    images = []
    labels = []
    files = [i for i in os.listdir(directory) if os.path.isfile(os.path.join(directory,i)) and 'data_batch' in i]
    images, labels = loadData(files, images, labels)

    meta_file = r'./cifar-10-batches-py/batches.meta'
    meta_data = unpickle(meta_file)
    label_names = meta_data['label_names']

    train_images = np.concatenate((images[0],images[1],images[2],images[3],images[4]))
    train_labels = np.concatenate((labels[0],labels[1],labels[2],labels[3],labels[4]))
    
    data_batch = unpickle(directory + '/test_batch')
    test_images = data_batch['data']
    test_images = test_images.reshape(len(test_images),3,32,32).transpose(0,2,3,1)
    test_labels = data_batch['labels']
    
    '''
    # define figure
    rows, columns = 5, 5
    fig=plt.figure(figsize=(10, 10))
    # visualize these random images
    for i in range(1, columns*rows +1):
        fig.add_subplot(rows, columns, i)
        j = np.random.choice(range(len(test_images)))
        plt.imshow((test_images[j:j+1][0]))
        plt.xticks([])
        plt.yticks([])
        plt.title("{}"
            .format(label_names[test_labels[j]]))
    plt.show()
    exit()'''

    # normalize train_image_data 
    test_images = test_images/255
    # normalize train_image_data 
    train_images = train_images/255
    num_train_images = len(train_images)

    # hyper parameters
    step_size = 0.001
    batch_size = 516
    num_iters = 3000
    refresh_rate = 10
    validation_size = 1700

    layer_depths = [64,64,128,128,256,256,512,512]

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(0x43966E87BD57227011B5B03B58785EC1)
    bar = trange(num_iters)

    eta_changed = False
    second_eta_changed = False

    resnet = ResNet(3, layer_depths, 10)
    print("num_params", tf.math.add_n([tf.math.reduce_prod(var.shape) for var in resnet.trainable_variables]))
    adam = Adam(resnet.trainable_variables)

    for i in bar:
        batch_indices = rng.uniform(
            shape=[batch_size], maxval= num_train_images - validation_size, dtype=tf.int32
        )
        validation_indices = rng.uniform(
            shape=[validation_size], minval=validation_size+1, maxval = num_train_images , dtype=tf.int32
        )

        with tf.GradientTape() as tape:
            input_batch = image_augment(tf.gather(train_images, batch_indices), rng)
            label_batch = tf.gather(train_labels, batch_indices)
            
            validation_input = train_images[num_train_images-validation_size:-1]
            validation_labels = train_labels[num_train_images-validation_size:-1]

            validation_labels_hat = resnet(validation_input)
            validation_labels_hat = tf.math.argmax(validation_labels_hat, axis=1)
            sum = 0

            for label, pred in zip(validation_labels, validation_labels_hat):
                if label == pred.numpy():
                    sum += 1
            accuracy = sum/len(validation_labels) 
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
            
        if accuracy > 0.6 and not eta_changed:
            adam.eta *= 0.5
            eta_changed = True
            print("Learning rate changed")

        if accuracy > 0.7 and eta_changed and not second_eta_changed:
            adam.eta *= 0.5
            second_eta_changed = True
            print("Learning rate changed")
    
        grads = tape.gradient(loss, resnet.trainable_variables)
        adam(i + 1, resnet.trainable_variables, grads)

        if i % refresh_rate == (refresh_rate - 1):
            bar.set_description(
                f"Step {i}; Loss => {loss.numpy():0.4f}, Accuracy => {accuracy:0.4f} step_size => {step_size:0.4f}"
            )   
            bar.refresh()

    # KAHAN CANT HANDLE BIG TENSORS NEED TO SPLIT UP TEST BATCHES
    test_labels_hat_1 = resnet(test_images[0:2000])
    test_labels_hat_2 = resnet(test_images[2000 : 2000 * 2])
    test_labels_hat_3 = resnet(test_images[2000 * 2 : 2000 * 3])
    test_labels_hat_4 = resnet(test_images[2000 * 3 : 2000 * 4])
    test_labels_hat_5 = resnet(test_images[2000 * 4 : 2000 * 5])
    test_labels_hat = tf.concat(
        [
            test_labels_hat_1,
            test_labels_hat_2,
            test_labels_hat_3,
            test_labels_hat_4,
            test_labels_hat_5,
        ],
        axis=0,
    )
    print(test_labels_hat.shape)
    test_labels_hat = tf.math.argmax(test_labels_hat, axis=1)
    sum = 0
    for label, pred in zip(test_labels, test_labels_hat):
        if label == pred.numpy():
            sum += 1
    accuracy = sum / len(test_labels)
    print(accuracy)




