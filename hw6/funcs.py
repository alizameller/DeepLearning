import tensorflow as tf

def dropout2d(x, rate=0.1, seed=4567897):
    return tf.nn.dropout(x, rate, seed=seed)

def layernorm(x, gamma, beta, eps=1e-5):
    # features of x are N = batch size, H = height, W = width, and C = channels. 
    mean, var = tf.nn.moments(x, [1, 2, 3], keepdims=True) # normalization is performed across spatial dims (N, H, W)
    x = (x - mean) / tf.sqrt(var + eps)
    return x * gamma + beta 

def groupnorm(x, gamma, G, beta, eps=1e-5):
    # x: input features with shape [N,C,H,W]
    # gamma, beta: learnable scale and offset, with shape [1,C,1,1] # G: number of groups for GN
    N, H, W, C = x.shape
    # breakpoint()
    x = tf.reshape(x, [N, G, C // G, H, W])

    mean, var = tf.nn.moments(x, [2, 3, 4], keepdims=True)
    x = (x - mean) / tf.sqrt(var + eps)
    x = tf.reshape(x, [N, H, W, C])

    return x * gamma + beta