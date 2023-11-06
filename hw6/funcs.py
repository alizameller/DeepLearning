import tensorflow as tf

def tokenize(example, max_length_sequence, dictionary):
    strings = tf.strings.lower(example)
    split_strings = tf.strings.split(strings, " ")
    #dict = {b'sos':0,b'man':1,b'bites':2,b'dog':3,b'<end>':4}
    token_ids = []
    for value in split_strings[0]:
        token_ids.append(dictionary[value.numpy()])
    truncated_token_ids = token_ids[:max_length_sequence]
    return truncated_token_ids

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

class Adam:
    def __init__(
        self, trainable_vars, eta=0.001, alpha=0.001, beta_1=0.9, beta_2=0.999, eps=1e-8
    ):
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