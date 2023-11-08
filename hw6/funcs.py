import tensorflow as tf

def tokenize(example, max_length_sequence,dictionary):
    #breakpoint()
    strings = tf.strings.lower(example)
    split_strings = tf.strings.split(strings, " ")
    #dict = {b'sos':0,b'man':1,b'bites':2,b'dog':3,b'<end>':4}
    token_ids = []
    for sentence in split_strings:
        temp_ids = []
        for value in sentence:
            temp_ids.append(dictionary[value.numpy()])
        token_ids.append(temp_ids)
    truncated_token_ids = token_ids[:max_length_sequence]
    return truncated_token_ids

def getPositionEncoding(seq_len, d, n=10000):
    P = tf.zeros((seq_len, d))
    for k in range(seq_len):
        for i in tf.arange(int(d/2)):
            denominator = tf.power(n, 2*i/d)
            P[k, 2*i] = tf.sin(k/denominator)
            P[k, 2*i+1] = tf.cos(k/denominator)
    return P

def dropout2d(x, rate=0.1, seed=4567897):
    return tf.nn.dropout(x, rate, seed=seed)

def layernorm(x, gamma, beta, eps=1e-5):
    # features of x are N = batch size, H = height, W = width, and C = channels. 
    x = tf.expand_dims(x,axis=0)
    # x: input features with shape [N,H,W,C]
    # gamma, beta: scale and offset, with shape [1,C,1,1]
    # G: number of groups for GN
    N, H, W, C = x.shape
    x = tf.reshape(x, [N, C, 1, H, W])
    mean, var = tf.nn.moments(x, [2, 3, 4], keepdims=True)
    x = (x - mean) / tf.sqrt(var + eps)
    x = tf.reshape(x, [N, H, W, C])
    x = x * gamma + beta

    return tf.squeeze(x, axis = 0)

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
        self, variables, eta=0.001, alpha=0.001, beta_1=0.9, beta_2=0.999, eps=1e-8):
        self.m_list = [tf.zeros(tf.shape(variable).numpy()) for variable in variables]
        self.v_list = [tf.zeros(tf.shape(variable).numpy()) for variable in variables]
        self.alpha = alpha
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        self.eta = eta

    def __call__(self, variables, grads, t):
        new_m_list = []
        new_v_list = []
        for m, v, var, grad in zip(self.m_list, self.v_list, variables, grads):
            grad = tf.convert_to_tensor(grad)
            m = self.beta_1 * m + (1 - self.beta_1) * grad
            v = self.beta_2 * v + (1 - self.beta_2) * (grad**2)
            new_m_list.append(m)
            new_v_list.append(v)
            m_hat = m / (1 - self.beta_1**t)
            v_hat = v / (1 - self.beta_2**t)
            var.assign_sub((self.alpha * m_hat) / ((v_hat**0.5) + self.eps))
        self.m_list = new_m_list
        self.v_list = new_v_list