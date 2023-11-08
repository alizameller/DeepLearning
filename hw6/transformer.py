#!/bin/env python
import math
import numpy as np
import random
import tensorflow as tf
from tensorflow import math, matmul, reshape, shape, transpose, cast, float32
from tensorflow import linalg, ones, zeros

from mlp import MLP
from linear import Linear
from funcs import dropout2d, layernorm, tokenize, Adam

def getPositionEncoding(seq_len, d, n=10000):
    P = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d/2)):
            denominator = np.power(n, 2*i/d)
            P[k, 2*i] = np.sin(k/denominator)
            P[k, 2*i+1] = np.cos(k/denominator)
    return P

# Implementing the Scaled-Dot Product Attention
class DotProductAttention(tf.Module):
    def __call__(self, queries, keys, values, d_k, mask=None):
        keys_transposed = tf.transpose(keys)
        scores = tf.einsum('ijk, kli -> ijl', queries, keys_transposed) 
        scores /= tf.math.sqrt(tf.convert_to_tensor(d_k, dtype=float32))

        # shape of scores = batch size, num heads, seq length, seq length
        # Apply mask to the attention scores
        if mask is not None:
            scores = scores + (-1e9 * mask)

        # Computing the weights by a softmax operation
        weights = tf.nn.softmax(scores)

        return tf.einsum('ijk, ikl -> ijl', weights, values)
        
# Implementing the Multi-Head Attention
class MultiHeadAttention(tf.Module):
    def __init__(self, h, d_k, d_v, d_model, input_dim):
        self.attention = DotProductAttention()  # Scaled dot product attention
        self.heads = h  # Number of attention heads to use
        self.d_k = d_k  # Dimensionality of the linearly projected queries and keys
        self.d_v = d_v  # Dimensionality of the linearly projected values
        self.d_model = d_model  # Dimensionality of the model
        self.head_dim = tf.cast(self.d_model/self.heads, tf.int32)

        rng = tf.random.get_global_generator()

        self.W_q = tf.Variable( # Learned projection matrix for queries
            rng.normal(shape=[d_model, d_k]),
            trainable=True,
            name="Wq",
        )
        self.W_k = tf.Variable( # Learned projection matrix for the keys
            rng.normal(shape=[d_model, d_k]),
            trainable=True,
            name="Wk",
        )
        self.W_v = tf.Variable( # Learned projection matrix for the values
            rng.normal(shape=[d_model, d_v]),
            trainable=True,
            name="Wv",
        )  
        self.W_o = tf.Variable( # Learned projection matrix for the multi-head output
            rng.normal(shape=[d_v, d_model]),
            trainable=True,
            name="Wo",
        )
    
    def __call__(self, queries, keys, values, mask=None):
        w_queries = tf.nn.relu(tf.einsum("ntk, kq -> ntq", queries, self.W_q))
        w_keys = tf.nn.relu(tf.einsum("ntk, kq -> ntq", keys, self.W_k))
        w_values = tf.nn.relu(tf.einsum("ntk, kq -> ntq", values, self.W_v))
        # expected size: [batch size, seq_length, d_k]
        # Davids expected size: [seq_length, seq_length, batch_size]

        # Compute the multi-head attention output using the reshaped queries, keys and values
        output = self.attention(w_queries, w_keys, w_values, self.d_k, mask)

        # Apply one final linear projection to the output to generate the multi-head attention
        return tf.nn.relu(tf.einsum("ntk, kq -> ntq", output, self.W_o))

# Implementing the Decoder Layer
class DecoderLayer(tf.Module):
    def __init__(self, h, d_k, d_v, d_model, d_ff, rate, sequence_length, hidden_layer_width, num_hidden_layers):
        self.rate = rate
        self.multihead_attention = MultiHeadAttention(h, d_k, d_v, d_model, sequence_length)
        self.feed_forward = MLP(d_ff, d_model, hidden_layer_width, num_hidden_layers) # check num_input and num_output d_ff vs. d_model
        self.gamma_1 = tf.Variable(
            tf.ones(
                shape=[
                    1,
                    1,
                    1,
                    d_model,
                ],
            ),
            trainable=True,
            name="gamma_1",
        )
        self.beta_1 = tf.Variable(
            tf.zeros(
                shape=[
                    1,
                    1,
                    1,
                    d_model,
                ],
            ),
            trainable=True,
            name="beta_1",
        )
        self.gamma_2 = tf.Variable(
            tf.ones(
                shape=[
                    1,
                    1,
                    1,
                    d_model,
                ],
            ),
            trainable=True,
            name="gamma_2",
        )
        self.beta_2 = tf.Variable(
            tf.zeros(
                shape=[
                    1,
                    1,
                    1,
                    d_model,
                ],
            ),
            trainable=True,
            name="beta_2",
        )

    def __call__(self, x, mask, training):
        # Multi-head attention layer
        multihead_output = self.multihead_attention(x, x, x, mask)
        # If trainng, add in a dropout layer
        if training: 
            multihead_output = dropout2d(multihead_output)

        # Followed by an Add & Norm layer
        addnorm_output = layernorm(x + multihead_output, self.gamma_1, self.beta_1) 
        # Fully connected layer
        feedforward_output = self.feed_forward(addnorm_output)
        # If training, add in another dropout layer
        if training:
            feedforward_output = dropout2d(feedforward_output)

        # Followed by another Add & Norm layer
        return layernorm(feedforward_output, self.gamma_2, self.beta_2)

# Implementing the Decoder
class Decoder(tf.Module):
    def __init__(self, vocab_size, sequence_length, h, d_k, d_v, d_model, d_ff, n, rate, hidden_layer_width, num_hidden_layers, dictionary, batch_size):
        rng = tf.random.get_global_generator()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.embedding_matrix = tf.Variable(
            rng.normal(shape=[vocab_size, d_model]),
            trainable=True,
            name="embedding_matrix",
        )
        self.batch_size = batch_size
        self.rate = rate
        self.decoder_layer = [DecoderLayer(h, d_k, d_v, d_model, d_ff, rate, sequence_length, hidden_layer_width, num_hidden_layers) for _ in range(n)]
        self.dictionary = dictionary
        #self.final_layer = Linear(d_model, vocab_size)
        self.W_f = tf.Variable( 
            rng.normal(shape=[d_model, vocab_size]),
            trainable=True,
            name="Wf",
        )

    def __call__(self, input_sentence, training):
        # Generate the tokens and positional encoding
        shape = len(tf.convert_to_tensor(input_sentence).numpy()[0].split())
        
        token_ids = tokenize(input_sentence, self.sequence_length, self.dictionary)
        embeddings = tf.nn.embedding_lookup(self.embedding_matrix, token_ids)
        pos = getPositionEncoding(shape, shape, n=100)
        embeddings = tf.transpose(tf.transpose(embeddings) + pos)
        mask = 1 - tf.linalg.band_part(ones((shape, shape)), -1, 0)
        # mask = tf.broadcast_to(mask, [self.batch_size, shape, shape])
        # If training the model, add in a dropout layer
        if training:
            x = dropout2d(embeddings)
        else:
            x = embeddings

        # Pass on the positional encoded values to each encoder layer
        for layer in self.decoder_layer:
            x = layer(x, mask, training)
        # x.shape = [64, 4, 512] (i.e. batch_size, vocab_size, d_model)
        # self.W_f.shape = [512, 4] (i.e. d_model, vocab_size)
        x = tf.nn.relu(tf.einsum("ntk, kq -> ntq", x, self.W_f))
        return x

if __name__ == "__main__":
    from tqdm import trange

    num_iters = 40
    batch_size = 64
    refresh_rate = 10
    num_hidden_layers = 5
    hidden_layer_widths = 2

    h = 8  # Number of self-attention heads
    d_k = 64  # Dimensionality of the linearly projected queries and keys
    d_v = 64  # Dimensionality of the linearly projected values
    d_ff = 512  # Dimensionality of the inner fully connected layer
    d_model = 512  # Dimensionality of the model sub-layers' outputs
    n = 1  # Number of layers in the encoder stack
    dropout_rate = 0.1  # Frequency of dropping the input units in the dropout layers
    #dec_vocab_size = 11 # Vocabulary size for the decoder
    input_seq_length = 10  # Maximum length of the input sequence
    input_sequences = ["SOS My name is david and I love ice cream"]
    output_sequences = ["My name is david and I love ice cream <end>"]
    token_dict = {}
    token_count = 0
    for string in (input_sequences + output_sequences):
        words = string.split()
        for word in words:
            if str.encode(word.lower()) not in token_dict.keys():
                token_dict[str.encode(word.lower())] = token_count
                token_count = token_count + 1
    transformer = Decoder(token_count, input_seq_length, h, d_k, d_v, d_model, d_ff, n, dropout_rate, hidden_layer_widths, num_hidden_layers, token_dict, batch_size)
    adam = Adam(transformer.trainable_variables)
    bar = trange(num_iters)
    
    rng = tf.random.get_global_generator()
    rng.reset_from_seed(0x43966E87BD57227011B5B03B58785EA)
    for i in bar:
        batch_indices = rng.uniform(
            shape=[batch_size], maxval=len(input_sequences), dtype=tf.int32
        )
        
        with tf.GradientTape() as tape:
            input_sequence = tf.gather(input_sequences, batch_indices)
            output_sequence = tf.gather(output_sequences, batch_indices)
            output_index = tokenize(output_sequence, input_seq_length, token_dict)
            output_hat = transformer(input_sequence, True)
            #breakpoint()
            loss = tf.math.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(tf.squeeze(output_index), tf.squeeze(tf.cast(output_hat,tf.float32)))
            )
            grads = tape.gradient(loss, transformer.trainable_variables)
            adam(transformer.trainable_variables, grads, i + 1)

        if i % refresh_rate == (refresh_rate - 1):
            bar.set_description(
                f"Step {i}; Loss => {loss.numpy():0.4f}"
            )
            bar.refresh()

    test_input = ["SOS my name is david and I"]
    prediction = transformer(test_input, False)
    prediction = tf.math.argmax(prediction[0],axis=1)
    predicted_words = []
    print(token_dict)
    print(prediction)
    #breakpoint()
    for index in prediction:
        predicted_words.append(list(token_dict.keys())[list(token_dict.values()).index(index)])
    print(predicted_words)