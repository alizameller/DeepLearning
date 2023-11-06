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

# Implementing the Scaled-Dot Product Attention
class DotProductAttention(tf.Module):
    def __call__(self, queries, keys, values, d_k, mask=None):
        keys_transposed = tf.transpose(keys)
        scores = tf.einsum('ijk, kli -> ijl', queries, keys_transposed) 
        scores /= tf.math.sqrt(tf.convert_to_tensor(d_k, dtype=float32))
        # Apply mask to the attention scores
        if mask is not None:
            scores += -1e9 * mask

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
            rng.normal(shape=[d_k, self.head_dim]),
            trainable=True,
            name="Wq",
        )
        self.W_k = tf.Variable( # Learned projection matrix for the keys
            rng.normal(shape=[d_k, self.head_dim]),
            trainable=True,
            name="Wk",
        )
        self.W_v = tf.Variable( # Learned projection matrix for the values
            rng.normal(shape=[d_v, self.head_dim]),
            trainable=True,
            name="Wk",
        )  
        self.W_o = tf.Variable( # Learned projection matrix for the multi-head output
            rng.normal(shape=[d_k, self.head_dim]),
            trainable=True,
            name="Wk",
        )

    def __call__(self, queries, keys, values, mask=None):
        breakpoint()
        queries = tf.nn.relu(tf.einsum("ntk, kq -> ntq", queries, self.W_q))
        keys = tf.nn.relu(tf.einsum("ntk, kq -> ntq", keys, self.W_k))
        values = tf.nn.relu(tf.einsum("ntk, kq -> ntq", values, self.W_v))
        # expected size: [batch size, seq_length, d_k]

        # Compute the multi-head attention output using the reshaped queries, keys and values
        output = self.attention(queries, keys, values, self.d_k, mask)

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
                    d_ff,
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
                    d_ff,
                ],
            ),
            trainable=True,
            name="beta_2",
        )

    def __call__(self, x, padding_mask):
        # Multi-head attention layer
        multihead_output = self.multihead_attention(x, x, x, padding_mask)
        # Add in a dropout layer
        multihead_output = dropout2d(multihead_output)
        # Followed by an Add & Norm layer
        addnorm_output = layernorm(x + multihead_output, self.gamma_1, self.beta_1) 
        # Fully connected layer
        feedforward_output = self.feed_forward(addnorm_output)
        # Add in another dropout layer
        feedforward_output = dropout2d(feedforward_output)
        # Followed by another Add & Norm layer
        return layernorm(addnorm_output + feedforward_output, self.gamma_2, self.beta_2)

# Implementing the Decoder
class Decoder(tf.Module):
    def __init__(self, vocab_size, sequence_length, h, d_k, d_v, d_model, d_ff, n, rate, hidden_layer_width, num_hidden_layers, embedding_size, dictionary, batch_size):
        rng = tf.random.get_global_generator()
        self.embedding_matrix = tf.Variable(
            rng.normal(shape=[batch_size, vocab_size, embedding_size]),
            trainable=True,
            name="embedding_matrix",
        )
        self.rate = rate
        self.decoder_layer = [DecoderLayer(h, d_k, d_v, d_model, d_ff, rate, sequence_length, hidden_layer_width, num_hidden_layers) for _ in range(n)]
        self.dictionary = dictionary

    def __call__(self, input_sentence, padding_mask, sequence_length):
        breakpoint()
        # Generate the tokens and positional encoding
        token_ids = tokenize(input_sentence, sequence_length, self.dictionary)
        embeddings = tf.nn.embedding_lookup(self.embedding_matrix, token_ids)

        # Add in a dropout layer
        x = dropout2d(embeddings)

        # Pass on the positional encoded values to each encoder layer
        for layer in self.decoder_layer:
            x = layer(x, padding_mask)

        return x

if __name__ == "__main__":
    from numpy import random
    from tqdm import trange

    h = 8
    d_k = 64  # Dimensionality of the linearly projected queries and keys
    d_v = 64  # Dimensionality of the linearly projected values
    d_ff = 512  # Dimensionality of the inner fully connected layer
    d_model = 512  # Dimensionality of the model sub-layers' outputs
    n = 1  # Number of layers in the encoder stack
    dropout_rate = 0.1  # Frequency of dropping the input units in the dropout layers
    dec_vocab_size = 5 # Vocabulary size for the decoder
    input_seq_length = 4  # Maximum length of the input sequence
    input_sequences = ["SOS Man bites dog"]
    output_sequences = ["Man bites dog <end>"]
    
    token_dict = {}
    token_count = 0
    hidden_layer_width = 2
    num_hidden_layers = 5
    embedding_size = 512

    refresh_rate = 10
    batch_size = 64  # Batch size from the training process
    dropout_rate = 0.1  # Frequency of dropping the input units in the dropout layers
    num_iters = 1000

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    for string in (input_sequences + output_sequences):
        words = string.split()
        for word in words:
            if str.encode(word.lower()) not in token_dict.keys():
                token_dict[str.encode(word.lower())] = token_count
                token_count = token_count + 1
    transformer = Decoder(dec_vocab_size, input_seq_length, h, d_k, d_v, d_model, d_ff, n, dropout_rate, hidden_layer_width, num_hidden_layers, embedding_size, token_dict, batch_size)
    adam = Adam(transformer.trainable_variables)
    bar = trange(num_iters)

    for i in bar:
        batch_indices = rng.uniform(
            shape=[batch_size], maxval=len(input_sequences), dtype=tf.int32
        )
        
        with tf.GradientTape() as tape:
            input_sequence = tf.gather(input_sequences, batch_indices)
            output_sequence = tf.gather(output_sequences, batch_indices)
            output_index = tokenize(output_sequence, input_seq_length, token_dict)
            mask = 1 - linalg.band_part(ones((input_seq_length, input_seq_length)), -1, 0)
            output_hat = transformer(input_sequence, mask, input_seq_length)
            loss = tf.math.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(tf.squeeze(output_index), tf.squeeze(tf.cast(output_hat,tf.float32)))
            )
            grads = tape.gradient(loss, transformer.trainable_variables)
           # breakpoint()
            adam(transformer.trainable_variables, grads, i + 1)

        if i % refresh_rate == (refresh_rate - 1):
            bar.set_description(
                f"Step {i}; Loss => {loss.numpy():0.4f}"
            )
            bar.refresh()
    test_input = ["SOS Man bites dog"]
    prediction = transformer(test_input,False)
    print(prediction)