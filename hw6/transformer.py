#!/bin/env python
import math
import numpy as np
import random
import tensorflow as tf
from tensorflow import math, matmul, reshape, shape, transpose, cast, float32
from positional_encoding import PositionEmbeddingFixedWeights
from sentence_transformers import SentenceTransformer

from mlp import MLP
from linear import Linear
from funcs import dropout2d, layernorm

# Implementing the Scaled-Dot Product Attention
class DotProductAttention(tf.nn.module):
    def call(self, queries, keys, values, d_k, mask=None):
        # Scoring the queries against the keys after transposing the latter, and scaling
        scores = matmul(queries, keys, transpose_b=True) / math.sqrt(cast(d_k, float32))

        # Apply mask to the attention scores
        if mask is not None:
            scores += -1e9 * mask

        # Computing the weights by a softmax operation
        weights = tf.nn.softmax(scores)

        # Computing the attention by a weighted sum of the value vectors
        return matmul(weights, values)
        
# Implementing the Multi-Head Attention
class MultiHeadAttention(tf.nn.module):
    def __init__(self, h, d_k, d_v, d_model, input_dim):
        self.attention = DotProductAttention()  # Scaled dot product attention
        self.heads = h  # Number of attention heads to use
        self.d_k = d_k  # Dimensionality of the linearly projected queries and keys
        self.d_v = d_v  # Dimensionality of the linearly projected values
        self.d_model = d_model  # Dimensionality of the model
        self.W_q = Linear(input_dim, d_k)  # Learned projection matrix for the queries
        self.W_k = Linear(input_dim, d_k)  # Learned projection matrix for the keys
        self.W_v = Linear(input_dim, d_v)  # Learned projection matrix for the values
        self.W_o = Linear(input_dim, d_model)  # Learned projection matrix for the multi-head output

    def reshape_tensor(self, x, heads, flag):
        if flag:
            # Tensor shape after reshaping and transposing: (batch_size, heads, seq_length, -1)
            x = reshape(x, shape=(shape(x)[0], shape(x)[1], heads, -1))
            x = transpose(x, perm=(0, 2, 1, 3))
        else:
            # Reverting the reshaping and transposing operations: (batch_size, seq_length, d_k)
            x = transpose(x, perm=(0, 2, 1, 3))
            x = reshape(x, shape=(shape(x)[0], shape(x)[1], self.d_k))
        return x

    def call(self, queries, keys, values, mask=None):
        # Rearrange the queries to be able to compute all heads in parallel
        q_reshaped = self.reshape_tensor(self.W_q(queries), self.heads, True)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)

        # Rearrange the keys to be able to compute all heads in parallel
        k_reshaped = self.reshape_tensor(self.W_k(keys), self.heads, True)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)

        # Rearrange the values to be able to compute all heads in parallel
        v_reshaped = self.reshape_tensor(self.W_v(values), self.heads, True)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)

        # Compute the multi-head attention output using the reshaped queries, keys and values
        o_reshaped = self.attention(q_reshaped, k_reshaped, v_reshaped, self.d_k, mask)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)

        # Rearrange back the output into concatenated form
        output = self.reshape_tensor(o_reshaped, self.heads, False)
        # Resulting tensor shape: (batch_size, input_seq_length, d_v)

        # Apply one final linear projection to the output to generate the multi-head attention
        # Resulting tensor shape: (batch_size, input_seq_length, d_model)
        return self.W_o(output)

# Implementing the Encoder Layer
class EncoderLayer(tf.nn.module):
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

    def call(self, x, padding_mask, training):
        # Multi-head attention layer
        multihead_output = self.multihead_attention(x, x, x, padding_mask)
        # Add in a dropout layer
        multihead_output = dropout2d(multihead_output, self.rate, training=training)
        # Followed by an Add & Norm layer
        addnorm_output = layernorm(x  + multihead_output, self.gamma_1, self.beta_1) 
        # Fully connected layer
        feedforward_output = self.feed_forward(addnorm_output)
        # Add in another dropout layer
        feedforward_output = dropout2d(feedforward_output, self.rate, training=training)
        # Followed by another Add & Norm layer
        return layernorm(addnorm_output + feedforward_output, self.gamma_2, self.beta_2)

# Implementing the Encoder
class Encoder(tf.nn.module):
    def __init__(self, vocab_size, sequence_length, h, d_k, d_v, d_model, d_ff, n, rate, hidden_layer_width, num_hidden_layers):
        self.pos_encoding = PositionEmbeddingFixedWeights(sequence_length, vocab_size, d_model)
        self.rate = rate
        self.encoder_layer = [EncoderLayer(h, d_k, d_v, d_model, d_ff, rate, sequence_length, hidden_layer_width, num_hidden_layers) for _ in range(n)]

    def call(self, input_sentence, padding_mask, training):
        # Generate the positional encoding
        pos_encoding_output = self.pos_encoding(input_sentence)
        # Expected output shape = (batch_size, sequence_length, d_model)

        # Add in a dropout layer
        x = dropout2d(pos_encoding_output, self.rate, training=training)

        # Pass on the positional encoded values to each encoder layer
        for layer in enumerate(self.encoder_layer):
            x = layer(x, padding_mask, training)

        return x

if __name__ == "__main__":
    from numpy import random

    enc_vocab_size = 20 # Vocabulary size for the encoder
    input_seq_length = 5  # Maximum length of the input sequence
    h = 8  # Number of self-attention heads
    d_k = 64  # Dimensionality of the linearly projected queries and keys
    d_v = 64  # Dimensionality of the linearly projected values
    d_ff = 2048  # Dimensionality of the inner fully connected layer
    d_model = 512  # Dimensionality of the model sub-layers' outputs
    n = 6  # Number of layers in the encoder stack
    hidden_layer_width = 256
    num_hidden_layers = 4

    batch_size = 64  # Batch size from the training process
    dropout_rate = 0.1  # Frequency of dropping the input units in the dropout layers

    input_seq = random.random((batch_size, input_seq_length))

    encoder = Encoder(enc_vocab_size, input_seq_length, h, d_k, d_v, d_model, d_ff, n, dropout_rate, hidden_layer_width, num_hidden_layers)
    print(encoder(input_seq, None, True))