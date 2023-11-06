# /bin/env python3.8
import pytest

# Testing masked multi-head attention Class
def test_masked_multi_head_attention():
    import tensorflow as tf
    from transformer import MultiHeadAttention, DotProductAttention
    import numpy as np
    from tensorflow import linalg, ones, zeros

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    input_seq_length = 5  # Maximum length of the input sequence
    h = 1  # Number of self-attention heads
    d_k = 3  # Dimensionality of the linearly projected queries and keys
    d_v = 3  # Dimensionality of the linearly projected values
    d_model = 3  # Dimensionality of the model sub-layers' outputs
    batch_size = 1  # Batch size from the training process

    queries = rng.normal(shape = [batch_size, input_seq_length, d_k])
    keys = rng.normal(shape =[batch_size, input_seq_length, d_k])
    values = rng.normal(shape = [batch_size, input_seq_length, d_v])
    input = [queries, keys, values]
    '''
     triangular matrix mask looks like this (it gets multiplied by -1e9 to act as -inf as mentioned in paper)
      [[0., 1., 1., 1., 1.],
       [0., 0., 1., 1., 1.],
       [0., 0., 0., 1., 1.],
       [0., 0., 0., 0., 1.],
       [0., 0., 0., 0., 0.]]
       '''
    mask = 1 - linalg.band_part(ones((input_seq_length, input_seq_length)), -1, 0)

    attention = MultiHeadAttention(h, d_k, d_v, d_model, input_seq_length)
    with tf.GradientTape() as tape:
        tape.watch(input)  # needed for non-Variables
        multihead_output = attention(input[0], input[1], input[2], mask)
    dy_dx = tape.jacobian(multihead_output, input)

    # dy_dx[2][0][0][0][0][0] not equal to zeroes -> first input token is dependant on start token
    # dy_dx[2][0][0][0][0][1] equal to zeroes -> first input token is not dependant on second token in sequence
    tf.debugging.assert_none_equal(
        dy_dx[2][0][0][0][0][0], zeros(shape = dy_dx[2][0][0][0][0][0].shape), summarize=2
    )
    tf.debugging.assert_equal(
        dy_dx[2][0][0][0][0][1], zeros(shape = dy_dx[2][0][0][0][0][1].shape), summarize=2
    )
    # print out all 5 words
    # dy_dx[2][0][0][0][0][1] dy_dx[2][0][1][0][0][1] dy_dx[2][0][2][0][0][1] dy_dx[2][0][3][0][0][1] dy_dx[2][0][4][0][0][1]
'''
# testing tokenization of inputs  
def test_tokens():
    import tensorflow as tf
    from funcs import tokenize

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    vocab_size = 5
    embedding_size = 512

    embedding_matrix = tf.Variable(
            rng.normal(shape=[vocab_size, embedding_size]),
            trainable=True,
            name="embedding_matrix",
        )
    example = ["Man bites dog"]
    tokens = tokenize(example, 5)
    example2 = ["Man dog bites"]
    tokens2 = tokenize(example2, 5)
    embeddings = tf.nn.embedding_lookup(embedding_matrix, tokens)
    print(embeddings)

    tf.debugging.assert_equal(tokens[1], tokens2[2], summarize=2)

def test_transformer():
    import tensorflow as tf
    from funcs import tokenize

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    vocab_size = 5
    embedding_size = 512

    embedding_matrix = tf.Variable(
            rng.normal(shape=[vocab_size, embedding_size]),
            trainable=True,
            name="embedding_matrix",
        )
    example = ["Man bites dog"]
    tokens = tokenize(example, 5)
    example2 = ["Man dog bites"]
    tokens2 = tokenize(example2, 5)
    embeddings = tf.nn.embedding_lookup(embedding_matrix, tokens)

    input_seq = random.random((batch_size, input_seq_length))
    decoder = Decoder(vocab_size, input_seq_length, h, d_k, d_v, d_model, d_ff, n, dropout_rate, hidden_layer_width, num_hidden_layers)
    print(decoder(input_seq, None, True))
    adam = Adam(decoder.trainable_variables)
    rng = tf.random.get_global_generator()
    rng.reset_from_seed(0x43966E87BD57227011B5B03B58785EC1)
    bar = trange(num_iters)

    for i in bar:
        with tf.GradientTape() as tape:
            label_hat = decoder(input_seq)
            label_hat = tf.cast(label_hat, dtype=tf.float32)
            label_batch = tf.cast(input_seq, dtype=tf.int32)

            loss = tf.math.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=label_batch, logits=label_hat
                )
            )

        grads = tape.gradient(loss, decoder.trainable_variables)
        adam(i + 1, decoder.trainable_variables, grads)
'''
