# /bin/env python3.8
import pytest


# Testing masking in masked multi head attention
def test_masked_multi_head_attention():
    import tensorflow as tf
    from transformer import MultiHeadAttention
    from tensorflow import linalg, ones, zeros

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    input_seq_length = 5  # Maximum length of the input sequence
    h = 1  # Number of self-attention heads
    d_k = 3  # Dimensionality of the linearly projected queries and keys
    d_v = 3  # Dimensionality of the linearly projected values
    d_model = 3  # Dimensionality of the model sub-layers' outputs
    batch_size = 1  # Batch size from the training process

    queries = rng.normal(shape=[batch_size, input_seq_length, d_k])
    keys = rng.normal(shape=[batch_size, input_seq_length, d_k])
    values = rng.normal(shape=[batch_size, input_seq_length, d_v])
    input = [queries, keys, values]
    """
     triangular matrix mask looks like this (it gets multiplied by 
     -1e9 to act as -inf as mentioned in paper)
      [[0., 1., 1., 1., 1.],
       [0., 0., 1., 1., 1.],
       [0., 0., 0., 1., 1.],
       [0., 0., 0., 0., 1.],
       [0., 0., 0., 0., 0.]]
       """
    mask = 1 - linalg.band_part(
        ones((input_seq_length, input_seq_length)), -1, 0
    )

    attention = MultiHeadAttention(h, d_model)
    with tf.GradientTape() as tape:
        tape.watch(input)  # needed for non-Variables
        multihead_output = attention(input[0], input[1], input[2], mask)
    dy_dx = tape.jacobian(multihead_output, input)

    # dy_dx[2][0][0][0][0][0] not equal to zeroes -> first input token is 
    #   dependant on start token
    # dy_dx[2][0][0][0][0][1] equal to zeroes -> first input token is not 
    #   dependant on second token in sequence
    tf.debugging.assert_none_equal(
        dy_dx[2][0][0][0][0][0],
        zeros(shape=dy_dx[2][0][0][0][0][0].shape),
        summarize=2,
    )
    tf.debugging.assert_equal(
        dy_dx[2][0][0][0][0][1],
        zeros(shape=dy_dx[2][0][0][0][0][1].shape),
        summarize=2,
    )
    # to print out all 5 words print:
    # dy_dx[2][0][0][0][0][1] dy_dx[2][0][1][0][0][1] 
    # dy_dx[2][0][2][0][0][1] dy_dx[2][0][3][0][0][1] 
    # dy_dx[2][0][4][0][0][1]


# Testing transformer prediction of the rest of the input sequence
# This test tests the Decoder and DecoderLayer classes
def test_transformer():
    import tensorflow as tf
    from tqdm import trange
    from transformer import Decoder, Adam, tokenize

    num_iters = 60
    batch_size = 64
    refresh_rate = 1
    dropout_rate = 0.1
    num_hidden_layers = 5
    hidden_layer_widths = 2

    h = 8  # Number of self-attention heads
    d_model = 512  # Dimensionality of the model sub-layers' outputs
    d_ff = 512  # Dimensionality of the inner fully connected layer
    n = 1  # Number of decoder layers

    input_sequences = ["<start> hi prof curro this is my transformer"]
    input_seq_length = len(
        tf.convert_to_tensor(input_sequences).numpy()[0].split()
    )  # Maximum length of the input sequence
    output_sequences = ["hi prof curro this is my transformer <end>"]
    token_dict = {}
    token_count = 0

    for string in input_sequences + output_sequences:
        words = string.split()
        for word in words:
            if str.encode(word.lower()) not in token_dict.keys():
                token_dict[str.encode(word.lower())] = token_count
                token_count = token_count + 1

    transformer = Decoder(
        token_count,
        input_seq_length,
        h,
        d_model,
        d_ff,
        n,
        dropout_rate,
        hidden_layer_widths,
        num_hidden_layers,
        token_dict,
        batch_size,
    )
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
            output_index = tokenize(
                output_sequence, input_seq_length, token_dict
            )
            output_hat = transformer(input_sequence)
            loss = tf.math.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    tf.squeeze(output_index),
                    tf.squeeze(tf.cast(output_hat, tf.float32)),
                )
            )
            grads = tape.gradient(loss, transformer.trainable_variables)
            adam(transformer.trainable_variables, grads, i + 1)

        if i % refresh_rate == (refresh_rate - 1):
            bar.set_description(f"Step {i}; Loss => {loss.numpy():0.4f}")
            bar.refresh()

    test_input = ["<start> hi prof curro"]
    next = []
    for i in range(
        0,
        input_seq_length
        + 1
        - len(tf.convert_to_tensor(test_input).numpy()[0].split()),
    ):
        prediction = transformer(test_input)
        prediction = tf.math.argmax(prediction[0], axis=1)

        predicted_words = []
        for index in prediction:
            predicted_words.append(
                list(token_dict.keys())[
                    list(token_dict.values()).index(index)
                ]
            )
        # print(predicted_words[-1])
        next.append(predicted_words[-1].decode("utf-8"))
        test_input = [
            "".join(
                test_input
                + list(" " + predicted_words[-1].decode("utf-8"))
            )
        ]

    print(" ".join(next))
    tf.debugging.assert_equal(
        " ".join(next), "this is my transformer <end>"
    )


# Testing the positional encoding by visualizaing position matrix
def test_positional_encoding():
    # visualizing the positional matrix on n = 10000 (value in paper)
    # set max sequence length = 512
    import matplotlib.pyplot as plt
    from transformer import positional_encoding

    P = positional_encoding(seq_len=100, d=512, n=10000)
    cax = plt.matshow(P)
    plt.gcf().colorbar(cax)
    plt.suptitle("Fixed Weight Embedding from Attention is All You Need")
    plt.savefig("positional_encoding.png")


# Testing tokenization of inputs (i.e. that the same word gets tokenized 
# to the same token)
def test_tokens():
    import tensorflow as tf
    from funcs import tokenize

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    token_dict = {}
    token_count = 0

    input_sequences = ["<start> man bites dog"]
    output_sequences = ["man bites dog <end>"]
    for string in input_sequences + output_sequences:
        words = string.split()
        for word in words:
            if str.encode(word.lower()) not in token_dict.keys():
                token_dict[str.encode(word.lower())] = token_count
                token_count = token_count + 1

    output_sequence = ["<start> man bites dog"]
    tokens = tokenize(output_sequence, 4, token_dict)
    example2 = ["<start> man dog bites"]
    tokens2 = tokenize(example2, 4, token_dict)

    tf.debugging.assert_equal(tokens[0][3], tokens2[0][2], summarize=2)
