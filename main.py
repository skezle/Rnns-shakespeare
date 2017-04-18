"""
Imports
"""
import numpy as np
import tensorflow as tf
import matplotlib
## %matplotlib inline ## ipython notebook
matplotlib.use('Agg') # for server w/o display
import matplotlib.pyplot as plt
import time
import os
import urllib
## from tensorflow.models.rnn.ptb import reader
## replaced with custom code here: https://gist.github.com/spitis/2dd1720850154b25d2cec58d4b75c4a0

from reader import *

def generate_characters(g, checkpoint, num_chars, prompt='A', pick_top_chars=None):
    """ Accepts a current character, initial state"""

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        g['saver'].restore(sess, checkpoint)

        state = None
        current_char = vocab_to_idx[prompt]
        chars = [current_char]

        for i in range(num_chars):
            if state is not None:
                feed_dict={g['x']: [[current_char]], g['init_state']: state}
            else:
                feed_dict={g['x']: [[current_char]]}

            preds, state = sess.run([g['preds'],g['final_state']], feed_dict)

            if pick_top_chars is not None:
                p = np.squeeze(preds)
                p[np.argsort(p)[:-pick_top_chars]] = 0
                p = p / np.sum(p)
                current_char = np.random.choice(vocab_size, 1, p=p)[0]
            else:
                current_char = np.random.choice(vocab_size, 1, p=np.squeeze(preds))[0]

            chars.append(current_char)

    chars = map(lambda x: idx_to_vocab[x], chars)
    print("".join(chars))
    return("".join(chars))

def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()

def gen_epochs(n, num_steps, batch_size):
    for i in range(n):
        ##yield ptb_iterator(data, batch_size, num_steps)
        yield shuffled_ptb_iterator(data, batch_size, num_steps)

def train_network(g, num_epochs, num_steps = 200, batch_size = 32, verbose = True, save=False):
    tf.set_random_seed(2345)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        training_losses = []
        for idx, epoch in enumerate(gen_epochs(num_epochs, num_steps, batch_size)):
            training_loss = 0
            steps = 0
            training_state = None
            for X, Y in epoch:
                steps += 1
                print("steps: {}".format(steps))
                print("X shape: {}".format(X.shape))
                print("Y shape: {}".format(Y.shape))
                feed_dict={g['x']: X, g['y']: Y}
                if training_state is not None:
                    feed_dict[g['init_state']] = training_state
                training_loss_, training_state, _ = sess.run([g['total_loss'],
                                                      g['final_state'],
                                                      g['train_step']],
                                                             feed_dict)
                training_loss += training_loss_
            if verbose:
                print("Average training loss for Epoch", idx, ":", training_loss/steps)
            training_losses.append(training_loss/steps)

        if isinstance(save, str):
            g['saver'].save(sess, save)

    return training_losses

def build_graph(
    num_classes,
    cell_type = None,
    num_weights_for_custom_cell = 5,
    state_size = 100,
    #num_classes = vocab_size,
    batch_size = 32,
    num_steps = 200,
    num_layers = 3,
    build_with_dropout=False,
    learning_rate = 1e-4):

    reset_graph()

    print("batch size: {}".format(batch_size))
    print("num_steps: {}".format(num_steps))
    x = tf.placeholder(tf.int32, shape = (batch_size, num_steps), name='input_placeholder')
    print("x shape: {}".format(x.get_shape()))
    y = tf.placeholder(tf.int32, shape = (batch_size, num_steps), name='labels_placeholder')
    print("y shape: {}".format(y.get_shape()))

    dropout = tf.constant(1.0)

    embeddings = tf.get_variable('embedding_matrix', [num_classes, state_size])

    rnn_inputs = tf.nn.embedding_lookup(embeddings, x)

    ## Get classes from:
    ## http://r2rt.com/recurrent-neural-networks-in-tensorflow-ii.html
    ## uses old API!!
    if cell_type == 'LSTM':
        cell = tf.contrib.rnn.LSTMCell(state_size, state_is_tuple=True)
    elif cell_type == 'GRU':
        cell = tf.nn.rnn_cell.GRUCell(state_size)
    ## elif cell_type == 'Custum':
    ##    cell = CustomCell(state_size, num_weights_for_custom_cell)
    ## elif cell_type == 'LN_LSTM':
    ##    cell = LayerNormalizedLSTMCell(state_size)
    else:
        cell = tf.nn.rnn_cell.BasicRNNCell(state_size)

    if build_with_dropout:
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=dropout)

    ## what si this?
    if cell_type == 'LSTM' or cell_type == 'LN_LSTM':
        cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
    else:
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)

    if build_with_dropout:
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=dropout)

    init_state = cell.zero_state(batch_size, tf.float32)
    rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=init_state)

    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [state_size, num_classes])
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))

    #reshape rnn_outputs and y
    rnn_outputs = tf.reshape(rnn_outputs, [-1, state_size])
    y_reshaped = tf.reshape(y, [-1])

    logits = tf.matmul(rnn_outputs, W) + b

    predictions = tf.nn.softmax(logits)

    total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

    return dict(
        x = x,
        y = y,
        init_state = init_state,
        final_state = final_state,
        total_loss = total_loss,
        train_step = train_step,
        preds = predictions,
        saver = tf.train.Saver()
    )

if __name__ == '__main__':
    """
    Load and process data, utility functions
    """
    file_url = 'https://raw.githubusercontent.com/jcjohnson/torch-rnn/master/data/tiny-shakespeare.txt'
    file_name = 'tinyshakespeare.txt'
    if not os.path.exists(file_name):
        urllib.urlretrieve(file_url, file_name)

    with open(file_name,'r') as f:
        raw_data = f.read()
        print("Data length:", len(raw_data))

    vocab = set(raw_data)
    vocab_size = len(vocab)
    idx_to_vocab = dict(enumerate(vocab))
    vocab_to_idx = dict(zip(idx_to_vocab.values(), idx_to_vocab.keys()))

    data = [vocab_to_idx[c] for c in raw_data]
    del raw_data

    #############
    print("Building graph")
    g = build_graph(num_classes = vocab_size, cell_type='LSTM', num_steps=80)
    t = time.time()
    print("Training network")
    losses = train_network(g, save="LSTM_3_epochs", num_epochs=3, num_steps=80)
    print("It took", time.time() - t, "seconds to train for 3 epochs.")
    print("Last epoch loss: {}".format(losses[-1]))
    print("Generating new sentences...")
    g = build_graph(num_classes = vocab_size, cell_type='LSTM', num_steps=1, batch_size=1)
    generate_characters(g, "./LSTM_3_epochs", 750, prompt='A', pick_top_chars=5)
