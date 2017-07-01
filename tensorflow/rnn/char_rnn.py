#!/usr/bin/env python

import os, sys
import argparse, sys, os
import inspect, time, math
import collections
import tensorflow as tf
import numpy as np
import urllib.request
from six.moves import xrange
from tensorflow.python.framework import dtypes

FLAGS = None
class Reader(object):
    def __init__(self, batch_size, num_steps):
        """ 
        Args:
            batch_size: int, the batch size.
            num_steps: int, the number of unrolls.
        """
        self._num_steps = num_steps
        self._batch_size = batch_size

    def ptb_iterator(self, raw_data, steps_ahead=1):
        """Iterate on the raw PTB data.
        This generates batch_size pointers into the raw PTB data, and allows
        minibatch iteration along these pointers.
        Args:
            raw_data: one of the raw data outputs from ptb_raw_data.
        Yields:
            Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
            The second element of the tuple is the same data time-shifted to the
            right by one.
        Raises:
            ValueError: if batch_size or num_steps are too high.
        """
        batch_size = self.batch_size
        num_steps = self.num_steps
        
        raw_data = np.array(raw_data, dtype=np.int32)

        data_len = len(raw_data)

        batch_len = data_len // batch_size

        data = np.zeros([batch_size, batch_len], dtype=np.int32)
        offset = 0
        if data_len % batch_size:
            offset = np.random.randint(0, data_len % batch_size)
        for i in range(batch_size):
            data[i] = raw_data[batch_len * i + offset:batch_len * (i + 1) + offset]

        epoch_size = (batch_len - steps_ahead) // num_steps

        if epoch_size == 0:
            raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

        for i in range(epoch_size):
            x = data[:, i*num_steps:(i+1)*num_steps]
            y = data[:, i*num_steps+1:(i+1)*num_steps+steps_ahead]
            yield (x, y)

        if epoch_size * num_steps < batch_len - steps_ahead:
            yield (data[:, epoch_size*num_steps : batch_len - steps_ahead], data[:, epoch_size*num_steps + 1:])

    def shuffled_ptb_iterator(raw_data, batch_size, num_steps):
        raw_data = np.array(raw_data, dtype=np.int32)
        r = len(raw_data) % num_steps
        if r:
            n = np.random.randint(0, r)
            raw_data = raw_data[n:n + len(raw_data) - r]

        raw_data = np.reshape(raw_data, [-1, num_steps])
        np.random.shuffle(raw_data)

        num_batches = int(np.ceil(len(raw_data) / batch_size))

        for i in range(num_batches):
            data = raw_data[i*batch_size:min(len(raw_data), (i+1)*batch_size),:]
        yield (data[:,:-1], data[:,1:])
    
    @property
    def num_steps(self):
        return self._num_steps
     
    @property
    def batch_size(self):
        return self._batch_size
     
def read_data(verbose=False):
    file_url = 'https://raw.githubusercontent.com/jcjohnson/torch-rnn/master/data/tiny-shakespeare.txt'
    file_name = 'tinyshakespeare.txt'
    if not os.path.exists(file_name):
        if verbose:
            print("downloading data")
        urllib.request.urlretrieve(file_url, file_name)
    if verbose:
        print("parsing data file")
    with open(file_name,'r') as f:
        raw_data = f.read()
        print("Data length:", len(raw_data))

    vocab = set(raw_data)
    vocab_size = len(vocab)
    idx_to_vocab = dict(enumerate(vocab))
    vocab_to_idx = dict(zip(idx_to_vocab.values(), idx_to_vocab.keys()))

    data = [vocab_to_idx[c] for c in raw_data]
    return data, idx_to_vocab, vocab_to_idx

def gen_epochs(reader, data, num_epochs):
    """ create an iterator on the data 
    returns num_epochs of  X, y tensor of batch_size * num_steps size 
    """ 
    for i in range(num_epochs):
        yield reader.ptb_iterator(data)

def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()

def train_network(g, data, reader, num_epochs, verbose = True, save=False):
    tf.set_random_seed(2345)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        #sess.run(tf.initialize_all_variables())
        sess.run(init)
        training_losses = []
        for idx, epoch in enumerate(gen_epochs(reader, data, num_epochs)):
            training_loss = 0
            steps = 0
            training_state = None
            for X, Y in epoch:
                steps += 1
                #if X.shape != (50, 80):
                #    print(steps, X.shape)
                feed_dict={g['x']: X, g['y']: Y}
                if training_state is not None:
                    feed_dict[g['init_state']] = training_state
                training_loss_, training_state, _ = sess.run([g['total_loss'], g['final_state'], g['train_step']], feed_dict)
                training_loss += training_loss_
            if verbose:
                print("Average training loss for Epoch", idx, ":", training_loss/steps)
            training_losses.append(training_loss/steps)

        if isinstance(save, str):
            g['saver'].save(sess, save)

    return training_losses

def build_graph(num_classes, state_size = 100, batch_size = 32, num_steps = 200, num_layers = 3, learning_rate = 1e-4):

    reset_graph()

    x = tf.placeholder(tf.int32, [batch_size, None], name='input_placeholder')
    y = tf.placeholder(tf.int32, [batch_size, None], name='labels_placeholder')

    #embeddings = tf.get_variable('embedding_matrix', [num_classes, state_size])
    #rnn_inputs = tf.nn.embedding_lookup(embeddings, x)
    #rnn_inputs = [tf.squeeze(i) for i in tf.split(1, num_steps, tf.nn.embedding_lookup(embeddings, x))]
    #rnn_inputs = tf.unstack(inputs, num=num_steps, axis=1)

    embeddings = tf.get_variable('embedding_matrix', [num_classes, state_size])

    # Note that our inputs are no longer a list, but a tensor of dims batch_size x num_steps x state_size
    rnn_inputs = tf.nn.embedding_lookup(embeddings, x)
    
    dropout = tf.constant(1.0)
    
    def lstm_cell():
        # With the latest TensorFlow source code (as of Mar 27, 2017),
        # the BasicLSTMCell will need a reuse parameter which is unfortunately not
        # defined in TensorFlow 1.0. To maintain backwards compatibility, we add
        # an argument check here:
        if 'reuse' in inspect.getargspec(tf.contrib.rnn.LSTMCell.__init__).args:
            return tf.contrib.rnn.LSTMCell(state_size, forget_bias=0.0, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
        else:
            return tf.contrib.rnn.LSTMCell(state_size, forget_bias=0.0, state_is_tuple=True)
        
    #attn_cell = lstm_cell
    def attn_cell():
        return tf.contrib.rnn.DropoutWrapper(lstm_cell(), input_keep_prob=dropout, output_keep_prob=dropout)
    #if is_training and config.keep_prob < 1:
        #def attn_cell():
            #return tf.contrib.rnn.DropoutWrapper(lstm_cell(), output_keep_prob=config.keep_prob)
    
    cell = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(num_layers)], state_is_tuple=True)

    init_state = cell.zero_state(batch_size, tf.float32)
    rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=init_state)
    #rnn_outputs, final_state = tf.contrib.rnn.static_rnn(cell, rnn_inputs, initial_state=init_state)

    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [state_size, num_classes])
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
        
    #reshape rnn_outputs and y so we can get the logits in a single matmul
    rnn_outputs = tf.reshape(rnn_outputs, [-1, state_size])
    #rnn_outputs =  tf.reshape(tf.stack(axis=1, values=rnn_outputs), [-1, state_size])
    
    y_reshaped = tf.reshape(y, [-1])
    #logits = tf.matmul(output, softmax_w) + softmax_b
    logits = tf.matmul(rnn_outputs, W) + b

    predictions = tf.nn.softmax(logits)

    total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

    return dict(x = x,  y = y,  init_state = init_state, final_state = final_state, total_loss = total_loss, train_step = train_step, preds=predictions, saver=tf.train.Saver())


def generate_characters(g, vocab_to_idx, idx_to_vocab, checkpoint, num_chars, prompt='A', pick_top_chars=None):
    """ Accepts a current character, initial state"""
    init = tf.global_variables_initializer()
    vocab_size = len(vocab_to_idx)
    with tf.Session() as sess:
        sess.run(init)
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

            #print(preds)            
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

def main(_):
    num_steps = 80
    state_size = 512
    batch_size = 50
    reader = Reader(batch_size, num_steps)
    data, idx_to_vocab, vocab_to_idx = read_data(True)
    #for idx, epoch in enumerate(gen_epochs(reader, data, 10)):
        #cnt = 0
        #for X, Y in epoch:
            #cnt += 1
            #print(idx, X.shape)
        #print(cnt)
        
    #g = build_graph(len(vocab_to_idx), state_size = state_size, batch_size = batch_size, num_steps = num_steps, num_layers = 3, learning_rate = 5e-4)
    #losses = train_network(g,  data, reader, 3, save="saves/LSTM_20_epochs")
    #losses = train_network(g, data, reader, 30, save="saves/LSTM_30_epochs_shakespear")
    g = build_graph(len(vocab_to_idx), state_size=state_size, num_steps=1, batch_size=1)
    print()
    generate_characters(g, vocab_to_idx, idx_to_vocab, "saves/LSTM_30_epochs_shakespear", 750, prompt='O', pick_top_chars=5)
    
def cmd_parser():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, default="./", help="path to save the data")
    parser.add_argument("--m_train", type=int, default=1000, help="the number of train sequence to generate")
    parser.add_argument("--m_test", type=int, default=100, help="the number of test sequence to generate")
    parser.add_argument("--max_steps", type=int, default=10, help="the number of steps to train the model")
    parser.add_argument("--log_dir", type=str, default="./log", help="the log directory")
    parser.add_argument("--batch_size", type=int, default=20, help="the batch size to use")
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Initial learning rate.')
    return parser

if __name__ == "__main__":
    parser = cmd_parser()
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
    #tf.app.run()
