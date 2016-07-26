import logging

import numpy as np
import tensorflow as tf

from s2ao.config import config


class s2ao():
    def __init__(self, hidden_num_1, hidden_num_2, lstm_step, class1_size, class2_size):
        self.class1_size = class1_size
        self.class2_size = class2_size
        self.t_v, self.f_v, self.t_n, self.f_n = 0, 0, 0, 0
        self.keep_prob = tf.placeholder(dtype=tf.float32)
        with tf.variable_scope("phase1"), tf.name_scope("phase1"):
            with tf.name_scope("fc1"), tf.variable_scope("fc1"):
                with tf.device("/cpu:0"):
                    self.input = tf.placeholder(shape=[None, lstm_step, 4096 * 2], dtype=tf.float32, name="input")
                W = tf.get_variable("W", shape=[1, 1, 4096 * 2, 1000], dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.get_variable("b", shape=[1000], initializer=tf.nn.init_ops.zeros_initializer)
                input_embedded = tf.nn.relu(tf.squeeze(
                    tf.nn.dropout(
                        tf.nn.relu(tf.nn.bias_add(
                            tf.nn.conv2d(tf.expand_dims(self.input, dim=1), W, strides=[1, 1, 1, 1], padding="VALID"),
                            b), name="relu1"),
                        keep_prob=self.keep_prob, name="dropout1"),
                    squeeze_dims=[1]))

            lengths = self.length(self.input)
            with tf.variable_scope('lstm1'), tf.name_scope("lstm1"):
                layer1_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_num_1, state_is_tuple=True)
                layer1_output, _ = tf.nn.dynamic_rnn(
                    layer1_lstm_cell,
                    input_embedded,
                    lengths,
                    dtype=tf.float32
                )

            # padding output TODO

            with tf.variable_scope('lstm2'), tf.name_scope("lstm2"):
                layer2_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_num_2, state_is_tuple=True)
                layer2_output, state = tf.nn.dynamic_rnn(
                    layer2_lstm_cell,
                    layer1_output,
                    lengths,
                    dtype=tf.float32
                )
            # last1 = SeqToTwoClass.last_relevant(layer2_output, self.length(self.input))
            # self.last = tf.concat(1, (last1, state[0]))
            last_h = s2ao.last_relevant(layer2_output, lengths)
            # with tf.variable_scope("fc2"), tf.
            with tf.variable_scope('fc2'), tf.name_scope("fc2"):
                W = tf.get_variable("W", shape=[hidden_num_2, hidden_num_2],
                                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.get_variable("b", shape=[hidden_num_2], initializer=tf.nn.init_ops.zeros_initializer)
                result = tf.nn.dropout(tf.nn.relu(tf.nn.xw_plus_b(last_h, W, b, name="result"), name="relu2"),
                                       keep_prob=self.keep_prob)

            with tf.variable_scope('fc3-softmax'), tf.name_scope('fc3-softmax'):
                W = tf.get_variable("W", shape=[hidden_num_2, class1_size],
                                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.get_variable("b", shape=[class1_size], initializer=tf.nn.init_ops.zeros_initializer)
                result_v = tf.nn.xw_plus_b(result, W, b, name="result")

            with tf.variable_scope('cost_verb'), tf.name_scope('cost_verb'):
                self.verb_label = tf.placeholder(shape=[None, class1_size], dtype=tf.float32, name="verb_label")
                self.cost_v = tf.nn.softmax_cross_entropy_with_logits(result_v, self.verb_label, name="verb_cost")

        with tf.variable_scope("phase2"), tf.name_scope("phase2"):
            with tf.variable_scope("fc4"), tf.name_scope("fc4"):
                inp = tf.concat(1, [last_h, result])
                W = tf.get_variable("W", shape=[hidden_num_2 * 2, hidden_num_2],
                                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.get_variable("b", shape=[hidden_num_2], initializer=tf.nn.init_ops.zeros_initializer)
                result = tf.nn.dropout(tf.nn.relu(tf.nn.xw_plus_b(inp, W, b, name="result"), name="relu4"), keep_prob=self.keep_prob)

            with tf.variable_scope("fc5-softmax"), tf.name_scope("fc5-softmax"):
                W = tf.get_variable("W", shape=[hidden_num_2, class2_size],
                                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.get_variable("b", shape=[class2_size], initializer=tf.nn.init_ops.zeros_initializer)
                result_n = tf.nn.xw_plus_b(result, W, b, name="result")

            with tf.variable_scope('cost_noun'), tf.name_scope('cost_noun'):
                self.noun_label = tf.placeholder(shape=[None, class2_size], dtype=tf.float32, name="noun_label")
                self.cost_n = tf.nn.softmax_cross_entropy_with_logits(result_n, self.noun_label, name="noun_cost")

        phase1_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "phase1")
        phase2_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "phase2")
        self.train_verb = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(self.cost_v,
                                                                                          var_list=phase1_vars)
        self.train_noun = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.cost_n, var_list=phase2_vars)
        self.init_op = tf.initialize_all_variables()
        self.predict_v = tf.argmax(tf.nn.softmax(result_v), dimension=1)
        self.predict_n = tf.argmax(tf.nn.softmax(result_n), dimension=1)
        self.sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=8))
        self.sess.run(self.init_op)
        tf.train.write_graph(self.sess.graph_def, '/home/dianli/', 'graph.pbtxt')
        writer = tf.train.SummaryWriter('/home/dianli')
        writer.add_graph(self.sess.graph)

    def train(self, batch_data, batch_verb, batch_noun):
        """
        :param batch_data: [batch_size * (4096 * 2)]
        :param batch_verb: [batch_size * 3]
        :param batch_noun: [batch_size * NUM_NOUN]
        :return:
        """
        feed_dict = {self.input: batch_data, self.verb_label: batch_verb, self.noun_label: batch_noun,
                     self.keep_prob: float(config["training"]["keep_prob"])}
        v_pred, _ = self.sess.run((self.predict_v, self.train_verb), feed_dict=feed_dict)
        n_pred, _ = self.sess.run((self.predict_n, self.train_noun), feed_dict=feed_dict)
        t_v = np.sum(v_pred == np.argmax(batch_verb, axis=1))
        f_v = len(v_pred) - t_v
        t_n = np.sum(n_pred == np.argmax(batch_noun, axis=1))
        f_n = len(n_pred) - t_n
        self.t_v += t_v
        self.f_v += f_v
        self.t_n += t_n
        self.f_n += f_n

    def stats(self, batch_data, batch_verb, batch_noun):
        v_pred, n_pred, v_cost, n_cost = self.sess.run((self.predict_v, self.predict_n, self.cost_v, self.cost_n),
                                                       feed_dict={self.input: batch_data, self.verb_label: batch_verb,
                                                                  self.noun_label: batch_noun, self.keep_prob: 1})
        return np.sum(v_cost), np.sum(v_pred == np.argmax(batch_verb, axis=1)) / len(v_pred), \
               np.sum(n_cost), np.sum(n_pred == np.argmax(batch_noun, axis=1)) / len(n_pred)

    @staticmethod
    def last_relevant(output, length):
        batch_size = tf.shape(output)[0]
        max_length = int(output.get_shape()[1])
        output_size = int(output.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(output, [-1, output_size])
        relevant = tf.gather(flat, index)
        return relevant

    def length(self, data):
        used = tf.sign(tf.reduce_max(tf.abs(data), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length
