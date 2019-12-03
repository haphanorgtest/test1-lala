import tensorflow as tf
import numpy as np
import os, io
from copy import deepcopy
from skimage.morphology import disk
from skimage.measure import label
from scipy.ndimage import binary_dilation
from scipy.ndimage import center_of_mass
import time
import matplotlib.pyplot as plt

class ConvLSTM(object):
    def __init__(self, session,
                 optimizer,
                 saver,
                 checkpoint_dir,
                 max_gradient=5,
                 summary_writer=None,
                 summary_every=100,
                 save_every=2000,
                 training=True,
                 size_x=64,
                 size_y=64,
                 seq_len=10,
                 batch_size=32,
                 training_data=None,
                 training_labels=None):
        self.session = session
        self.optimizer = optimizer
        self.saver = saver
        self.max_gradient = max_gradient
        self.summary_writer = summary_writer
        self.summary_every = summary_every
        self.save_every = save_every
        self.checkpoint_dir = checkpoint_dir
        self.training = training

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.size_x = size_x
        self.size_y = size_y
        self.training_data = np.expand_dims(training_data, -1)  # np.vstack((training_data, training_data[-1:, :, :]))
        self.training_labels = training_labels
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.epoch_size = int(self.training_data.shape[0] / self.batch_size)

        self.create_variables()
        self.summary_writer.add_graph(self.session.graph)

        self.compress_jump = 1
        self.z_tolerance = 1
        self.xy_tolerance = 10

    def create_variables(self):
        self.input = tf.placeholder(tf.float32, [self.batch_size, self.seq_len, self.size_x, self.size_y, 1], name='input')
        self.labels = tf.placeholder(tf.float32, [self.batch_size, self.seq_len, self.size_x, self.size_y], name='labels')
        self.logits = self.model()
        self.output = tf.sigmoid(self.logits)

        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        self.loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.logits)) / (self.size_x * self.size_y)
        self.train_op = self.optimizer.minimize(self.loss, var_list=params, global_step=self.global_step)

        # filtered_output = tf.clip_by_value(self.output, 0.6, 1.0)
        # filtered_output -= 0.6
        # filtered_output = tf.reduce_sum(filtered_output, [1, 2, 3])

        """
        plot_buf = self.plot(self.input, self.output)
        image = tf.image.decode_png(plot_buf.getvalue(), channels=1)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        # Add image summary
        """
        images = self.plot(self.input, self.output, self.labels)
        image_sm = tf.summary.image("plot", images, self.seq_len * 3)

        cost_sm = tf.summary.scalar("cost", self.loss)
        self.merge_list = [image_sm, cost_sm]
        self.summarize = tf.summary.merge(self.merge_list)

    def model(self):
        conv_lstm_fw = tf.contrib.rnn.ConvLSTMCell(conv_ndims=2,
                                                   input_shape=[self.size_x, self.size_y, 1],
                                                   output_channels=32,
                                                   kernel_shape=[5, 5],
                                                   use_bias=True)
        conv_lstm_bw = tf.contrib.rnn.ConvLSTMCell(conv_ndims=2,
                                                   input_shape=[self.size_x, self.size_y, 1],
                                                   output_channels=32,
                                                   kernel_shape=[5, 5],
                                                   use_bias=True)
        outputs, states = tf.nn.bidirectional_dynamic_rnn(conv_lstm_fw, conv_lstm_bw,
                                                          inputs=self.input,
                                                          dtype=tf.float32)
        conv_lstm_2 = tf.contrib.rnn.ConvLSTMCell(conv_ndims=2,
                                                  input_shape=[self.size_x, self.size_y, 1],
                                                  output_channels=32,
                                                  kernel_shape=[5, 5],
                                                  use_bias=True)
        self.convlstm_output, state = tf.nn.dynamic_rnn(conv_lstm_2,
                                          inputs=tf.concat(outputs, -1),
                                          dtype=tf.float32)
        self.convlstm_output = tf.squeeze(self.convlstm_output)
        self.convlstm_output = tf.reshape(self.convlstm_output, [self.batch_size * self.seq_len, self.size_x, self.size_y, 32])
        # self.convlstm_output = tf.expand_dims(self.convlstm_output, axis=-1)
        conv1 = tf.layers.conv2d(self.convlstm_output, 16, [5, 5], padding='same')
        conv2 = tf.layers.conv2d(self.convlstm_output, 16, [3, 3], padding='same')
        conv2 = tf.layers.conv2d(self.convlstm_output, 1, [1, 1], padding='same')
        conv2 = tf.squeeze(conv2)
        output = tf.reshape(conv2, [self.batch_size, self.seq_len, self.size_x, self.size_y])
        return output

    def plot(self, input, output, labels):
        # input = tf.squeeze(input)
        output = tf.expand_dims(output, axis=-1)
        labels = tf.expand_dims(labels, axis=-1)
        images = tf.concat([input[-1, :, :, :, :], output[-1, :, :, :, :], labels[-1, :, :, :, :]], 0)
        """
        fig = plt.figure()
        fig.patch.set_facecolor('white')
        for j in range(0, self.seq_len):
            im = input[-1, j, :, :]
            fig.add_subplot(4,5,j+1)
            plt.imshow(im, cmap='Greys_r')
        for j in range(0, self.seq_len):
            im = output[-1, j, :, :]
            fig.add_subplot(4,5,j + self.seq_len + 1)
            plt.imshow(im, cmap='Greys_r')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        return buf
        """
        return images


    def get_sample(self):
        # self.epoch_size = int(self.training_data.shape[0] / self.batch_size)
        n = self.gs % self.epoch_size

        seq = self.training_data[self.batch_size * n:self.batch_size * (n+1), :, :, :]
        labels = self.training_labels[self.batch_size * n:self.batch_size * (n+1), :, :, :]
        # print seq.shape, labels.shape, 'seq, labels ============='
        return seq, labels

    def run_train(self):
        with self.session.as_default(), self.session.graph.as_default():
            print 'started ---', self.epoch_size, 'epoch_size'
            self.gs = self.session.run(self.global_step)
            try:
                while self.gs < 20000:
                    self.train_step()

                tf.logging.info("Reached global step {}. Stopping.".format(self.gs))
                self.saver.save(self.session, os.path.join(self.checkpoint_dir, 'my_model'), global_step=self.gs)
            except KeyboardInterrupt:
                print 'a du ----'
                self.saver.save(self.session, os.path.join(self.checkpoint_dir, 'my_model'), global_step=self.gs)
            return

    def train_step(self):
        start_time = time.time()
        # Get sample
        input, labels = self.get_sample()

        feed_dict = {
            self.input: input,
            self.labels: labels,
        }
        output, loss, summary, _, self.gs = self.session.run([self.output, self.loss, self.summarize, self.train_op, self.global_step], feed_dict)
        duration = time.time() - start_time

        # emit summaries
        if self.gs % 10 == 9:
            print loss, duration, self.gs, 'loss, duration, gs'

        if self.gs % 30 == 29:
            print 'summary ---'
            self.summary_writer.add_summary(summary, self.gs)

        if self.gs % 2000 == 100:
            print("Saving model checkpoint: {}".format(str(self.gs)))
            self.saver.save(self.session, os.path.join(self.checkpoint_dir, 'my_model'), global_step=self.gs)

    def get_test_sample(self, n=0):
        epoch_size = int(self.test_data.shape[0] / self.batch_size)
        if n < epoch_size:
            seq = self.test_data[self.batch_size * n:self.batch_size * (n + 1), :, :, :]
            labels = self.test_labels[self.batch_size * n:self.batch_size * (n + 1), :, :, :]
        else:
            seq = None; labels = None
        return seq, labels

    def get_center(self, tensor, threshold=0.6):
        tensor[tensor < threshold] = 0
        tensor[tensor >= threshold] = 1
        if np.sum(tensor) > 0:
            center = center_of_mass(tensor)
        else:
            center = None
        return center

    def test(self, test_data, test_labels):
        self.test_data = np.expand_dims(test_data, -1)
        self.test_labels = test_labels
        n = 0; seq = True; labels = True
        TP, FP, TN, FN = 0.0, 0.0, 0.0, 0.0
        while seq is not None:
            print n, 'n'
            start_time = time.time()
            seq, labels = self.get_test_sample(n)
            n += 1
            feed_dict = {
                self.input: seq,
                self.labels: labels,
            }
            output = self.session.run([self.output], feed_dict)
            duration = time.time() - start_time
            for i in range(self.batch_size):
                center_pred = self.get_center(output[i, :, :, :], threshold=0.6)
                center_label = self.get_center(labels[i, :, :, :])
                if center_label is None:
                    if center_pred is None:
                        TN += 1.0
                    else:
                        FP += 1.0
                else:
                    if center_pred is None:
                        FN += 1.0
                    else:
                        xy_dist = np.sqrt((center_pred[1] - center_label[1])**2 + (center_pred[2] - center_label[2])**2)
                        z_dist = np.abs(center_pred[0] - center_label[0])
                        if xy_dist <= self.xy_tolerance and z_dist <= self.z_tolerance:
                            TP += 1.0
                        else:
                            FP += 1.0
                print TP, FP, TN, FN, duration, 'TP, FP, TN, FN, duration'

        precision = (TP / (TP + FP)) * 100
        recall = (TP / (TP + FN)) * 100
        F1_score = 2 * (precision * recall / (precision + recall))
        print precision, recall, F1_score, 'precision, recall, F1_score'




