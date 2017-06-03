import numpy as np
import os
import tensorflow as tf

from network import Network


class Classifier(Network):
    """Supplies fully connected prediction model with training loop which absorbs minibatches and updates weights."""

    def __init__(self, checkpoint_path='logs/checkpoints/', summaries_path='logs/summaries/', *args, **kwargs):
        super(Classifier, self).__init__(*args, **kwargs)
        self.checkpoint_path = checkpoint_path
        self. summaries_path = summaries_path
        self.writer = None
        self.merged = None
        self.optimizer = None
        self.train_step = None
        self.accuracy = None
        self.loss = None

        self.create_loss_and_accuracy()

    def train(self, sess, model_name, model_init_name, dataset, num_updates, mini_batch_size, fisher_multiplier,
              learning_rate, log_frequency=None, dataset_lagged=None):  # pass previous dataset as convenience
        print 'training ' + model_name + ' with weights initialized at ' + str(model_init_name)
        self.prepare_for_training(sess, model_name, model_init_name, fisher_multiplier, learning_rate)
        for i in range(num_updates):
            self.minibatch_sgd(sess, i, dataset, mini_batch_size, log_frequency)
        self.update_fisher_full_batch(sess, dataset)
        self.save_weights(i, sess, model_name)
        print 'finished training ' + model_name

    def test(self, sess, model_name, batch_xs, batch_ys):
        self.restore_model(sess, model_name)
        feed_dict = self.create_feed_dict(batch_xs, batch_ys, keep_input=1.0, keep_hidden=1.0)
        accuracy = sess.run(self.accuracy, feed_dict=feed_dict)
        return accuracy

    def minibatch_sgd(self, sess, i, dataset, mini_batch_size, log_frequency):
        batch_xs, batch_ys = dataset.next_batch(mini_batch_size)
        feed_dict = self.create_feed_dict(batch_xs, batch_ys)
        sess.run(self.train_step, feed_dict=feed_dict)
        if log_frequency and i % log_frequency is 0:
            self.evaluate(sess, i, feed_dict)

    def evaluate(self, sess, iteration, feed_dict):
        if self.apply_dropout:
            feed_dict.update({self.keep_prob_input: 1.0, self.keep_prob_hidden: 1.0})
        summary, accuracy = sess.run([self.merged, self.accuracy], feed_dict=feed_dict)
        self.writer.add_summary(summary, iteration)
        # print 'accuracy on iteration', iteration, 'is', accuracy

    def update_fisher_full_batch(self, sess, dataset):
        dataset._index_in_epoch = 0  # ensures that all training examples are included without repetitions
        sess.run(self.fisher_zero_op)
        for _ in range(0, self.ewc_batches):
            self.accumulate_fisher(sess, dataset)
        sess.run(self.fisher_full_batch_average_op)
        sess.run(self.update_theta_op)

    def accumulate_fisher(self, sess, dataset):
        batch_xs, batch_ys = dataset.next_batch(self.ewc_batch_size)
        sess.run(self.fisher_accumulate_op, feed_dict={self.x_fisher: batch_xs, self.y_fisher: batch_ys})

    def prepare_for_training(self, sess, model_name, model_init_name, fisher_multiplier, learning_rate):
        self.writer = tf.summary.FileWriter(self.summaries_path + model_name, sess.graph)
        self.merged = tf.summary.merge_all()
        self.train_step = self.create_train_step(fisher_multiplier if model_init_name else 0.0, learning_rate)
        init = tf.global_variables_initializer()
        sess.run(init)
        if model_init_name:
            self.restore_model(sess, model_init_name)

    def create_loss_and_accuracy(self):
        with tf.name_scope("loss"):
            average_nll = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.y))  # optimized
            tf.summary.scalar("loss", average_nll)
            self.loss = average_nll
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.scores, 1), tf.argmax(self.y, 1)), tf.float32))
            tf.summary.scalar('accuracy', accuracy)
            self.accuracy = accuracy

    def create_train_step(self, fisher_multiplier, learning_rate):
        with tf.name_scope("optimizer"):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            penalty = tf.add_n([tf.reduce_sum(tf.square(w1-w2)*f) for w1, w2, f
                                in zip(self.theta, self.theta_lagged, self.fisher_diagonal)])
            return self.optimizer.minimize(self.loss + (fisher_multiplier / 2) * penalty, var_list=self.theta)

    def save_weights(self, time_step, sess, model_name):
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        self.saver.save(sess=sess, save_path=self.checkpoint_path + model_name + '.ckpt', global_step=time_step,
                        latest_filename=model_name)
        print 'saving model ' + model_name + ' at time step ' + str(time_step)

    def restore_model(self, sess, model_name):
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir=self.checkpoint_path, latest_filename=model_name)
        self.saver.restore(sess=sess, save_path=ckpt.model_checkpoint_path)

    def create_feed_dict(self, batch_xs, batch_ys, keep_hidden=0.5, keep_input=0.8):
        feed_dict = {self.x: batch_xs, self.y: batch_ys}
        if self.apply_dropout:
            feed_dict.update({self.keep_prob_hidden: keep_hidden, self.keep_prob_input: keep_input})
        return feed_dict
