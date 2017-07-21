import numpy as np

from copy import deepcopy
from classifiers import Classifier
from numpy.random import RandomState
from queue import PriorityQueue
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets


PRNG = RandomState(12345)
MINI_BATCH_SIZE = 250
LOG_FREQUENCY = 1000


class HyperparameterTuner(object):
    def __init__(self, sess, hidden_layers, hidden_units, num_perms, trials, epochs):
        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units
        self.num_perms = num_perms
        self.epochs = epochs
        self.task_list = self.create_permuted_mnist_task(num_perms)
        self.trial_learning_rates = [PRNG.uniform(1e-4, 1e-3) for _ in range(0, trials)]
        self.best_parameters = []
        self.sess = sess
        self.classifier = Classifier(num_class=10,
                                     num_features=784,
                                     fc_hidden_units=[hidden_units for _ in range(hidden_layers)],
                                     apply_dropout=True)

    def search(self):
        for t in range(0, self.num_perms):
            queue = PriorityQueue()
            for learning_rate in self.trial_learning_rates:
                self.train_on_task(t, learning_rate, queue)
            self.best_parameters.append(queue.get())
            self.evaluate()

    def evaluate(self):
        accuracies = []
        for parameters in self.best_parameters:
            accuracy = self.classifier.test(sess=self.sess,
                                            model_name=parameters[1],
                                            batch_xs=self.task_list[0].test.images,
                                            batch_ys=self.task_list[0].test.labels)
            accuracies.append(accuracy)
        print(accuracies)

    def train_on_task(self, t, lr, queue):
        model_name = self.file_name(lr, t)
        dataset_train = self.task_list[t].train
        dataset_lagged = self.task_list[t - 1] if t > 0 else None
        model_init_name = self.best_parameters[t - 1][1] if t > 0 else None
        self.classifier.train(sess=self.sess,
                              model_name=model_name,
                              model_init_name=model_init_name,
                              dataset=dataset_train,
                              dataset_lagged=dataset_lagged,
                              num_updates=(55000//MINI_BATCH_SIZE)*self.epochs,
                              mini_batch_size=MINI_BATCH_SIZE,
                              log_frequency=LOG_FREQUENCY,
                              fisher_multiplier=1.0/lr,
                              learning_rate=lr)
        accuracy = self.classifier.test(sess=self.sess,
                                        model_name=model_name,
                                        batch_xs=self.task_list[0].validation.images,
                                        batch_ys=self.task_list[0].validation.labels)
        queue.put((-accuracy, model_name))

    def create_permuted_mnist_task(self, num_datasets):
        mnist = read_data_sets("MNIST_data/", one_hot=True)
        task_list = [mnist]
        for seed in range(1, num_datasets):
            task_list.append(self.permute(mnist, seed))
        return task_list

    @staticmethod
    def permute(task, seed):
        np.random.seed(seed)
        perm = np.random.permutation(task.train._images.shape[1])
        permuted = deepcopy(task)
        permuted.train._images = permuted.train._images[:, perm]
        permuted.test._images = permuted.test._images[:, perm]
        permuted.validation._images = permuted.validation._images[:, perm]
        return permuted

    def file_name(self, lr, t):
        return 'layers=%d,hidden=%d,lr=%.5f,multiplier=%.2f,mbsize=%d,epochs=%d,perm=%d' \
               % (self.hidden_layers, self.hidden_units, lr, 1 / lr, MINI_BATCH_SIZE, self.epochs, t)

