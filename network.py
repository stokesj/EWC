import tensorflow as tf


class Network(object):
    """Creates the computation graph for a fully connected rectifier/dropout network
     prediction model and Fisher diagonal."""

    def __init__(self, num_features, num_class, fc_hidden_units, apply_dropout, ewc_batch_size=100, ewc_batches=550):

        # tf.reset_default_graph()

        self.num_features = num_features
        self.num_class = num_class
        self.fc_units = fc_hidden_units
        self.sizes = [self.num_features] + self.fc_units + [self.num_class]
        self.apply_dropout = apply_dropout
        self.ewc_batch_size = ewc_batch_size
        self.ewc_batches = ewc_batches

        self.x = None
        self.y = None
        self.x_fisher = None
        self.y_fisher = None
        self.keep_prob_input = None
        self.keep_prob_hidden = None

        self.biases = None
        self.weights = None
        self.theta = None
        self.biases_lagged = None
        self.weights_lagged = None
        self.theta_lagged = None

        self.scores = None
        self.fisher_diagonal = None
        self.fisher_minibatch = None

        self.fisher_accumulate_op = None
        self.fisher_full_batch_average_op = None
        self.fisher_zero_op = None
        self.update_theta_op = None

        self.create_graph()

        self.saver = tf.train.Saver(max_to_keep=1000, var_list=self.theta + self.theta_lagged + self.fisher_diagonal)

    def create_graph(self):
        self.create_placeholders()
        self.create_fc_variables()
        self.scores = self.fc_feedforward(self.x, self.biases, self.weights, self.apply_dropout)
        self.create_fisher_diagonal()

    def fc_feedforward(self, h, biases, weights, apply_dropout):
        if apply_dropout:
            h = tf.nn.dropout(h, self.keep_prob_input)
        for (w, b) in zip(weights, biases)[:-1]:
            h = self.create_fc_layer(h, w, b)
            if apply_dropout:
                h = tf.nn.dropout(h, self.keep_prob_hidden)
        return self.create_fc_layer(h, weights[-1], biases[-1], apply_relu=False)

    def create_fisher_diagonal(self):
        nll, biases_per_example, weights_per_example = self.unaggregated_nll()
        self.fisher_minibatch = self.fisher_minibatch_sum(nll, biases_per_example, weights_per_example)
        self.create_fisher_ops()

    def unaggregated_nll(self):
        x_examples = tf.unstack(self.x_fisher)
        y_examples = tf.unstack(self.y_fisher)
        biases_per_example = [self.clone_variable_list(self.biases) for _ in range(0, self.ewc_batch_size)]
        weights_per_example = [self.clone_variable_list(self.weights) for _ in range(0, self.ewc_batch_size)]
        nll_list = []
        for (x, y, biases, weights) in zip(x_examples, y_examples, biases_per_example, weights_per_example):
            scores = self.fc_feedforward(tf.reshape(x, [1, self.num_features]), biases, weights, apply_dropout=False)
            nll = - tf.reduce_sum(y * tf.nn.log_softmax(scores))
            nll_list.append(nll)
        nlls = tf.stack(nll_list)
        return tf.reduce_sum(nlls), biases_per_example, weights_per_example

    def fisher_minibatch_sum(self, nll_per_example, biases_per_example, weights_per_example):
        bias_grads_per_example = [tf.gradients(nll_per_example, biases) for biases in biases_per_example]
        weight_grads_per_example = [tf.gradients(nll_per_example, weights) for weights in weights_per_example]
        return self.sum_of_squared_gradients(bias_grads_per_example, weight_grads_per_example)

    def sum_of_squared_gradients(self, bias_grads_per_example, weight_grads_per_example):
        bias_grads2_sum = []
        weight_grads2_sum = []
        for layer in range(0, len(self.fc_units) + 1):
            bias_grad2_sum = tf.add_n([tf.square(example[layer]) for example in bias_grads_per_example])
            weight_grad2_sum = tf.add_n([tf.square(example[layer]) for example in weight_grads_per_example])
            bias_grads2_sum.append(bias_grad2_sum)
            weight_grads2_sum.append(weight_grad2_sum)
        return bias_grads2_sum + weight_grads2_sum

    def create_fisher_ops(self):
        self.fisher_diagonal = self.bias_shaped_variables(name='bias_grads2', c=0.0, trainable=False) +\
                               self.weight_shaped_variables(name='weight_grads2', c=0.0, trainable=False)

        self.fisher_accumulate_op = [tf.assign_add(f1, f2) for f1, f2 in zip(self.fisher_diagonal, self.fisher_minibatch)]
        scale = 1 / float(self.ewc_batches * self.ewc_batch_size)
        self.fisher_full_batch_average_op = [tf.assign(var, scale * var) for var in self.fisher_diagonal]
        self.fisher_zero_op = [tf.assign(tensor, tf.zeros_like(tensor)) for tensor in self.fisher_diagonal]

    @staticmethod
    def create_fc_layer(input, w, b, apply_relu=True):
        with tf.name_scope('fc_layer'):
            output = tf.matmul(input, w) + b
            if apply_relu:
                output = tf.nn.relu(output)
        return output

    @staticmethod
    def create_variable(shape, name, c=None, sigma=None, trainable=True):
        if sigma:
            initial = tf.truncated_normal(shape, stddev=sigma, name=name)
        else:
            initial = tf.constant(c if c else 0.0, shape=shape, name=name)
        return tf.Variable(initial, trainable=trainable)

    @staticmethod
    def clone_variable_list(variable_list):
        return [tf.identity(var) for var in variable_list]

    def bias_shaped_variables(self, name, c=None, sigma=None, trainable=True):
        return [self.create_variable(shape=[i], name=name + '{}'.format(layer + 1),
                                     c=c, sigma=sigma, trainable=trainable) for layer, i in enumerate(self.sizes[1:])]

    def weight_shaped_variables(self, name, c=None, sigma=None, trainable=True):
        return [self.create_variable([i, j], name=name + '{}'.format(layer + 1),
                                     c=c, sigma=sigma, trainable=trainable)
                for layer, (i, j) in enumerate(zip(self.sizes[:-1], self.sizes[1:]))]

    def create_fc_variables(self):
        with tf.name_scope('fc_variables'):
            self.biases = self.bias_shaped_variables(name='biases_fc', c=0.1, trainable=True)
            self.weights = self.weight_shaped_variables(name='weights_fc', sigma=0.1, trainable=True)
            self.theta = self.biases + self.weights
        with tf.name_scope('fc_variables_lagged'):
            self.biases_lagged = self.bias_shaped_variables(name='biases_fc_lagged', c=0.0, trainable=False)
            self.weights_lagged = self.weight_shaped_variables(name='weights_fc_lagged', c=0.0, trainable=False)
            self.theta_lagged = self.biases_lagged + self.weights_lagged
        self.update_theta_op = [v1.assign(v2) for v1, v2 in zip(self.theta_lagged, self.theta)]

    def create_placeholders(self):
        with tf.name_scope("prediction-inputs"):
            self.x = tf.placeholder(tf.float32, [None, self.num_features], name='x-input')
            self.y = tf.placeholder(tf.float32, [None, self.num_class], name='y-input')
        with tf.name_scope("dropout-probabilities"):
            self.keep_prob_input = tf.placeholder(tf.float32)
            self.keep_prob_hidden = tf.placeholder(tf.float32)
        with tf.name_scope("fisher-inputs"):
            self.x_fisher = tf.placeholder(tf.float32, [self.ewc_batch_size, self.num_features])
            self.y_fisher = tf.placeholder(tf.float32, [self.ewc_batch_size, self.num_class])