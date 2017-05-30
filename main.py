import argparse
from tuner import HyperparameterTuner
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--hidden_layers', help='the number of hidden layers', type=int, required=True)
parser.add_argument('--hidden_units', help='the number of units per hidden layer', type=int, required=True)
parser.add_argument('--num_perms', help='the number of tasks', type=int, required=True)
parser.add_argument('--trials', help='the number of hyperparameter trials per task', type=int, required=True)
parser.add_argument('--epochs', help='the number of training epochs per task', type=int, required=True)
args = parser.parse_args()

if __name__ == "__main__":
    with tf.Session() as sess:
        tuner = HyperparameterTuner(sess=sess, hidden_layers=args.hidden_layers, hidden_units=args.hidden_units,
                                    num_perms=args.num_perms, trials=args.trials, epochs=args.epochs)
        tuner.search()
        print tuner.best_parameters

