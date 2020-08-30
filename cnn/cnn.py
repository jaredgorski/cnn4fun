import numpy as np

from .util import log
from .layers.activation import ActivationLayer

"""
If program `gnuplot` is available, set global
"""
def has_gnuplot():
    from shutil import which
    return which('gnuplot') is not None

has_gnuplot = has_gnuplot()

class CNN:

    def __init__(self, layers):
        self.layers = layers.copy()
        layers.reverse()
        self.reverse_layers = layers

    def __set_depth_if_grayscale(self, input):
        input_shape = input.shape
        if len(input_shape) == 2:
            h = input_shape[0]
            w = input_shape[1]
            input = input.reshape(h, w, 1)
        
        return input

    def __normalize_input(self, input):
        return (input / 255) - 0.5

    def __categorical_cross_entropy_loss(self, class_index, output):
        return -np.log(output[class_index])

    def __feedforward_output(self, input):
        input = self.__set_depth_if_grayscale(input)
        input = self.__normalize_input(input)

        output = input
        for layer in self.layers:
            output = layer.feedforward(output)

        return self.activation_layer.feedforward(output)

    def __feedforward(self, input, class_index):
        output = self.__feedforward_output(input)

        loss = self.__categorical_cross_entropy_loss(class_index, output)
        guess = np.argmax(output)
        correct = guess == class_index

        return output, loss, correct

    def __backpropagate(self, loss_gradient, rate):
        loss_gradient = self.activation_layer.backpropagate(loss_gradient, rate)

        for layer in self.reverse_layers:
            loss_gradient = layer.backpropagate(loss_gradient, rate)

    def __train_cycle(self, input, label, rate):
        class_index = self.classes.index(label)

        output, loss, correct = self.__feedforward(input, class_index)

        loss_gradient = np.zeros(len(self.classes))
        loss_gradient[class_index] = -1 / output[class_index]

        self.__backpropagate(loss_gradient, rate)

        return loss, correct

    def train(self, training_images, training_labels, classes, num_epochs=10, rate=0.001):
        self.training_images = training_images
        self.training_labels = training_labels
        self.classes = classes
        self.activation_layer = ActivationLayer(len(classes))

        print('\n\n>>> Train is leaving the station.\n');

        if not has_gnuplot:
            print('\n    NOTE: to see graphs, install `gnuplot`.\n')

        correct_last_50 = 0
        percent_correct_last_50 = 0

        for epoch in range(num_epochs):
            print(f'\n+++ Commence epoch {epoch + 1}. +++')

            rand_perm = np.random.permutation(len(self.training_images))
            t_inputs = self.training_images[rand_perm]
            t_labels = self.training_labels[rand_perm]
            t_len = len(t_inputs)

            num_correct = 0
            loss = 0
            x_training_cycles = []
            y_mean_correct = []

            for i, (input, label) in enumerate(zip(t_inputs, t_labels)):
                i_loss, i_correct = self.__train_cycle(input, label, rate)

                if (i + 1) % 50 == 0:
                    percent_correct_last_50 = round(((correct_last_50 / 50) * 100), 3)
                    correct_last_50 = 0

                num_cycles = i + 1
                log.log_progress(percent_correct_last_50, loss, num_cycles, t_len)

                x_training_cycles.append(i)
                y_mean_correct.append(percent_correct_last_50)

                correct_add = 1 if i_correct else 0
                num_correct += correct_add
                correct_last_50 += correct_add
                loss += i_loss

            log.log_epoch_results(self.layers, epoch + 1, rate, num_correct, loss, t_len, x_training_cycles, y_mean_correct, len(self.classes))
            print('\n')

        print('\nEnd training.\n')

    def predict(self, input):
        output = self.__feedforward_output(input)
        return np.argmax(output)
