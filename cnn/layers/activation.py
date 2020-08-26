import numpy as np

class ActivationLayer:

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.weights = []
        self.biases = []

    def __relu_activation(self, input):
        return max(0.0, input)

    def __flatten_and_rectify(self, input):
        input = input.flatten()
        # TODO: implement backprop for relu before uncommenting this
        # input = [self.__relu_activation(x) for x in input]

        return input

    def __init_weights_and_biases(self):
        if self.flattened_input_len and len(self.weights) == 0 and len(self.biases) == 0:
            input_len = self.flattened_input_len
            num_classes = self.num_classes

            self.weights = np.random.randn(input_len, num_classes) / input_len
            self.biases = np.zeros(num_classes)

    def feedforward(self, input):
        self.prev_input_shape = input.shape

        input = self.__flatten_and_rectify(input)

        self.flattened_input_len = len(input)
        self.prev_input = input

        self.__init_weights_and_biases()

        totals = np.dot(input, self.weights) + self.biases
        self.prev_totals = totals

        exp = np.exp(totals)
        return exp / np.sum(exp, axis=0)

    def backpropagate(self, loss_gradient, rate):
        for i, gradient in enumerate(loss_gradient):
            if gradient == 0:
                continue

            totals_exp = np.exp(self.prev_totals)
            totals_exp_sum = np.sum(totals_exp)

            delta_gradient_total = -totals_exp[i] * totals_exp / (totals_exp_sum ** 2)
            delta_gradient_total[i] = totals_exp[i] * (totals_exp_sum - totals_exp[i]) / (totals_exp_sum ** 2)

            delta_totals_weights = self.prev_input
            delta_totals_bias = 1

            delta_loss_gradient_totals = gradient * delta_gradient_total

            delta_loss_weights = delta_totals_weights[np.newaxis].T @ delta_loss_gradient_totals[np.newaxis]
            delta_loss_biases = delta_loss_gradient_totals * delta_totals_bias
            delta_loss_gradient = self.weights @ delta_loss_gradient_totals

            self.weights -= rate * delta_loss_weights
            self.biases -= rate * delta_loss_biases

            return delta_loss_gradient.reshape(self.prev_input_shape)
