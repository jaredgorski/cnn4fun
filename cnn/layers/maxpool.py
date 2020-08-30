import numpy as np

class MaxPool:

    def __init__(self, kernel_dimension=2, stride=2):
        self.l_name = f'MaxPool kernel_dim={kernel_dimension} stride={stride}'
        self.depth = None
        self.kernel_dimension = kernel_dimension
        self.stride = stride

    def __calc_stride_dimensions(self, input):
        input_shape = input.shape
        h = input_shape[0]
        w = input_shape[1]
        kernel_dimension = self.kernel_dimension
        stride = self.stride

        assert h % stride == 0, "Invalid stride for input height"
        assert w % stride == 0, "Invalid stride for input width"
        assert kernel_dimension >= stride, "Kernel will not cover input"

        h_stride = ((h - kernel_dimension) // stride) + 1
        w_stride = ((w - kernel_dimension) // stride) + 1

        return h_stride, w_stride

    def __slice_generator(self, input):
        stride = self.stride
        size = self.kernel_dimension
        depth = input.shape[2]

        h_stride, w_stride = self.__calc_stride_dimensions(input)

        for i in range(h_stride):
            for j in range(w_stride):
                for k in range(depth):
                    slice = input[(i * stride):((i * stride) + size), (j * stride):((j * stride) + size), k]
                    yield slice, i, j, k

    def feedforward(self, input):
        self.prev_input = input

        depth = input.shape[2]

        h_stride, w_stride = self.__calc_stride_dimensions(input)
        output = np.zeros((h_stride, w_stride, depth))

        for slice, i, j, k in self.__slice_generator(input):
            output[i, j, k] = np.amax(slice)

        return output

    def backpropagate(self, loss_gradient, _):
        prev_input = self.prev_input
        stride = self.stride

        delta_loss_gradient = np.zeros(prev_input.shape)

        for slice, i, j, k in self.__slice_generator(prev_input):
            h, w = slice.shape
            amax = np.amax(slice)

            for m in range(h):
                for n in range(w):
                    if slice[m, n] == amax:
                        delta_loss_gradient[(i * stride) + m, (j * stride) + n, k] = loss_gradient[i, j, k]

        return delta_loss_gradient
