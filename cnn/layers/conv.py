import numpy as np

class ConvLayer:

    def __init__(self, num_kernels, kernel_dimension=3, stride=1):
        self.l_name = f'Conv num_kernels={num_kernels} kernel_dim={kernel_dimension} stride={stride}'
        self.num_kernels = num_kernels
        self.kernel_dimension = kernel_dimension
        self.stride = stride
        self.biases = np.zeros((num_kernels))

        self.depth = None
        self.kernels = []

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

        valid_padding = (h - h_stride) // 2

        return h_stride, w_stride, valid_padding

    def __slice_generator(self, input):
        stride = self.stride
        size = self.kernel_dimension

        h_stride, w_stride, valid_padding = self.__calc_stride_dimensions(input)
        npad = ((valid_padding, valid_padding), (0, 0), (0, 0))
        input = np.pad(input, npad)

        for i in range(h_stride):
            for j in range(w_stride):
                slice = input[(i * stride):((i * stride) + size), (j * stride):((j * stride) + size), 0:self.depth]
                yield slice, i, j 

    def __init_kernels(self):
        if self.depth and len(self.kernels) == 0:
            kernel_dim = self.kernel_dimension
            self.kernels = np.random.randn(self.num_kernels, kernel_dim, kernel_dim, self.depth) / (kernel_dim ** kernel_dim)

    def feedforward(self, input):
        self.prev_input = input

        input_shape = input.shape
        h = input_shape[0]
        w = input_shape[1]

        self.depth = input.shape[2]

        self.__init_kernels()
        num_kernels = self.num_kernels

        output = np.zeros((h, w, num_kernels))

        for i in range(num_kernels):
            for slice, j, k in self.__slice_generator(input):
                output[j, k, i] = np.sum(slice * self.kernels[i]) + self.biases[i]

        return output

    def backpropagate(self, loss_gradient, rate):
        prev_input = self.prev_input

        num_kernels = self.num_kernels
        delta_kernels = np.zeros(self.kernels.shape)
        delta_bias = np.zeros((num_kernels, 1))
        delta_loss_gradient = np.zeros(prev_input.shape)

        for i in range(num_kernels):
            for slice, j, k in self.__slice_generator(prev_input):
                delta_kernels[i] += loss_gradient[j, k, i] * slice

        for i in range(num_kernels):
            for slice, j, k in self.__slice_generator(loss_gradient):
                delta_loss_gradient[j, k] += np.sum(slice * self.kernels[i])
            
            # TODO: clarify bias update
            delta_bias[i] = np.sum(loss_gradient[:, :, i])

        self.kernels -= rate * delta_kernels
        self.bias = delta_bias

        return delta_loss_gradient
