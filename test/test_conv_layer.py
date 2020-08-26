import numpy as np

from cnn.layers.conv import ConvLayer

# setup test maps
test_maps_8x8x1 = np.arange(64).reshape(8, 8, 1)
test_maps_8x8x3 = np.arange(192).reshape(8, 8, 3)

# setup test layers
conv_layer_3x3 = ConvLayer(num_kernels=5, kernel_dimension=3)
conv_layer_5x5 = ConvLayer(num_kernels=5, kernel_dimension=5)

"""
A layer with 5 kernels should result in output maps with depth=5.
"""
def test_feedforward_kernel_output_maps():
    layer = conv_layer_3x3
    input = test_maps_8x8x3
    output = layer.feedforward(input)

    o_h, o_w, o_n = output.shape

    assert 5 == o_n, "Number of output maps does not account for kernels"

"""
Layer should apply valid padding, so output dimensions should match input dimensions.
"""
def test_valid_padded_output():
    layer = conv_layer_3x3
    input = test_maps_8x8x3
    output = layer.feedforward(input)

    h, w, n = input.shape
    o_h, o_w, o_n = output.shape

    assert h == o_h, "Output height does not match input height"
    assert w == o_w, "Output width does not match input width"
