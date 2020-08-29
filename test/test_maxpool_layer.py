import numpy as np

from cnn import MaxPoolLayer

# setup test maps
test_maps_8x8x1 = np.arange(64).reshape(8, 8, 1)
test_maps_8x8x3 = np.arange(192).reshape(8, 8, 3)
test_maps_9x9x1 = np.arange(81).reshape(9, 9, 1)
test_maps_9x9x3 = np.arange(243).reshape(9, 9, 3)

# setup test layers
maxpool_layer_122 = MaxPoolLayer(stride=2, kernel_dimension=2)
maxpool_layer_322 = MaxPoolLayer(stride=2, kernel_dimension=2)
maxpool_layer_133 = MaxPoolLayer(stride=3, kernel_dimension=3)

"""
The layer should not change the number of maps in the input.
"""
def test_feedforward_map_retention():
    layer = maxpool_layer_122
    input = test_maps_8x8x1
    output = layer.feedforward(input)

    h, w, n = input.shape
    o_h, o_w, o_n = output.shape

    assert n == o_n, "Number of output maps does not equal number of input maps"

"""
A layer with stride of 2 should resize the input by half.
"""
def test_feedforward_resize_half():
    layer = maxpool_layer_122
    input = test_maps_8x8x1
    output = layer.feedforward(input)

    h, w, n = input.shape
    o_h, o_w, o_n = output.shape

    assert h // 2 == o_h, "Output height not half of input height"
    assert w // 2 == o_w, "Output width not half of input width"

"""
A layer with stride of 3 should resize the input by one-third.
"""
def test_feedforward_resize_third():
    layer = maxpool_layer_133
    input = test_maps_9x9x1
    output = layer.feedforward(input)

    h, w, n = input.shape
    o_h, o_w, o_n = output.shape

    assert h // 3 == o_h, "Output height not one-third of input height"
    assert w // 3 == o_w, "Output width not one-third of input width"

"""
The layer should perform an 'amax' operation on the input such that the image is dilated only, not translated or otherwise changed.
"""
def test_feedforward_picture_retention():
    layer = maxpool_layer_122
    input = test_maps_8x8x1
    output = layer.feedforward(input)

    assert np.amax(input[2:4, 2:4, 0], axis=(0, 1)) == output[1, 1], "Output map is not 2x2 amax of input map"
