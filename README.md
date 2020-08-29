<div align="center">
  <h1>cnn4fun</h1>
  <p>This is a rather basic <b>C</b>onvolutional <b>N</b>eural <b>N</b>etwork.</p>
</div>

<div align="center">
  <img src="https://github.com/jaredgorski/cnn4fun/raw/master/.media/screenshot.png" width="600" />
</div>

The `cnn` package contains a primary `CNN` class as well as convolution and max-pooling layers at `cnn.layers.conv.ConvLayer` and `cnn.layers.maxpool.MaxPoolLayer`, respectively. These layers can be configured along with the learning rate in order to fine-tune the training of the network. This network currently works with the [MNIST handwritten digits dataset](http://yann.lecun.com/exdb/mnist/), which can be tested by running `python run_mnist.py`.

The network supports both grayscale and RGB images.

#### To run with the MNIST dataset:
1. clone this repo locally
2. have Python 3 and pip installed on your machine
3. install dependencies with `pip install -r requirements.txt`
4. run `python run_mnist.py`

To run unit tests, run `python -m pytest`.

#### To do:
- improve unit test coverage and depth
- improve activation layer logic
- improve code clarity/internal documentation
