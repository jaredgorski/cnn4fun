<div align="center">
  <h1>cnn4fun</h1>
  <p>This is a rather basic <b>C</b>onvolutional <b>N</b>eural <b>N</b>etwork.</p>
</div>

<div align="center">
  <img src="https://github.com/jaredgorski/cnn4fun/raw/master/.media/screenshot.png" width="600" />
</div>

The `cnn` package contains a primary `cnn.CNN` class as well as convolution, max-pooling, and softmax activation layers at `cnn.layers.Conv`, `cnn.layers.MaxPool` and `cnn.layers.SoftMax`, respectively. These layers can be configured along with the learning rate in order to fine-tune the training of the network. This network currently works with the [MNIST handwritten digits dataset](http://yann.lecun.com/exdb/mnist/), which can be tested by running `python run_mnist.py`.

The network supports both grayscale and RGB images.

## To run with the MNIST dataset:
1. clone this repo locally
2. have Python 3 and pip installed on your machine
3. install dependencies with `pip install -r requirements.txt`
4. run `python run_mnist.py`

## Package usage
```python
# package must exist locally, whether cloned or copied into a project
import cnn

# get training images (RGB or grayscale) and labels, ordered
training_images = get_ordered_images_list()
training_labels = get_ordered_labels_list()

# define list of classes
classes = ['cat', 'dog']

# initialize layer stack
layers = [
    cnn.layers.Conv(num_kernels=16, kernel_dimension=5, stride=1),
    cnn.layers.MaxPool(kernel_dimension=2, stride=2),
    cnn.layers.Conv(num_kernels=16, kernel_dimension=3, stride=1),
    cnn.layers.MaxPool(kernel_dimension=2, stride=2),
    cnn.layers.SoftMax(num_classes=2),
]

# initialize network object
net = cnn.CNN(layers)

# train
net.train(training_images, training_labels, classes, num_epochs=20, rate=0.001)

# get test image and label
test_image = get_dog_png()
test_label = 'dog'

# test model prediction
prediction_index = net.predict(test_image)

prediction = classes[prediction_index]
correct = prediction == test_label
```

#### Tests:
- To run unit tests, run `python -m pytest`.

## To do:
- improve unit test coverage and depth
- improve code clarity/internal documentation
