import mnist
import numpy as np

import cnn
from cnn.layers.conv import ConvLayer
from cnn.layers.maxpool import MaxPoolLayer

training_images = mnist.train_images()[:1000]
training_labels = mnist.train_labels()[:1000]

layers = [
    ConvLayer(num_kernels=8),
    MaxPoolLayer(),
]
classes = [x for x in range(10)]

# initialize
n = cnn.CNN(layers)

# train
n.train(training_images, training_labels, classes, num_epochs=3, rate=0.005)

# predict
print('\n\n>>> Testing model...\n')
test_images = mnist.test_images()[:500]
test_labels = mnist.test_labels()[:500]

num_correct = 0
for image, label in zip(test_images, test_labels):
  prediction_index = n.predict(image)

  prediction = classes[prediction_index]
  correct_add = 1 if prediction == label else 0
  num_correct += correct_add

num_tests = len(test_images)
percent_accurate = round(((num_correct / num_tests) * 100), 3)
print(f'Prediction accuracy ({num_tests} attempts): {percent_accurate}%\n')
