import mnist
import numpy as np
import pickle

import cnn

training_images = mnist.train_images()[:5000]
training_labels = mnist.train_labels()[:5000]

## uncomment below to train mnist images as RGB data
# import cv2
# training_images_rgb = []
# for i, image in enumerate(training_images):
#     training_images_rgb.append(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))
# training_images = np.array(training_images_rgb)

classes = [x for x in range(10)]

# initialize
net = None
answer = input("Would you like to load a model? (enter 'y' to load): ")
should_load = answer == 'y'
if should_load:
    filename = input("Enter a filename (without the extension): ")
    pickle_in = open(f'{filename}.pickle','rb')
    net = pickle.load(pickle_in)
else:
    layers = [
        cnn.layers.Conv(num_kernels=16),
        cnn.layers.MaxPool(),
        cnn.layers.SoftMax(num_classes=10),
    ]
    net = cnn.CNN(layers)

# train
answer = input("Would you like to train? (enter 'y' to train): ")
should_train = answer == 'y'
if should_train:
    net.train(training_images, training_labels, classes, num_epochs=3, rate=0.005)

# predict
answer = input("Would you like to test the model? (enter 'y' to test): ")
should_test = answer == 'y'
if should_test:
    print('\n\n>>> Testing model...\n')
    test_images = mnist.test_images()[:1000]
    test_labels = mnist.test_labels()[:1000]

    num_correct = 0
    for image, label in zip(test_images, test_labels):
        prediction_index = net.predict(image)

        prediction = classes[prediction_index]
        correct_add = 1 if prediction == label else 0
        num_correct += correct_add

    num_tests = len(test_images)
    percent_accurate = round(((num_correct / num_tests) * 100), 3)
    print(f'Prediction accuracy ({num_tests} attempts): {percent_accurate}%\n')

# save model
answer = input("Would you like to save the model? (enter 'y' to save): ")
should_save = answer == 'y'
if should_save:
    filename = input("Enter a filename (without the extension): ")
    with open(f'{filename}.pickle','wb') as f:
        pickle.dump(net, f)
