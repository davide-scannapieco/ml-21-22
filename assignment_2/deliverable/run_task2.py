import numpy as np
from tensorflow.keras import utils
from tensorflow.keras.models import load_model
from utils import load_cifar10

if __name__ == '__main__':
    # Load the test CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = load_cifar10()

    # Preprocessing
    x_train = x_train / 255.
    x_test = x_test / 255.
    n_classes = 3
    y_train = utils.to_categorical(y_train, n_classes)
    y_test = utils.to_categorical(y_test, n_classes)

    # Load the trained models (one or more depending on task and bonus)
    model_task1 = load_model('./nn_task2.h5')

    # Predict on the given samples
    loss, accuracy = model_task1.evaluate(x_test, y_test)
    print(f'Accuracy model task 2: {accuracy}, Loss model task 2: {loss}')
