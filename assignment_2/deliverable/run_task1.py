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
    model_task1 = load_model('./nn_task1.h5')
    best_task_model1 = load_model('./best_nn_task1.h5')

    # Predict on the given samples
    loss, accuracy = model_task1.evaluate(x_test, y_test)
    print(f'Accuracy model task 1: {accuracy}, Loss model task 1: {loss}')
    best_loss, best_accuracy = best_task_model1.evaluate(x_test, y_test)
    print(f'Accuracy best model task 1: {best_accuracy}, Loss best model task 1: {best_loss}')

    # T-Test
    v_model_task1 = accuracy * (1 - accuracy)
    v_model_best_task1 = best_accuracy * (1 - best_accuracy)

    T = (best_accuracy - accuracy)
    T /= np.sqrt(v_model_best_task1 / len(x_test) + v_model_task1 / len(x_test))
    print(f"is T={T} in 95% confidence interval (-1.96, 1.96) ?")
