from tensorflow.keras import Sequential
from tensorflow.keras import utils
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dense, Flatten
from tensorflow.keras.optimizers import RMSprop

from utils import load_cifar10, save_keras_model

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

idx_to_label = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird'
}

label_to_idx = {
    'airplane': 0,
    'automobile': 1,
    'bird': 2
}


def plot_sample(imgs, labels, nrows, ncols, resize=None, tograyscale=False):
    # create a grid of images
    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    # take a random sample of images
    indices = np.random.choice(len(imgs), size=nrows * ncols, replace=False)
    for ax, idx in zip(axs.reshape(-1), indices):
        ax.axis('off')
        # sample an image
        ax.set_title(idx_to_label[labels[idx][0]])
        im = imgs[idx]
        if isinstance(im, np.ndarray):
            im = Image.fromarray(im)
        if resize is not None:
            im = im.resize(resize)
        if tograyscale:
            im = im.convert('L')
        ax.imshow(im, cmap='gray')


def initial_setup():
    (x_train, y_train), (x_test, y_test) = load_cifar10()

    # plot_sample(x_train, y_train, 3, 3)

    # Normalization pixels in [0,1] range
    x_train = x_train / 255.
    x_test = x_test / 255.

    # Pre-process targets
    n_classes = 3
    y_train = utils.to_categorical(y_train, n_classes)
    y_test = utils.to_categorical(y_test, n_classes)
    return x_train, x_test, y_train, y_test


def show(h):
    plt.plot(h.history['accuracy'], label='Accuracy')
    plt.plot(h.history['val_accuracy'], label='Val_Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()


# Build neural network
def build_neural_network_model(lr=0.003, n=8):
    model = Sequential()
    model.add(Conv2D(8, (5, 5), activation="relu", strides=(1, 1), input_shape=(32, 32, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (3, 3), activation="relu", strides=(2, 2)))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    # from 2D to vectors
    model.add(Flatten())
    model.add(Dense(n, activation="tanh"))
    # 3 = #n_classes
    model.add(Dense(3, activation="softmax"))

    model.compile(optimizer=RMSprop(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# Grid Search Model
def optional_configuration_model(x_train, y_train, x_test, y_test, early_stopping, batch_size, epochs):
    best_model, best_lr, best_n, best_acc = 0, 0, 0, 0
    l_rates = [0.01, 0.0001]
    n_neurons = [16, 64]

    for l_rate in l_rates:
        for n_neuron in n_neurons:
            print(f'Learning Rate : {l_rate}   and #neurons: {n_neuron}')
            model_x = build_neural_network_model(l_rate, n_neuron)

            model_x.fit(x_train,
                        y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=0.2,
                        callbacks=[early_stopping])

            loss, accuracy = model_x.evaluate(x_test, y_test)
            print(f'Accuracy: {accuracy}')
            if best_acc < accuracy:
                best_model = model_x
                best_lr = l_rate
                best_n = n_neuron
                best_acc = accuracy
    loss, accuracy = best_model.evaluate(x_test, y_test)
    print(f'Test Loss: {loss}  -  Accuracy: {accuracy}')
    save_keras_model(best_model, "../deliverable/best_nn_task1.h5")
    print(f'Learning Rate best model: {best_lr}  -- Neurons: {best_n}')


def main():
    # Download and Load CIFAR-10 Dataset
    x_train, x_test, y_train, y_test = initial_setup()

    model = build_neural_network_model()
    # # train the model
    epochs = 500
    batch_size = 128
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
    #
    h = model.fit(x_train,
                  y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_split=0.2,
                  callbacks=[early_stopping])

    # save model
    save_keras_model(model, "../deliverable/nn_task1.h5")

    # Plot with epochs on x-axis, train accuracy and validation accuracy
    show(h)

    # Assess performance on test set

    loss, accuracy = model.evaluate(x_test, y_test)
    print(f'Test Loss: {loss}  -  Accuracy: {accuracy}')

    # Optional Task
    #
    # function call Optional Task
    optional_configuration_model(x_train, y_train, x_test, y_test, early_stopping, batch_size, epochs)


if __name__ == '__main__':
    main()
