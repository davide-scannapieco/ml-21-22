from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dropout, Dense, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras import Sequential, applications
from utils import save_keras_model

from assignment_2.src.task1 import show, initial_setup


def main():
    # TASK 2 step 1 and 2 same as TASK 1
    x_train, x_test, y_train, y_test = initial_setup()
    n_classes = 3

    def evaluate_predictions(y_true, y_pred):
        """
        Evaluates the mean squared error between the values in y_true and the values
        in y_pred.
        ### YOU CAN NOT EDIT THIS FUNCTION ###

        :param y_true: Numpy array, the true target values from the test set;
        :param y_pred: Numpy array, the values predicted by your model.
        :return: float, the the mean squared error between the two arrays.
        """
        assert y_true.shape == y_pred.shape
        return ((y_true - y_pred) ** 2).mean()

    def build_fully_connected_neural_network(lr=0.003, n=20):
        model = Sequential()
        model.add(Flatten())
        model.add(Dense(n, activation='relu', input_shape=(32, 32, 3)))
        model.add(Dense(n, activation='relu'))
        model.add(Dense(n_classes, activation='softmax'))

        model.compile(optimizer=RMSprop(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    model = build_fully_connected_neural_network()
    epochs = 500
    batch_size = 128
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

    h = model.fit(x_train,
                  y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_split=0.2,
                  callbacks=[early_stopping])
    model.summary()

    save_keras_model(model, "../deliverable/nn_task2.h5")

    # Plot with epochs on x-axis, train accuracy and validation accuracy
    show(h)

    loss, accuracy = model.evaluate(x_test, y_test)
    print(f'Test Loss: {loss}  -  Accuracy: {accuracy}')

    y_pred = model.predict(x_test)
    mse = evaluate_predictions(y_test, y_pred)
    print(f'MSE: {mse}')


if __name__ == '__main__':
    main()
