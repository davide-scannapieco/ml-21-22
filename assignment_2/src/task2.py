from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dropout, Dense, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras import Sequential, applications
from utils import save_keras_model

from assignment_2.src.task1 import show, initial_setup


def main():
    # TASK 2 step 1 and 2 same as TASK 1
    x_train, x_test, y_train, y_test = initial_setup()

    model = Sequential()
    model.add(Flatten())
    model.add(Dense(20, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer=RMSprop(lr=0.003), loss='categorical_crossentropy', metrics=['accuracy'])

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


if __name__ == '__main__':
    main()
