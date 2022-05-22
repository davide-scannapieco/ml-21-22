from pathlib import Path
import random

import numpy as np
from keras import Sequential
from keras.layers import Dense
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from numpy import cos
from sklearn.model_selection import train_test_split
import tensorflow as tf
from as1_scannapieco_davide.deliverable.run_model import load_data, create_matrix_x, load_model, evaluate_predictions

from as1_scannapieco_davide.src.utils import save_sklearn_model


def polynomial_function(x, theta_hat):
    """
        definition for linear function
        :param x: numpy array of shape, theta: number of input parameters
        :return y
    """
    return theta_hat[0] + theta_hat[1] * x[:, 0] + theta_hat[2] * x[:, 1] + theta_hat[3] * cos(x[:, 1]) + theta_hat[
        4] * pow(x[:, 0], 2)


def create_ffnn(neurons=30, activation="tanh"):
    model = Sequential()
    model.add(Dense(neurons, activation=activation))
    model.add(Dense(neurons, activation=activation))
    model.add(Dense(neurons, activation=activation))
    model.add(Dense(neurons, activation=activation))
    model.add(Dense(1, activation="linear"))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=["mse"])

    return model


def variance(data):
    # Number of observations
    n = len(data)
    # Mean of the data
    mean = sum(data) / n
    # Square deviations
    deviations = [(x - mean) ** 2 for x in data]
    # Variance
    return sum(deviations) / n


def main():
    data_file = Path('../data/data.npz')
    # Load data from filename
    x, y = load_data(data_file)
    # Create matrix for this model"
    # f(x, theta) = theta_0 + theta_1 * x_1 + theta_2 * x_2 + theta_3 * cos(x_2) + theta_4 * x_1 * x_1
    matrix_x = create_matrix_x(x, y)

    ## TASK 1
    # Create object LinearRegression, assume data is centered
    lr = LinearRegression(fit_intercept=False)
    X_t, X_tst_lr, y_t, y_tst_lr = train_test_split(matrix_x, y, test_size=0.1, shuffle=True)
    lr.fit(X_t, y_t)

    theta_hat = lr.coef_
    print(theta_hat)

    save_sklearn_model(lr, Path('../deliverable/linear_regression.pickle'))

    # predict function
    y_pred_lr = lr.predict(X_tst_lr)
    # calculate error for linear regression (used for T test)
    lr_e = (y_tst_lr - y_pred_lr) ** 2
    mean_lr_e = lr_e.mean()
    # calculate Mean squared error
    mse_lr = mean_squared_error(y_pred_lr, y_tst_lr)
    print(f'Linear MSE: {mse_lr}')
    # Split arrays or matrices into random train and test subsets.
    X_t, X_tst, y_t, y_tst = train_test_split(x, y, test_size=0.1, shuffle=True)
    # # Split arrays in train and validation subsets
    X_atr, X_val, y_atr, y_val = train_test_split(X_t, y_t, test_size=0.1, shuffle=True)
    #
    # TASK 2
    # neurons - activation
    model_parameters = [(30, "tanh"), (15, "tanh"), (10, "tanh"), (20, "tanh")]
    epochs = 500
    # variable to choose the best model
    mse_list = []
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=70, mode='min'),
    # Choose which model is the best one according to MSE
    for (neurons, activation) in model_parameters:
        print(f'Training NN with {neurons} neurons and {activation} activation')
        model = create_ffnn(neurons=neurons, activation=activation)
        model.fit(X_atr, y_atr, epochs=epochs, verbose=0, batch_size=32, callbacks=[early_stopping])
        _, mse_value = model.evaluate(X_val, y_val)
        mse_list.append(mse_value)

    imin = np.argmin(mse_list)
    print("Best model parameters:", model_parameters[imin])
    (neurons, activation) = model_parameters[imin]
    best_model = create_ffnn(neurons=neurons, activation=activation)
    best_model.fit(X_t, y_t, validation_data=(X_atr, y_atr), batch_size=32, epochs=epochs, verbose=0)
    best_model.summary()


    save_sklearn_model(best_model, Path('../deliverable/non_linear_regression.pickle'))
    print("\n")
    # predict on best model
    y_pred_non_linear = best_model.predict(X_tst)
    # remove axes of length 1
    y_pred_non_linear = np.squeeze(np.asarray(y_pred_non_linear))

    # calculate errors of non linear regression
    nn_e = (y_tst - y_pred_non_linear) ** 2
    nn_mean = nn_e.mean()
    mse_nl = mean_squared_error(y_tst, y_pred_non_linear)
    print(f'Non Linear MSE: {mse_nl}')
    print("\n")
    ## TASK2 T-TEST
    print('Variances')
    v_lr = variance(lr_e)
    v_nn = variance(nn_e)

    print(f'Linear regression variance:{v_lr}')
    print(f'Non-linear regression variance:{v_nn}')
    print("\n")

    #
    # Test statistics
    t_test = (nn_mean - mean_lr_e)
    t_test /= np.sqrt(v_nn / len(X_tst) + v_lr / len(X_tst_lr))
    print(f"is T={t_test} in 95\% confidence interval (-1.96, 1.96) ?")
    #
    #
    # TEST for BONUS
    baseline_model_path = Path('../deliverable/baseline_model.pickle')
    baseline_model = load_model(baseline_model_path)

    # print(baseline_model.get_params())

    # predict on the given sample
    y_pred_base = baseline_model.predict(X_tst)
    base_err = (y_tst - y_pred_base) ** 2
    mean_e_base = base_err.mean()

    print('Variances')
    v_base = variance(base_err)
    print(f'Baseline model variance: {v_base}')
    mse_base = mean_squared_error(y_tst, y_pred_base)
    print(f'mse base equal : {mse_base}')

    t_test = (nn_mean - mean_e_base)
    t_test /= np.sqrt((v_nn  + v_base) / len(X_tst))
    print(f"is T={t_test} in 95\% confidence interval (-1.96, 1.96) ?")


if __name__ == '__main__':
    main()
