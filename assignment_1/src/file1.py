from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from numpy import cos
from assignment_1.deliverable.run_model import load_data, create_matrix_x
import matplotlib.pyplot as plt
import pickle


def main():
    data_file = Path('../data/data.npz')
    # Load data from filename
    x, y = load_data(data_file)
    # Create matrix for this model"
    # f(x, theta) = theta_0 + theta_1 * x_1 + theta_2 * x_2 + theta_3 * cos(x_2) + theta_4 * x_1 * x_1
    matrix_x = create_matrix_x(x, y)

    # Create object LinearRegression, assume data is centered
    lr = LinearRegression(fit_intercept=False)

    # estimate parameters
    lr.fit(matrix_x, y)

    # get coefficient
    theta_hat = lr.coef_

    # predict function
    y_pred = lr.predict(matrix_x)
    # calculate Mean squared error
    error = mean_squared_error(y, y_pred)
    print(error)

    # plot 3d figure
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_trisurf(x[:, 0], x[:, 1], polynomial_function(x, theta_hat), cmap='viridis', edgecolor='none')

    # Saving into pickle file object LinearRegression
    with open(Path('../deliverable/linear_regression.pickle'), 'wb') as f:
        pickle.dump(lr, f)
    pass


def polynomial_function(x, theta_hat):
    """
        definition for linear function
        :param x: numpy array of shape, theta: number of input parameters
        :return y
    """
    return theta_hat[0] + theta_hat[1] * x[:, 0] + theta_hat[2] * x[:, 1] + theta_hat[3] * cos(x[:, 1]) + theta_hat[
        4] * pow(x[:, 0], 2)


if __name__ == '__main__':
    main()
