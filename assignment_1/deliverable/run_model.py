import joblib
import numpy as np
import matplotlib.pyplot as plt


def load_data(filename):
    """
    Loads the data from a saved .npz file.
    ### YOU CAN NOT EDIT THIS FUNCTION ###

    :param filename: string, path to the .npz file storing the data.
    :return: two numpy arrays:
        - x, a Numpy array of shape (n_samples, n_features) with the inputs;
        - y, a Numpy array of shape (n_samples, ) with the targets.
    """
    data = np.load(filename)
    x = data['x']
    y = data['y']

    return x, y


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


def load_model(filename):
    """
    Loads a Scikit-learn model saved with joblib.dump.
    This is just an example, you can write your own function to load the model.
    Some examples can be found in src/utils.py.

    :param filename: string, path to the file storing the model.
    :return: the model.
    """
    model = joblib.load(filename)

    return model


def create_matrix_x(x, y):
    ones_col_vec = np.ones(shape=(y.shape[0], 1))
    cosx = np.cos(x[:, 1]).reshape((-1, 1))
    x_2 = np.power(x[:, 0], 2).reshape((-1, 1))
    matrix_X = np.hstack((ones_col_vec, x, cosx, x_2))
    return matrix_X


if __name__ == '__main__':
    # Load the data
    # This will be replaced with the test data when grading the assignment
    data_path = '../data/data.npz'
    x, y = load_data(data_path)

    ############################################################################
    # EDITABLE SECTION OF THE SCRIPT: if you need to edit the script, do it here
    ############################################################################

    # Load the trained model
    baseline_model_path = './baseline_model.pickle'
    baseline_model = load_model(baseline_model_path)
    linear_regression_model_path = './linear_regression.pickle'
    linear_regression_model = load_model(linear_regression_model_path)

    # Predict on the given samples
    y_pred = baseline_model.predict(x)
    # TODO: check if the predict has to be on the given sample or on the matrix generated
    matrix = create_matrix_x(x, y)
    y_pred_lin_reg = linear_regression_model.predict(matrix)

    ############################################################################
    # STOP EDITABLE SECTION: do not modify anything below this point.
    ############################################################################

    # Evaluate the prediction using MSE
    mse = evaluate_predictions(y_pred, y)
    print('MSE: {}'.format(mse))
