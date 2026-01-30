import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import relu, linear
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam


def generate_data():
    """
    Generates test data. This is just for illustration purposes. Not supposed
    to work.
    """
    # gen_data is a DRL-specific function.
    X, y, x_ideal, y_ideal = gen_data(18, 2, 0.7)
    print("X.shape", X.shape, "y.shape", y.shape)

    # split the data using sklearn routine
    # EXAMPLE 1: Only training and test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=1
    )
    print("X_train.shape", X_train.shape, "y_train.shape", y_train.shape)
    print("X_test.shape", X_test.shape, "y_test.shape", y_test.shape)

    # EXAMPLE 2: Training, test and cross-validation set
    X_train, X_, y_train, y_ = train_test_split(X, y, test_size=0.40, random_state=1)
    X_cv, X_test, y_cv, y_test = train_test_split(
        X_, y_, test_size=0.50, random_state=1
    )
    print("X_train.shape", X_train.shape, "y_train.shape", y_train.shape)
    print("X_cv.shape", X_cv.shape, "y_cv.shape", y_cv.shape)
    print("X_test.shape", X_test.shape, "y_test.shape", y_test.shape)


def plot_data():
    """
    Also illustrative for plotting data.
    """
    # EXAMPLE 1: Only training and test set
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.plot(x_ideal, y_ideal, "--", color="orangered", label="y_ideal", lw=1)
    ax.set_title("Training, Test", fontsize=14)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.scatter(X_train, y_train, color="red", label="train")
    ax.scatter(X_test, y_test, color=dlc["dlblue"], label="test")
    ax.legend(loc="upper left")
    plt.show()

    # EXAMPLE 2: Training, test and cross-validation set
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.plot(x_ideal, y_ideal, "--", color="orangered", label="y_ideal", lw=1)
    ax.set_title("Training, CV, Test", fontsize=14)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.scatter(X_train, y_train, color="red", label="train")
    ax.scatter(X_cv, y_cv, color=dlc["dlorange"], label="cv")
    ax.scatter(X_test, y_test, color=dlc["dlblue"], label="test")
    ax.legend(loc="upper left")
    plt.show()


def eval_mse(y, yhat):
    """
    Calculate the mean squared error on a data set. Used to evaluate a
    linear regression algorithm
    Args:
      y    : (ndarray  Shape (m,) or (m,1))  target value of each example
      yhat : (ndarray  Shape (m,) or (m,1))  predicted value of each example
    Returns:
      err: (scalar)
    """
    m = len(y)
    err = 0.0
    for i in range(m):
        ### START CODE HERE ###
        err += (yhat[i] - y[i]) ** 2

    err = err / (2 * m)
    ### END CODE HERE ###

    return err


def eval_cat_err(y, yhat):
    """
    Calculate the categorization error
    Args:
      y    : (ndarray  Shape (m,) or (m,1))  target value of each example
      yhat : (ndarray  Shape (m,) or (m,1))  predicted value of each example
    Returns:|
      cerr: (scalar)
    """
    m = len(y)
    incorrect = 0
    for i in range(m):
        ### START CODE HERE ###
        if yhat[i] != y[i]:
            incorrect += 1

    cerr = incorrect / m
    ### END CODE HERE ###

    return cerr


def evaluate_model():
    """
    Evaluate the model's performance on the training vs test set
    """
    degree = 10
    lmodel = lin_model(degree)
    lmodel.fit(X_train, y_train)

    # predict on training data, find training error
    yhat = lmodel.predict(X_train)
    err_train = lmodel.mse(y_train, yhat)

    # predict on test data, find error
    yhat = lmodel.predict(X_test)
    err_test = lmodel.mse(y_test, yhat)

    print(f"training err {err_train:0.2f}, test err {err_test:0.2f}")


def find_optimal_polynomial_degree():
    """
    Proves that the cv set is useful for evaluating high variance (overfitting)
    or high bias (underfitting)
    """
    max_degree = 9
    err_train = np.zeros(max_degree)
    err_cv = np.zeros(max_degree)
    x = np.linspace(0, int(X.max()), 100)
    y_pred = np.zeros((100, max_degree))  # columns are lines to plot

    for degree in range(max_degree):
        lmodel = lin_model(degree + 1)
        lmodel.fit(X_train, y_train)
        yhat = lmodel.predict(X_train)
        err_train[degree] = lmodel.mse(y_train, yhat)
        yhat = lmodel.predict(X_cv)
        err_cv[degree] = lmodel.mse(y_cv, yhat)
        y_pred[:, degree] = lmodel.predict(x)

    optimal_degree = np.argmin(err_cv) + 1
