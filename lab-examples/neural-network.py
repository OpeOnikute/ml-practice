import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential

plt.style.use("./deeplearning.mplstyle")
import tensorflow as tf
from lab_utils_common import dlc, sigmoid
from lab_coffee_utils import (
    load_coffee_data,
    plt_roast,
    plt_prob,
    plt_layer,
    plt_network,
    plt_output_unit,
)
import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)


def load_coffee_data():
    return 0, 0


# sigmoid
def g():
    pass


def my_sequential(x, W1, b1, W2, b2):
    a1 = my_dense(x, W1, b1)
    a2 = my_dense(a1, W2, b2)
    return a2


def my_predict(X, W1, b1, W2, b2):
    m = X.shape[0]
    p = np.zeros((m, 1))
    for i in range(m):
        p[i, 0] = my_sequential(X[i], W1, b1, W2, b2)
    return p


def my_dense(a_in, W, b):
    """
    Computes dense layer
    Args:
      a_in (ndarray (n, )) : Data, 1 example
      W    (ndarray (n,j)) : Weight matrix, n features per unit, j units
      b    (ndarray (j, )) : bias vector, j units
    Returns
      a_out (ndarray (j,))  : j units
    """
    # Calculate the activation for each neuron, i.e. each column
    units = W.shape[1]
    a_out = np.zeros(units)
    for j in range(units):
        w = W[:, j]
        z = np.dot(w, a_in) + b[j]
        a_out[j] = g(z)
    return a_out


def my_softmax(z):
    """Softmax converts a vector of values to a probability distribution.
    Args:
      z (ndarray (N,))  : input data, N features
    Returns:
      a (ndarray (N,))  : softmax of z
    """
    z_len = len(z)
    sum_ezk = 0
    for k in range(z_len):
        sum_ezk += np.exp(z[k])

    a = np.array(np.zeros(z_len))
    for j in range(z_len):
        e_zj = np.exp(z[j])
        a_j = e_zj / sum_ezk
        a[j] = a_j

    return a


def neural_sample():
    """
    Example syntax of building a neural network
    """
    tf.random.set_seed(1234)
    model = Sequential(
        [
            Dense(units=120, activation="relu", name="L1"),
            Dense(units=40, activation="relu", name="L2"),
            Dense(units=6, activation="linear", name="L3"),
        ],
        name="Complex",
    )
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(0.01),
    )

    # Test the model
    model_s.fit(X_train, y_train, epochs=1000)

    model_s.summary()
    model_s_test(model_s, classes, X_train.shape[1])

    model_predict_s = lambda Xl: np.argmax(
        tf.nn.softmax(model_s.predict(Xl)).numpy(), axis=1
    )
    plt_nn(
        model_predict_s, X_train, y_train, classes, X_cv, y_cv, suptitle="Simple Model"
    )


if __name__ == "__main__":
    X, Y = load_coffee_data()
    print(X.shape, Y.shape)
    print(
        f"Temperature Max, Min pre normalization: {np.max(X[:,0]):0.2f}, {np.min(X[:,0]):0.2f}"
    )
    print(
        f"Duration    Max, Min pre normalization: {np.max(X[:,1]):0.2f}, {np.min(X[:,1]):0.2f}"
    )
    norm_l = tf.keras.layers.Normalization(axis=-1)
    norm_l.adapt(X)  # learns mean, variance
    Xn = norm_l(X)
    print(
        f"Temperature Max, Min post normalization: {np.max(Xn[:,0]):0.2f}, {np.min(Xn[:,0]):0.2f}"
    )
    print(
        f"Duration    Max, Min post normalization: {np.max(Xn[:,1]):0.2f}, {np.min(Xn[:,1]):0.2f}"
    )

    X_tst = np.array([[200, 13.9], [200, 17]])  # postive example  # negative example
    X_tstn = norm_l(X_tst)  # remember to normalize
    predictions = my_predict(X_tstn, W1_tmp, b1_tmp, W2_tmp, b2_tmp)

    yhat = np.zeros_like(predictions)
    for i in range(len(predictions)):
        if predictions[i] >= 0.5:
            yhat[i] = 1
        else:
            yhat[i] = 0
    print(f"decisions = \n{yhat}")

    yhat = (predictions >= 0.5).astype(int)
    print(f"decisions = \n{yhat}")
