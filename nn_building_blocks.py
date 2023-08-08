"""
This script contains the basic building blocks for creating a
deep feed-forward neural network from scratch.

Typical usage example:
    # define neural network hyperparameters
    hyperparameters = {
        'nn_structure': (4, 4, 1),
        'epochs': 5_000,
        'lr': 0.0075,
        'nonlinear_functions': {'relu': relu, 'sigmoid': sigmoid},
    }

    hyperparameters['L'] = len(hyperparameters['nn_structure']) - 1

    X_train, Y_train, X_test, Y_test = load_dataset()
    #Train model
    parameters, costs = nn_model(X_train, Y_train, hyperparameters)
    #Test model
    y_hat, accuracy = predict(parameters, hyperparameters, X_test, Y_test)
"""

import numpy as np

g = np.random.default_rng(42)


def initialize_parameters(layers):
    """
    Randomly initializes the weights and biases of the neural network

    Arguments:
    layers (iterable)--contains the number of neurons in each layer of
                       the neural network - including the input layer.

    Returns:
    parameters (dict) -- python dictionary containing model parameters
                         "W1", "b1", ... , "WL", "bL"
    """
    parameters = dict()
    L = len(layers)

    for l in range(1, L):
        n_l, n_p = layers[l], layers[l-1]
        parameters['W%d' % l] = g.normal(size=(n_l, n_p), scale=0.01)
        parameters['b%d' % l] = np.zeros((n_l, 1))
    return parameters


def relu(Z):
    """
    Assumes Z is a numpy ndarray. Returns the activations of Z

    Arguments:
    Z -- an numpy ndarray of any shape

    Returns:
    A -- an numpy ndarray of activations. Same shape as Z
    """
    return np.maximum(0, Z)


def sigmoid(Z):
    """
    Assumes Z in an numpy ndarray. Returns the activation of Z

    Arugments:
    Z -- an numpy ndarray of any shape

    Returns:
    A -- an numpy ndarray of activations. Same shape as Z
    """
    return 1.0 / (1.0 + np.exp(-Z))


def forward_propagate(A, W, b, g):
    """
    Implements the forward propagation of a signal.

    Arguments:
    A -- an numpy ndarray of activations from previous layer
    W -- a numpy ndarray of weights
    b -- bias, numpy ndarray of shape (size of the current layer, 1)
    g -- a non-linear activation function i.e (tanh, relu, sigmoid)

    Returns:
    A -- activations for current layer
    """
    Z = W.dot(A) + b
    return g(Z), Z


def compute_cost(yhat, Y, m):
    """
    Implements the cross-entropy loss function

    Arguments:
    yhat -- the neural network output, a numpy ndarray. Same shape as Y
    Y -- a numpy ndarray of target labels.
    m (int) -- number of samples in the training set.

    Returns:
    J (float) -- the overall cost
    """
    return (- 1 / m) * np.sum((Y * np.log(yhat)) + ((1-Y) * np.log(1-yhat)))


def sigmoid_derivative(Z):
    """
    Assumes Z is the input of the sigmoid function. Returns the
    derivative of the sigmoid function with respect to Z.

    Arguments:
    Z -- input passed into the sigmoid function, an numpy ndarray

    Returns:
    sigmoid'(Z) -- the derivative of the sigmoid function with respect to Z
    """
    A = sigmoid(Z)
    return A * (1 - A)


def relu_derivative(Z):
    """
    Assumes Z is the input to the ReLU function. Returns the
    derivative of the ReLU function with respect to Z

    Arguments:
    Z -- input to the ReLU function, an numpy ndarray

    Returns:
    ReLU'(Z) -- the derivative of the ReLU function with respect to Z
    """
    return np.array(Z > 0, dtype=np.float32)


def tanh_derivative(Z):
    """
    Assumes Z in the input to the tanh function (g). Returns the
    derivative of the tanh function with respect to Z

    Arguments:
    Z -- input to the tanh function, an numpy ndarray

    Returns:
    tanh'(Z) -- the derivative of the tanh function with respect to Z
    """
    a = np.tanh(Z)
    return 1 - (a ** 2)


def retrieve_weights_for_layer(l, parameters):
    """
    Returns the weights and bias for layer l

    Arguments:
    l (int) -- layer number
    parameters (dict) -- containing all the weights and biases of the
                         entire network

    Returns:
    tuple (weights, bias) -- weights and bias matrixes
    """
    return parameters['W' + str(l)], parameters['b' + str(l)]


def nonlinear_function_for_layer(l, hyperparameters):
    """
    Returns the ReLU activation function for hidden layers of the
    neural network and sigmoid activation function for the output layer
     (L).

    Arguments:
    l (int) -- layer number
    hyperparameters (dict) -- the neural network hyperparameters.

    Returns:
    g -- an activation function
    """
    activation_functions = hyperparameters['nonlinear_functions']
    g = (activation_functions['sigmoid'] if l == hyperparameters['L']
         else activation_functions['relu'])
    return g


def back_propagate(dA, A_prev, Z, W, b, l, L, m, parameters, hyperparameters):
    """
    Returns the derivative of L with respect to A[l-1] ( dA[l-1] ). 
    Also updates the weights and biases for layer l.

    Arguments:
    dA -- the derivative of the cost with respect to A[l]
    A_prev -- input to layer l
    Z -- Z value for layer l
    W -- weights for layer l
    l (int) -- layer number
    L (int) -- the last layer
    m (int) -- number of training examples
    parameters (dict) -- network weights and biases

    Returns:
    dA[l-1] -- the derivative of the loss(L) with respective to A[l-1]
    """
    derivative = sigmoid_derivative if l == L else relu_derivative
    g_prime = derivative(Z)

    dZ = dA * g_prime
    dW = 1/m * np.dot(dZ, A_prev.T)
    db = 1/m * np.sum(dZ, axis=1, keepdims=True)

    alpha = hyperparameters['lr']

    parameters['W' + str(l)] -= alpha * dW
    parameters['b' + str(l)] -= alpha * db

    if l != 1:
        return np.dot(W.T, dZ)


def optimize(parameters, hyperparameters, A_prev, Y, l):
    """
    Recursive function that implements both forward and backward 
    propagation. The base case initializes back propagation by 
    computing and returning dLdAL, the derivative of the loss with 
    respective to the neural network output. The inductive case forward
     propagate as long as the last layer hasn't been reached yet.

    Arguments:
    A -- numpy ndarray of activations from previous layer
    Y -- numpy ndarray of target labels
    parameters (dict) -- containing all the weights and biases for each 
                         layer in the network
    hyperparameters (dict) -- hyperparameters (anything that is not a 
                              weight or a bias)
    l (int) -- the current layer

    Returns:
    dA[l-1] -- the derivative of the loss with respect to A[l-1]
    """

    L = hyperparameters.get('L')
    m = np.shape(Y)[1]

    if l == L+1:
        hyperparameters["J"] = compute_cost(A_prev, Y, m)
        dA = np.divide(-Y, A_prev) + np.divide(1-Y, 1-A_prev)
        return dA
    else:
        W, b = retrieve_weights_for_layer(l, parameters)
        g = nonlinear_function_for_layer(l, hyperparameters)
        A, Z = forward_propagate(A_prev, W, b, g)
        dA = optimize(
            parameters, hyperparameters, A, Y, l+1)

    return back_propagate(dA, A_prev, Z, W, b, l, L, m, parameters, hyperparameters)


def train(A, Y, parameters, hyperparameters):
    """
    Optimizes parameters using back propagation and gradient descent.

    Arguments:
    A -- X, the input matrix
    Y -- the target labels
    parameters -- dict, of randomly initialized weights and biases for 
                  the entire neural network.
    hyperparameters -- dict, neural network hyperparameters ( anything 
                       that is not a weight or bias )

    Returns:
    parameters -- optimized weights and biases that can be used for 
                  making predictions
    costs -- list of the cost (J) for each iteration
    """
    costs = list()
    epochs = hyperparameters.get('epochs')

    for t in range(epochs):
        optimize(parameters, hyperparameters, A, Y, 1)
        if t % 100 == 0 or t == epochs-1:
            J = hyperparameters.get('J')
            costs.append(J)
    print(f"Cost after iteration {t}: {J}")

    return parameters, costs


def nn_model(A, Y, hyperparameters):
    """
    Initializes, optimizes and returns the weights of a neural network
    """
    parameters = initialize_parameters(hyperparameters.get('nn_structure'))
    parameters, costs = train(A, Y, parameters, hyperparameters)
    return parameters, costs


def predict(parameters, hyperparameters=None, A=None, Y=None, evaluation=None):
    """
    Returns the predicted probabilities, predicted classes and accuracy
     of the model.

    Arguments:
    parameters -- weights and biases of the model
    hyperparameters -- the model's hyperparameters
    A -- input to the model
    Y -- target labels
    evaluation -- bool, returns accuracy if set to True


    Returns:
    y_hat -- predicted class
    pred -- predicted probabilites
    acc -- the model's accuracy
    """
    if not hyperparameters is None:
        L = hyperparameters['L']
    else:
        L = 1
    
    w_and_b = [(parameters.get('W%d' % l), parameters.get('b%d' % l))
               for l in range(1, L+1)]
    *hidden_layers_w_and_b, output_layer_w_and_b = w_and_b

    for W, b in hidden_layers_w_and_b:
        A = relu(np.dot(W, A) + b)
    W, b = output_layer_w_and_b
    pred = sigmoid(W.dot(A) + b)

    yhat = np.where(pred > 0.5, 1, 0)
    return yhat, np.mean(yhat == Y) if evaluation else yhat
