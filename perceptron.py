import numpy as np


class Perceptron(object):
    """
        A class used to represent a perceptron

        ...

        Attributes
        ----------
        no_of_inputs : int
            used to determine how many weights we need to learn
        threshold : int
            the number of epochs we’ll allow our learning algorithm to iterate through before ending
        learning_rate : float
            determines the magnitude of change for our weights during each step through our training data
        self.weights: float
            the amount of influence the input has over the output.

        Methods
        -------
        predict(inputs)
            houses our perceptron algorithm method

        train(training_inputs, targets)
            Prints the animals name and what sound it makes
        """

    def __init__(self, no_of_inputs, threshold=1, learning_rate=0.02):
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.weights = np.zeros(no_of_inputs + 1)

    def predict(self, inputs):
        """Houses our perceptron algorithm.

        The predict method takes one argument, inputs, which it expects
        to be an numpy array/vector of a dimension equal to the no_of_inputs
        parameter that the perceptron was initialized with

        Parameters
        ----------
        inputs : int
            It implements the numpy dot product function and bias in a n attempt to get the target

        self.weights : float
            We adjust the weight vector

        Returns
        ------
        activation : int
            It determines whether the perceptron is turned on or off.\ based on the summation value
        """
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        if summation > 0:
            activation = 1
        else:
            activation = 0
        return activation

    '''Training the neural network'''

    def train(self, training_inputs, targets):
        """Houses our perceptron algorithm.

        Training the neural network

        Parameters
        ----------
        targets : float
            A numpy array of expected output values for each of the corresponding inputs



        Returns
        ------
        self.weights[1:] : float
            A list of the new weights
        """
        for _ in range(self.threshold):
            for inputs, target in zip(training_inputs, targets):
                prediction = self.predict(inputs)
                # Obtaining the new weight
                self.weights[1:] += self.learning_rate * (target - prediction) * inputs
                self.weights[0] += self.learning_rate * (target - prediction)

                # Printing the new weights to the console
                print(self.weights[1:])
