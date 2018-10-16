import numpy as np
from perceptron import Perceptron

# A numpy array of the inputs
training_inputs = [np.array([0.3, 0.8]), np.array([-0.3, 1.6]), np.array([0.9, 0])]

# A numpy array of the expected outputs
targets = np.array([0.7, -0.1, 1.3])

# Training the perceptron
perceptron = Perceptron(2)
perceptron.train(training_inputs, targets)

# Printing the activation result in the console based on the input
inputs = np.array([0.3, 0.8])
print(perceptron.predict(inputs))


inputs = np.array([-0.3, 1.6])
print(perceptron.predict(inputs))

inputs = np.array([0.9, 0])
print(perceptron.predict(inputs))


