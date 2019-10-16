"""
Date: October 15th, 2019
Author: Vladyslav Ivanov <vladyslav.iva@gmail.com>
Description: Neural Network implementation in pure Python
"""
import math
import random


class NeuralNetwork():

    def __init__(self, num_predictors):
        self.weights = [random.uniform(-1, 1) for _ in range(num_predictors)]

    def sigmoid(self, value_array):
        return [1 / (1 + math.pow(math.e, -x)) for x in value_array]

    def sigmoid_derivative(self, xarray):
        return [x * (1 - x) for x in xarray]

    def dot(self, m0, m1):
        return [sum([x * y for x, y in zip(m0, vec)]) for vec in m1]

    def multiply(self, m0, m1):
        return [x * y for x, y in zip(m0, m1)]

    def difference(self, m0, m1):
        return [x - y for x, y in zip(m0, m1)]

    def add(self, m0, m1):
        return [x + y for x, y in zip(m0, m1)]

    def transpose(self, xarray):
        return list(zip(*xarray))

    def train(self, inputs, outputs, epochs):
        for _ in range(epochs):
            dendrites = self.sigmoid(self.dot(self.weights, inputs))
            residuals = self.difference(outputs, dendrites)
            weight_adjustments = self.multiply(self.sigmoid_derivative(dendrites), residuals)
            self.weights = self.add(self.weights, self.dot(weight_adjustments, self.transpose(inputs)))

    def predict(self, inputs):
        return self.sigmoid(self.dot(self.weights, inputs))


if __name__ == "__main__":
    # Training Data
    x_train = [[0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]]
    y_train = [0, 0, 0, 1, 1]

    # Test Data
    x_test = [[0, 1, 0, 0]]
    epochs = 1000

    model = NeuralNetwork(len(x_train[0]))
    model.train(x_train, y_train, epochs)
    result = model.predict(x_test)
