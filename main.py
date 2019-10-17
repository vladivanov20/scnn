"""
Date: October 15th, 2019
Author: Vladyslav Ivanov <vladyslav.iva@gmail.com>
Description: Neural Network implementation in pure Python
"""
import csv
import math
import random


class NeuralNetwork:
    def __init__(self, num_predictors):
        self.weights = [random.uniform(-1, 1) for _ in range(num_predictors)]
        self.percent = 0.5
        self.epochs = 30000

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

    def train(self, inputs, outputs):
        previous_weights = []
        for epoch in range(self.epochs):
            if epoch % 10 == 0:
                # Check for convergence
                if self.weights in previous_weights:
                    break
                else:
                    previous_weights = []
            previous_weights.append(self.weights)
            dendrites = self.sigmoid(self.dot(self.weights, inputs))
            residuals = self.difference(outputs, dendrites)
            weight_adjustments = self.multiply(
                self.sigmoid_derivative(dendrites), residuals
            )
            self.weights = self.add(
                self.weights, self.dot(weight_adjustments, self.transpose(inputs))
            )

    def normalize(self, predictors):
        num_rows = len(predictors)
        num_cols = len(predictors[0])
        normalized_predictors, min_values, max_values = [], [], []
        averages = [
            sum([row[index] for row in predictors]) / num_rows
            for index in range(num_cols)
        ]
        for i in range(num_cols):
            min_values.append(min(predictors, key=lambda x: x[i])[i])
            max_values.append(max(predictors, key=lambda x: x[i])[i])
        for row in predictors:
            temp = []
            for i in range(num_cols):
                temp.append((row[i] - min_values[i]) / (max_values[i] - min_values[i]))
            normalized_predictors.append(temp)
        return normalized_predictors

    def predict(self, inputs):
        return self.sigmoid(self.dot(self.weights, inputs))

    def split_data(self, predictors, response):
        cutoff_index = round(len(predictors) * self.percent)
        return (
            (predictors[0:cutoff_index], response[0:cutoff_index]),
            (
                predictors[cutoff_index : len(predictors)],
                response[cutoff_index : len(predictors)],
            ),
        )

    def decision_function(self, results):
        thresresults = []
        for threshold in range(0, 1001, 1):
            prediction_scores = [1 if prediction > (threshold / 100) else 0 for prediction in results]
            difference = [1 if predicted == actual else 0 for predicted, actual in zip(prediction_scores, y_test)]
            thresresults.append((threshold / 100, difference.count(1) / len(difference)))
        print(max(thresresults, key=lambda x: x[1]))


if __name__ == "__main__":

    # Select top 5 features
    feature_selection = [
        "area_mean",
        "radius_se",
        "texture_worst",
        "compactness_worst",
        "smoothness_worst",
        "diagnosis",
    ]
    predictors, response = [], []
    response_variable = "diagnosis"
    positive_response = "M"
    with open("breastcancer.csv") as csvfile:
        current_file = csv.reader(csvfile, delimiter=",")
        for i, row in enumerate(current_file):
            if i == 0:
                index_list = [
                    (j, col) for j, col in enumerate(row) if col in feature_selection
                ]
            else:
                row_predictors, row_response = [], []
                for (index, variable) in index_list:
                    if variable != response_variable:
                        row_predictors.append(float(row[index]))
                    else:
                        row_response.append(1 if row[index] == positive_response else 0)
                predictors.append(row_predictors)
                response.extend(row_response)

    model = NeuralNetwork(len(predictors[0]))

    # Split Dataset into Test and Training Data
    ((x_train, y_train), (x_test, y_test)) = model.split_data(predictors, response)
    
    # Normalize the data
    x_train = model.normalize(x_train)
    x_test = model.normalize(x_test)
    
    # Train the neural network
    model.train(x_train, y_train)
    
    # Prediction for test dataset
    results = model.predict(x_test)
    
    # Determine optimal decision threshold
    model.decision_function(results)

