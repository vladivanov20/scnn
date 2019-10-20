"""
Date: October 15th, 2019
Author: Vladyslav Ivanov <vladyslav.iva@gmail.com>
Description: Neural Network implementation in pure Python
"""
import csv
import math
import random
import argparse

class NeuralNetwork:
    def __init__(self, num_predictors):
        self.weights = [random.uniform(-1, 1) for _ in range(num_predictors)]
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

    def predict(self, inputs):
        return self.sigmoid(self.dot(self.weights, inputs))

    def decision_function(self, results):
        thresresults = []
        for threshold in range(0, 1001, 1):
            prediction_scores = [
                1 if prediction > (threshold / 100) else 0 for prediction in results
            ]
            difference = [
                1 if predicted == actual else 0
                for predicted, actual in zip(prediction_scores, y_test)
            ]
            thresresults.append(
                (threshold / 100, difference.count(1) / len(difference))
            )
        print(max(thresresults, key=lambda x: x[1]))


class Data_Tools:
    def __init__(self):
        pass

    def read_csv(self, filename, predictors, response, scheme):
        x, y = [], []
        with open(filename) as csvfile:
            current_file = csv.reader(csvfile, delimiter=",")
            for i, row in enumerate(current_file):
                if i == 0:
                    predictors.append(response)
                    indexes = [(j, col) for j, col in enumerate(row) if col in predictors]
                else:
                    x_row, y_row = [], []
                    for (index, variable) in indexes:
                        if variable != response:
                            x_row.append(float(row[index]))
                        else:
                            y_row.append(1.0 if row[index] == scheme else 0.0)
                    x.append(x_row)
                    y.extend(y_row)
        return x, y

    def split_data(self, x, y, percent):
        nrows = len(x)
        split = round(nrows * percent)
        return ((x[0:split], y[0:split]), (x[split:nrows], y[split:nrows]))

    def normalize(self, df):
        num_cols = len(df[0])
        normalized_df = []
        
        min_values = [min(df, key=lambda x: x[i])[i] for i in range(num_cols)]
        max_values = [max(df, key=lambda x: x[i])[i] for i in range(num_cols)]
            
        for row in df:
            temp = []
            for i in range(num_cols):
                temp.append((row[i] - min_values[i]) / (max_values[i] - min_values[i]))
            normalized_df.append(temp)
        return normalized_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Neural Network Auto Optimizer')

    parser.add_argument(
        '-f',
        '--file',
        default=None,
        help='Filename in a string format. E.g. "breastcancer.csv"'
    )

    parser.add_argument(
        '-p',
        '--predictors',
        default=None,
        help='List of predictors in a string format separated by comma. \
        E.g. "area_mean, radius_se, texture_worst, compactness_worst, \
        smoothness_worst"'
    )

    parser.add_argument(
        '-r',
        '--response',
        default=None,
        help='Name of the response variable in a string format. E.g. \
            "diagnosis"'
    )

    parser.add_argument(
        '-s',
        '--scheme',
        default=1,
        help='Naming scheme for the positive response. E.g. "M"'
    )

    parser.add_argument(
        '-c',
        '--percent',
        default=0.5,
        help='Percentage cross validation split (train/test ratio). E.g. 0.5'
    )

    config = parser.parse_args()
    tools = Data_Tools()
    
    # --file "breastcancer.csv" -s "M" -r "diagnosis" -p 
    predictors = config.predictors.replace(' ', '').split(',')
    print("######################## CONFIGURATION ##########################")
    print("File name: " + config.file)
    print("Selected predictors: " + ', '.join(predictors))
    print("Selected response variable: " + str(config.response))
    print("Positive response scheme: " + str(config.scheme))
    print("Cross validation split: " + str(config.percent))
    print("#################################################################")
    
    x, y = tools.read_csv(config.file, predictors, config.response, config.scheme)
    
    # Split Dataset into Test and Training Data
    ((x_train, y_train), (x_test, y_test)) = tools.split_data(x, y, config.percent)
    
    # Normalize the data
    x_train = tools.normalize(x_train)
    x_test = tools.normalize(x_test)
    
    # Initialize the model
    model = NeuralNetwork(len(x[0]))
    
    # Train the neural network
    model.train(x_train, y_train)

    # Prediction for test dataset
    results = model.predict(x_test)

    # Determine an optimal decision threshold
    model.decision_function(results)
