"""
Author: Vladyslav Ivanov
Contact Email: vladyslav.iva@gmail.com
File Description: Feed-forward Neural Network
"""
from util import Utility
import random
import math


class NeuralNetwork:
    def __init__(self, response: (int, str)):
        self.util = Utility()
        self.epochs, self.weights = None, None
        self.response = response
        self.util.log("INFO", "Neural Network has been initialized")

    def sigmoid(self, m0: list) -> list:
        """Estimate sigmoid activation function given by 1 / 1 + e^{-x}

        Args:
            m0: Matrix

        Returns:
            Matrix of sigmoid function values
        """
        return [1 / (1 + math.pow(math.e, -x)) for x in m0]

    def sigmoid_derivative(self, m0: list) -> list:
        """Estimate derivative of the sigmoid function given by x - x^2

        Args:
            matrix: Matrix

        Returns:
            Matrix of derivative of the sigmoid function values
        """
        return [x * (1 - x) for x in m0]

    def dot(self, v0: list, m0: list) -> list:
        """Calculate dot product of the vector and a matrix

        Args:
            v0: Vector
            m0: Matrix

        Returns:
            Dot product of the vector and a matrix
        """
        return [sum([x * y for x, y in zip(v0, v1)]) for v1 in m0]

    def transpose(self, m0: list) -> list:
        """Transpose the matrix

        Args:
            m0: Matrix

        Returns:
            Transposed matrix
        """
        return list(zip(*m0))

    def product(self, m0: list, m1: list) -> list:
        """Calculate product of the two matrices

        Args:
            m0, m1: Matrix

        Returns:
            Product of the two matrices
        """
        return [x * y for x, y in zip(m0, m1)]

    def difference(self, m0: list, m1: list) -> list:
        """Calculate difference of the two matrices

        Args:
            m0, m1: Matrix

        Returns:
            Difference of the two matrices
        """
        return [x - y for x, y in zip(m0, m1)]

    def add(self, m0: list, m1: list) -> list:
        """Calculate sum of the two matrices

        Args:
            m0, m1: Matrix

        Returns:
            Sum of the two matrices
        """
        return [x + y for x, y in zip(m0, m1)]

    def train(self, inputs: list, outputs: list):
        """Train the neural network using backpropagation algorithm

        Args:
            inputs: Predictors' input values
            outputs: Response variable values

        Returns:
            None
        """
        last_weights = []
        for epoch in range(0, self.epochs):
            # Check for convergence every 10 epochs
            if epoch % 10 == 0:
                if self.weights in last_weights:
                    break
                else:
                    last_weights = []
            last_weights.append(self.weights)

            # Estimate the predicted values
            y_hat = self.sigmoid(self.dot(self.weights, inputs))

            # Calculate the difference between predicted and true values
            residuals = self.difference(outputs, y_hat)

            # Estimate the derivative of the predicted values
            y_hat_prime = self.sigmoid_derivative(y_hat)

            # Backward propagation of errors
            deltas = self.product(y_hat_prime, residuals)
            weight_increments = self.dot(deltas, self.transpose(inputs))
            self.weights = self.add(self.weights, weight_increments)

    def predict(self, inputs: list):
        """Generate output predictions for the input samples

        Args:
            inputs: Predictors' input values

        Returns:
            Predicted probabilities between 0.0 and 1.0
        """
        return self.sigmoid(self.dot(self.weights, inputs))

    def metrics(self, predicted: list, true: list) -> (float, float):
        """Estimate F1 score and model accuracy

        Args:
            pred: Predictions
            true: True output values

        Returns:
            (f1, accuracy): F1 score and accuracy
        """

        # Generate confusion matrix
        confusion_matrix = [[0, 0], [0, 0]]
        for i in range(0, len(true)):
            confusion_matrix[int(true[i])][int(predicted[i])] += 1
        tn = confusion_matrix[0][0]
        fp = confusion_matrix[0][1]
        fn = confusion_matrix[1][0]
        tp = confusion_matrix[1][1]

        f1 = 0
        if (tp + fn) * (tp + fp) != 0:
            precision = tp / float(tp + fp)
            recall = tp / float(tp + fn)

            # Calculate F1 score if the denominator is not equal to zero
            if precision + recall != 0:
                f1 = (2 * precision * recall) / (precision + recall)
        accuracy = (tp + tn) / float(tp + tn + fn + fp)
        return (f1, accuracy)

    def run(self, df: {}, predictors: list, threshold, y_te: list, y_tr=None):
        """Train, test, and evaluate the neural network

        Args:
            df: Dataset
            predictors: Set of selected predictors
            threshold: Decision threshold
            y_te: Outputs of the test dataset
            y_tr: Outputs of the training dataset

        Returns:
            F1 score and accuracy performance metrics
        """
        self.weights = [random.uniform(-1, 1) for _ in range(len(predictors))]
        if y_tr:
            x_tr = self.util.select(df, predictors, "train")
            x_te = self.util.select(df, predictors, "test")

            # Train neural network using training set
            self.train(x_tr, y_tr)
            y_hats_tr = self.predict(x_tr)

            # Convert probabilities to the binary
            predicted_t = [1 if y > threshold else 0 for y in y_hats_tr]
        else:
            x_te = self.util.select(df, predictors, "data")

        y_hats_te = self.predict(x_te)
        predicted = [1 if y > threshold else 0 for y in y_hats_te]
        test_performance = self.metrics(predicted, y_te)
        if y_tr:
            return (self.metrics(predicted_t, y_tr), test_performance)
        else:
            return test_performance

    def select(self, df: {int: {str: "", str: (), str: []}}, permutations: []):
        """Select best performing model by maximizing accuracy and the F1 score

        Args:
            df: Dataset
            permutations: Possible combinations of predictors to test

        Returns:
            None
        """
        self.util.log("INFO", "Running model selection...")
        # Get relative accuracy by training all models on 30 epochs
        self.epochs = 30
        models = []
        y_te = df[self.response[0]]["test"]
        y_tr = df[self.response[0]]["train"]
        for predictors in permutations:
            trials = []
            # Bruteforce different thresholds to find the highest accuracy
            for threshold in range(1, 10):
                threshold = float(threshold / 10)
                performance = self.run(df, predictors, threshold, y_te, y_tr)
                trials.append([performance, threshold])
            # Maximize training and test accuracy
            hp = max(trials, key=lambda x: x[0][0][1] + x[0][1][1])
            models.append([predictors, hp])

        # Select best performing models for training and evaluation
        num_models = len(models)
        sorted_models = sorted(models, key=lambda x: x[1][0][0] + x[1][0][1])
        top_models = sorted_models[num_models - math.ceil(num_models * 0.02):]

        # Display statistics
        num_top = len(top_models)
        msg = "Top {} best performing model(s) were selected:".format(num_top)
        self.util.log("INFO", msg)

        for i, model in enumerate(reversed(top_models)):
            varnames = ', '.join([df[p]['name'] for p in model[0]])
            (f1_tr, accuracy_tr) = model[1][0][0]
            (f1_te, accuracy_te) = model[1][0][1]
            threshold = model[1][1]
            msg = "Predictors: {:145s} | Threshold: {:.0%} | F1-score (Train):\
 {:.2f} | Accuracy (Train): {:.2%} | F1-score (Test): {:.2f} | Accuracy \
(Test): {:.2%}"
            self.util.log(i + 1, msg.format(varnames, threshold, f1_tr,
                                            accuracy_tr, f1_te, accuracy_te))

        # Run extensive training and evaluation on the best performing models
        self.epochs = 30000
        validation = []
        for model in top_models:
            predictors = model[0]
            threshold = model[1][1]
            performance = self.run(df, predictors, threshold, y_te, y_tr)
            validation.append([predictors, performance, threshold])
        selected_model = max(validation, key=lambda x: x[1][0] + x[1][1])
        self.util.log("INFO", "Best performing model has been identified:")
        (f1_tr, accuracy_tr) = selected_model[1][0]
        (f1_te, accuracy_te) = selected_model[1][1]
        threshold = selected_model[2]
        predictors = selected_model[0]
        varnames = ', '.join([df[p]['name'] for p in predictors])
        self.util.log(0, msg.format(varnames, threshold, f1_tr, accuracy_tr,
                                    f1_te, accuracy_te))

        # Export model configuration
        self.util.log("ACTION", "Export model configuration? [Y/N]")
        if input().lower() == "y":
            self.util.save_configuration(predictors, self.weights, threshold)
