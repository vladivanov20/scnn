"""
Author: Vladyslav Ivanov
Contact Email: vladyslav.iva@gmail.com
File Description: Self-configuring Neural Network
"""
from nnet import NeuralNetwork
from util import Utility
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Self-configuring Neural Network"
    )

    parser.add_argument(
        "-f",
        "--file",
        required=True,
        help='Dataset name, e.g. "breast-cancer-wisconsin.csv"'
    )

    parser.add_argument(
        "-p",
        "--percent",
        default=0.7,
        required=False,
        help="Train/test split ratio in the decimal form, e.g. 0.7",
    )

    args = parser.parse_args()
    util = Utility()

    # Display input parameters
    print('{0} {1:^2} {0}'.format('#' * 25, 'Input parameters'))
    print("Filename: " + args.file.split('/')[-1])
    print("Cross-validation split: {0:.0%} training and {1:.0%} test data"
          .format(args.percent, 1 - args.percent))
    print('{0} {1:^2} #{0}'.format('#' * 31, 'Log'))

    # Import and structure the dataset
    structured_data = util.data_import(args.file)

    # Remove variables with the no data type
    data = util.feature_removal(structured_data, "data_type", None)

    # Normalize data
    normalized_data = util.normalization(data)

    # Encode categorical variables
    data = util.label_encoding(normalized_data)

    # Compute Pearson's correlation matrix
    correlations = util.correlation_matrix(data)

    # Generate feature permutations using correlation matrix analysis
    feature_permutations = util.permutate(correlations)

    # Initial the model
    model = NeuralNetwork(util.response)

    # Decide whether to use the pre-trained model
    util.log("ACTION", "Import model configuration? [Y/N]")

    if input().lower() == "n":
        # Split the data into train and test sets
        splitted_data = util.split(data, args.percent)

        # Determine best performing model
        model.select(splitted_data, feature_permutations)
    else:
        # Import configuration
        parameters = util.load_configuration()
        predictors = parameters['vars']
        threshold = parameters['thold']
        model.weights = parameters['weights']
        model.epochs = 30000
        y = data[model.response[0]]["data"]

        # Display model's performance
        util.log("INFO", "Testing the model with the imported parameters...")
        (f1, accuracy) = model.run(data, predictors, threshold, y)
        varnames = ', '.join([data[p]['name'] for p in predictors])
        msg = "Predictors: {:145s} | Threshold: {:.0%} | F1-score: {:.2f}\
| Accuracy: {:.2%}"
        util.log(0, msg.format(varnames, threshold, f1, accuracy))
