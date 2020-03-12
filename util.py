"""
Author: Vladyslav Ivanov
Contact Email: vladyslav.iva@gmail.com
File Description: Data manipulation tools
"""
import collections
import statistics
import datetime
import json
import csv
import os


class Utility:
    def __init__(self):
        self.response = (None, None)
        self.binary_variables = []
        self.path = os.path.dirname(os.path.realpath(__file__))
        self.configuration_path = os.path.join(self.path, 'configuration.json')

    def log(self, msg_type: str or int, msg):
        """Output log message

        Args:
            msg_type: Type of the message: INFO, ACTION, INT
            msg: Log message content
        Returns:
            None
        """
        time = datetime.datetime.now().strftime("%I:%M:%S %p")
        if isinstance(msg_type, str):
            print("[{0}] [{1}] {2}".format(time, msg_type, msg))
        else:
            if msg_type != 0:
                print("{0:>13} {1}. {2}".format("L", msg_type, msg))
            else:
                print("{0:>13} {1}".format("L", msg))

    def data_import(self, filename: str) -> {int: {str: "", str: (), str: []}}:
        """Import data, determine appropriate data types and response variable

        Args:
            filename: Relative path to the dataset

        Returns:
            df: Structured dataset
        """
        df = collections.defaultdict()

        # Extract data from the CSV file
        with open(os.path.join(self.path, filename)) as csv_file:
            dataset = csv.reader(csv_file, delimiter=",")
            for index, row in enumerate(dataset):
                # Record variable names from the header and get data from rows
                if index == 0:
                    for index, varname in enumerate(row):
                        df[index] = {"name": varname, "type": None, "data": []}
                else:
                    for index, value in enumerate(row):
                        # Attempt type conversion
                        try:
                            value = float(value)
                        except ValueError:
                            pass
                        df[index]["data"].append(value)

        for index in df.keys():
            unique_values = set(df[index]["data"])
            num_unique = len(unique_values)

            # Determine statistical data type
            if num_unique < 2:
                continue
            elif num_unique == 2:
                statistical_type = "binary"
                self.binary_variables.append((index, df[index]["name"]))
            elif num_unique < 0.1 * len(df[index]["data"]):
                statistical_type = "discrete"
            else:
                statistical_type = "continuous"
            data_type = type(list(unique_values)[0]).__name__
            df[index]["type"] = (data_type, statistical_type)

        # Handle response variable selection
        if len(self.binary_variables) == 1:
            self.response = self.binary_variables[0]
            self.log("INFO", "Response variable has been detected: {}"
                     .format(self.response[1]))
        else:
            names = [variable[1] for variable in self.binary_variables]
            self.log("ACTION", "Several binary variables have been detected. \
Please select the relevant response variable: " + ", ".join(names))
            while self.response == (None, None):
                response = input("Response variable: ")
                if response in names:
                    index = names.index(response)
                    self.response = self.binary_variables[index]
                else:
                    self.log("ERROR", "Variable '{}' does not exist. Try again"
                             .format(response))
            self.log("INFO", "Selected response variable: {}"
                     .format(self.response[1]))
        return df

    def normalization(self, df):
        """Normalize the data on range from -1 to 1

        Args:
            df: Dataset

        Returns:
            df: Normalized dataset
        """
        for index in df.keys():
            if (index, df[index]["name"]) not in self.binary_variables:
                data = df[index]["data"]
                mi = float(min(data))
                mx = float(max(data))
                df[index]["data"] = [(float(v) - mi) / (mx - mi) for v in data]
        self.log("INFO", "Data has been normalized")
        return df

    def label_encoding(self, df):
        """Encode target labels with value between 0 and number of classes - 1

        Args:
            df: Dataset

        Returns:
            df: Dataset with encoded variable labels
        """
        for key in df.keys():
            if "str" in df[key]["type"]:
                states = sorted(set(df[key]["data"]))
                for row in range(0, len(df[key]["data"])):
                    df[key]["data"][row] = states.index(df[key]["data"][row])
                data_type = "binary" if len(states) == 2 else "discrete"
                df[key]["type"] = (df[key]["type"][0], data_type)
        self.log("INFO", "Labels have been encoded")
        return df

    def correlation(self, df_x: list, df_y: list) -> float:
        """Calculate population's Pearson correlation coefficient

        Args:
            df_x: Values of the variable X
            df_y: Values of the variable Y

        Returns:
            Correlation coeffficient computed as: cov(x, y) / (std(x) * std(y))
        """
        # Calculate means
        mean_x = statistics.mean(df_x)
        mean_y = statistics.mean(df_y)

        # Calculate standard deviations
        std_x = statistics.stdev(df_x)
        std_y = statistics.stdev(df_y)

        changes = [(x - mean_x) * (y - mean_y) for x, y in zip(df_x, df_y)]
        covariance = (sum(changes) / len(df_x))
        return covariance / (std_x * std_y)

    def correlation_matrix(self, df):
        """Select features based on the correlation coefficient

        Args:
            df: Dataset

        Returns:
            correlation_matrix: Person's correlation matrix
        """
        correlation_matrix = {}
        for row in df.keys():
            for col in df.keys():
                if row != col:
                    x = df[row]
                    y = df[col]
                    # Isolate the response variable
                    if self.response[1] not in (x["name"], y["name"]):
                        correlation = self.correlation(x["data"], y["data"])
                        if -0.5 < correlation < 0.5:
                            # Filter out repeating correlations
                            if (col, row) not in correlation_matrix.keys():
                                correlation_matrix[(row, col)] = correlation
        self.log("INFO", "Correlation matrix has been computed")
        return correlation_matrix

    def permutate(self, cor_matrix):
        """Generate variable permutations using correlation matrix analysis

        Args:
            cor_matrix: Correlation matrix

        Returns:
            Permutations of the selected features
        """
        permutations = [[c[0], c[1]] for c in cor_matrix.keys()]
        for permutation in permutations:
            for correlation in cor_matrix.keys():
                # Check if there are more variables to add
                if permutation[-1] == correlation[0]:
                    correlated = True
                    for pair in permutation:
                        # Verify correlation between all variables
                        chk = [(pair, correlation[1]), (correlation[1], pair)]
                        if not any(k in cor_matrix.keys() for k in chk):
                            correlated = False
                    if correlated:
                        permutation.append(correlation[1])
        self.log("INFO", "Feature permutations have been generated")
        return [p for p in sorted(permutations, key=len) if len(p) > 2]

    def split(self, df, ratio=0.7):
        """

        Args:
            df: Dataset
            ratio: Train/test set split ratio in the decimal form

        Returns:
            df: Split dataset
        """
        ratio = ratio / (1 - ratio)
        for index in df.keys():
            df[index]["train"], df[index]["test"] = [], []
            for row in df[index]["data"]:
                num_train = len(df[index]["train"])
                num_test = len(df[index]["test"])
                set_type = "train" if num_train < num_test * ratio else "test"
                df[index][set_type].append(row)

            del df[index]["data"]
        return df

    def feature_removal(self, df, by="data_type", value=None):
        """Remove column from dataset by data or statistical type of a feature

        Args:
            df: Dataset
            by: Filter parameter (data_type or statistical_type)
            value: Value of the parameter, i.e. None, continuous, etc.

        Returns:
            df: Dataset without the removed feature
        """
        operational_df = df.copy()
        removed = False
        for index in operational_df.keys():
            if by == "data_type":
                if df[index]["type"] == value:
                    del df[index]
                    removed = True
            else:
                if value in df[index]["type"]:
                    del df[index]
                    removed = True

        if removed:
            self.log("INFO", "Invalid predictors were removed")
        return df

    def select(self, df: list, predictors: list, set_type: str):
        """Select predictors data and transform dataset into n-size vectors

        Args:
            set_type: Type of the data set: training or test
            predictors: Selected predictors

        Returns:
            dft: Dataset of the selected predictors
        """
        cdf = []
        # Get data for the selected predictors
        for predictor in df.keys():
            if predictor != self.response[0] and predictor in predictors:
                cdf.append(df[predictor][set_type])
        dft = []
        for row in range(0, len(cdf[0])):
            dft.append([cdf[col][row] for col in range(0, len(cdf))])
        return dft

    def save_configuration(self, predictors, weights, thold):
        """Save neural network model configuration as a JSON file

        Args:
            predictors: Indices of the selected predictors
            weights: Model input weights
            thold: Decision threshold

        Returns:
            None
        """
        parameters = {'vars': predictors, 'weights': weights, 'thold': thold}
        with open(self.configuration_path, 'w') as configuration:
            json.dump(parameters, configuration, ensure_ascii=False, indent=4)
        self.log("INFO", "Configuration file was successfully saved: {}"
                 .format(self.configuration_path))

    def load_configuration(self):
        """Load neural network model configuration from a JSON file

        Args:
            None
        Returns:
            parameters: Predictors, weights, and threshold of the model
        """
        with open(self.configuration_path, 'r') as configuration:
            parameters = json.load(configuration)
        self.log("INFO", "Configuration file {} was successfully loaded"
                 .format(self.configuration_path))
        return parameters
