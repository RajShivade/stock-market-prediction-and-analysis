import numpy as np
import math


def scale_range(x, input_range, target_range):
    """

    Rescale a numpy array from input to target range
    :param x: data to scale
    :param input_range: optional input range for data: default 0.0:1.0
    :param target_range: optional target range for data: default 0.0:1.0
    :return: rescaled array, incoming range [min,max]
    """

    range = [np.amin(x), np.amax(x)]
    x_std = (x - input_range[0]) / (1.0*(input_range[1] - input_range[0]))
    x_scaled = x_std * (1.0*(target_range[1] - target_range[0])) + target_range[0]
    return x_scaled, range

def train_test_split_lstm(stocks, prediction_time=1, test_data_size=50, unroll_length=5):
    """
    Split the data set into training and testing feature for Long Short Term Memory Model
    :param stocks: whole data set containing ['Open','Close','Volume'] features
    :param prediction_time: no of days
    :param test_data_size: size of test data to be used
    :param unroll_length: how long a window should be used for train test split
    :return: X_train : training sets of feature
    :return: X_test : test sets of feature
    :return: y_train: training sets of label
    :return: y_test: test sets of label
    """
    # Training data
    test_data_cut = test_data_size + unroll_length + prediction_time

    x_train = stocks.iloc[:-test_data_cut].values
    y_train = stocks.iloc[prediction_time:-test_data_cut]['Close'].values

    # Test data
    x_test = stocks.iloc[-test_data_cut:-prediction_time].values
    y_test = stocks.iloc[prediction_time - test_data_cut:]['Close'].values

    return x_train, x_test, y_train, y_test


def train_test_split_lstm(stocks, prediction_time=1, test_data_size=450, unroll_length=50):
    """
    Split the data set into training and testing feature for Long Short Term Memory Model
    :param stocks: whole data set containing ['Open','Close','Volume'] features
    :param prediction_time: no of days
    :param test_data_size: size of test data to be used
    :param unroll_length: how long a window should be used for train test split
    :return: X_train : training sets of feature
    :return: X_test : test sets of feature
    :return: y_train: training sets of label
    :return: y_test: test sets of label
    """
    # Training data
    test_data_cut = test_data_size + unroll_length + prediction_time

    x_train = stocks.iloc[:-test_data_cut].values
    y_train = stocks.iloc[prediction_time:-test_data_cut]['Close'].values

    # Test data
    x_test = stocks.iloc[-test_data_cut:-prediction_time].values
    y_test = stocks.iloc[prediction_time - test_data_cut:]['Close'].values

    return x_train, x_test, y_train, y_test



def unroll(data, sequence_length=24):
    """
    use different windows for testing and training to stop from leak of information in the data
    :param data: data set to be used for unrolling
    :param sequence_length: window length
    :return: data sets with different window.
    """
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
    return np.asarray(result)


