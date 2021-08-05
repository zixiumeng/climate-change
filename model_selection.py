"""CSC110 Fall 2020 Final Project: Global Warming and Coral Bleaching

Copyright and Usage Information
===============================

This file is one of the steps of project for CSC110. All forms of
distribution of this code, whether as given or with any changes, are
expressly prohibited.

This file is Copyright (c) 2020 Krystal Miao, Idris Sun, Qianning Lian, Zixiu Meng.

This is file storing python for model selection
"""
import doctest
import statistics
from typing import List, Tuple
import math
import python_ta
from ploting import get_xy_data, get_frequency_x_without_repeat, \
    get_frequency_y_coords, frequency_line_coefficient


def model_selection(filename: str, power_x: List[int], power_y: List[int]) -> Tuple[int, int]:
    """Return the best model that
     fits the data best with the try-outs of powers in power_x and power_y

    Precondition:
    - power_x and power_y starts with 1 and get to at most 2"""
    greatest_r = 0
    best_model = (1, 1)
    for x in power_x:
        for y in power_y:
            if math.fabs(r_value(filename, x, y)) > greatest_r:
                greatest_r = math.fabs(r_value(filename, x, y))
                best_model = (x, y)
    return best_model


def r_value(filename: str, power_x: int, power_y: int) -> float:
    """Return r value for the linear regression
     which represents how well the line represents the module"""
    m = frequency_line_coefficient(filename, power_x, power_y)[0]
    p = frequency_line_coefficient(filename, power_x, power_y)[1]
    data = get_xy_data(filename)
    x_coords = get_frequency_x_without_repeat(data, power_x)
    y_coords = get_frequency_y_coords(data, x_coords, power_y)
    y_estimate = [m * x + p for x in x_coords]
    y_average = statistics.mean(y_coords)
    total_sum_of_square = sum([(y_coords[i] - y_average) ** 2
                               for i in range(0, len(y_coords))])
    residual_sum_of_square = sum((y_coords[i] - y_estimate[i]) ** 2
                                 for i in range(0, len(y_coords)))
    return 1 - residual_sum_of_square / total_sum_of_square


if __name__ == "__main__":
    doctest.testmod()

    python_ta.check_all(config={
        'extra-imports': ['sklearn.impute', 'numpy', 'csv', 'matplotlib.pyplot', 'math',
                          'statistics', 'model_selection', 'ploting', 'processing_data',
                          'sklearn.preprocessing'],
        'allowed-io': ['load_csv_file'],
        'max-line-length': 100,
        'disable': ['R1705', 'C0200']
    })
