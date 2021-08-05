"""CSC110 Fall 2020 Final Project: Global Warming and Coral Bleaching

Copyright and Usage Information
===============================

This file is one of the steps of project for CSC110. All forms of
distribution of this code, whether as given or with any changes, are
expressly prohibited.

This file is Copyright (c) 2020 Krystal Miao, Idris Sun, Qianning Lian, Zixiu Meng.

This is a file for functions that is designed to plot the points and find the lines
"""
import doctest
from typing import List, Dict
import statistics
import numpy as np
import python_ta
import processing_data


class PowerOutOfRange(Exception):
    """Raise when the input of power is not in the expected range"""


def frequency_line_coefficient(filename: str, power_x: int, power_y: int) -> list:
    """Return the coefficienct of the linear regression """
    data = get_xy_data(filename)
    x_coords = get_frequency_x_without_repeat(data, power_x)
    y_coords = get_frequency_y_coords(data, x_coords, power_y)
    co = np.polyfit(x_coords, y_coords, 1)
    return list(co)


def get_xy_data(filename: str) -> List[Dict[str, list]]:  # [{'':[[],[]...], '':[[]]}, {}]
    """Return data from processing_data.get_processsed_data"""
    return processing_data.get_processed_data(filename)


def get_frequency_x_without_repeat(data: list, power: int) -> list:
    """Return x_coords which is TSA frequency from data without repeated value

    Precondition:
    - power should be 1 or 2
    """
    x_coords = []
    big_list = data[0]
    for pair in big_list:  # number of ecoregion
        if power == 1:
            x_coords.append(pair[1])
        if power == 2:
            x_coords.append(pair[3])
    return list(set(x_coords))


def get_average_y_coords(data: list, power: int, num: float) -> float:
    """Return the average value of y if they have the same x value"""
    acc = []
    for pair in data:
        if num in [pair[1], pair[3]]:
            if power == 1:
                acc.append(pair[2])
            elif power == 2:
                acc.append(pair[5])
    return statistics.mean(acc)


def get_frequency_y_coords(data: list, x_coords: list, power: int) -> list:
    """Return y_coords which is the percentage of bleaching from data"""
    return [get_average_y_coords(data[0], power, x) for x in x_coords]


def get_dhw_x_without_repeat(data: list, power: int) -> list:
    """Return x_coords which is TSA frequency from data without repeated value

    Precondition:
    - power should be 1 or 2
    """
    x_coords = []
    big_list = data[1]
    for pair in big_list:  # number of ecoregion
        if power == 1:
            x_coords.append(pair[1])
        if power == 2:
            x_coords.append(pair[3])
    return list(set(x_coords))


def get_dhw_y_coords(data: list, x_coords: list, power: int) -> list:
    """Return y_coords which is the percentage of bleaching from data of dhw"""
    return [get_average_y_coords(data[1], power, x) for x in x_coords]


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
