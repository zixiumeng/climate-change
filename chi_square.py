"""CSC110 Fall 2020 Final Project: Global Warming and Coral Bleaching

Copyright and Usage Information
===============================

This file is one of the steps of project for CSC110. All forms of
distribution of this code, whether as given or with any changes, are
expressly prohibited.

This file is Copyright (c) 2020 Krystal Miao, Idris Sun, Qianning Lian, Zixiu Meng.

This is a file for chi_square modeling and calculation
"""
import doctest
from typing import List
import python_ta
from processing_data import get_severity


def chi_square(data: list, severity: int,
               frequency: int) -> list:  # bl is the number between high bleaching and low bleaching
    # frequency is the number between low frequency and high frequency
    """Return the list of the number of high frequency and low frequency

    Precondition:
    - the input should be a list that has not been processed for calculating average
    """
    lst = [0, 0, 0, 0]
    # [high bleaching and fre, high bleaching and low fre, low bleaching and high fre, low and low]
    for pair in data:
        severe = get_severity(pair[2])
        if pair[3] >= frequency and severe > severity:
            lst[0] += 1
        elif pair[3] < frequency and severe > severity:
            lst[1] += 1
        elif pair[3] >= frequency and severe <= severity:
            lst[2] += 1
        else:
            lst[3] += 1
    return lst


def cal_chi_square(square: List[int]) -> float:
    """Return the chi square calculation of the result obtained previously

    >>> cal_chi_square([0, 1, 2, 3])
    0.6
    """
    total = sum(square)
    column_1 = square[0] + square[2]
    column_2 = square[1] + square[3]
    row_1 = square[0] + square[1]
    row_2 = square[2] + square[3]
    square = (square[0] * square[3] - square[1] * square[2]) ** 2
    dividened = square * total
    divisor = column_1 * column_2 * row_1 * row_2
    return dividened / divisor


if __name__ == "__main__":
    doctest.testmod()

    python_ta.check_all(config={
        'extra-imports': [],  # the names (strs) of imported modules
        'allowed-io': [],  # the names (strs) of functions that call print/open/input
        'max-line-length': 100,
        'disable': ['R1705', 'C0200']
    })
