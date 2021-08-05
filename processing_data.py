"""CSC110 Fall 2020 Final Project: Global Warming and Coral Bleaching

Copyright and Usage Information
===============================

This file is one of the steps of project for CSC110. All forms of
distribution of this code, whether as given or with any changes, are
expressly prohibited.

This file is Copyright (c) 2020 Krystal Miao, Idris Sun, Qianning Lian, Zixiu Meng.

This is the processing done before our manipulation
 on the data so that we can gain easier access to build our model
"""
import csv
from typing import List, Tuple, Any
import math
import doctest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures
import python_ta


class NotSimplifiedError(Exception):
    """Raised when the input does not fit the simplified precondition"""


def load_csv_file(filename: str) -> Tuple[List[str], List[str], List[List[str]]]:
    """Return a list of list of strings that contains all the information
    in the given filename(without the tiles and unit)
    """
    with open(filename) as file:
        reader = csv.reader(file)
        header = next(reader)
        unit = next(reader)
        data = list(reader)

    return (header, unit, data)


def search_location(header: List[str], key_word: str) -> Any:
    """Return the exact location of key_word in header

    Precondition:
    - There are no replicated key words in header
    """
    for i in range(len(header)):
        if header[i] == key_word:
            return i
    return None


def simplify_data(header: List[str], data: List[List[str]]) -> List[List[str]]:
    """Return a simplified version of data with only the following items
    Ecoregion
    Date2
    Average_Bleaching
    TSA_Frequency
    TSA_DHW
    """
    loc_ecoregion = search_location(header, 'Ecoregion')
    loc_date = search_location(header, 'Date2')
    loc_bleaching = search_location(header, 'Average_Bleaching')
    loc_frequency = search_location(header, 'TSA_Frequency')
    loc_dhw = search_location(header, 'TSA_DHW')

    simplified_list = []
    for pairs in data:
        new_data = [pairs[loc_ecoregion], int(pairs[loc_date]), float(pairs[loc_bleaching]),
                    float(pairs[loc_frequency]), float(pairs[loc_dhw])]
        simplified_list.append(new_data)
    return simplified_list


def count_data(data: list, region: str) -> int:
    """Use the simplified version of the data and return the number of data of
    one specific ecoregion

    Precondition:
    - The input should be data that has been processed by simplify_data
    """
    if len(data[0]) == 5:
        count = 0
        for pairs in data:
            if pairs[0] == region:
                count = count + 1
        return count
    else:
        raise NotSimplifiedError


def filter_data(data: list) -> list:
    """Filter the data that meets our conditions:

    Condition1: the corresponding ecoregion has at least 400 pairs of data
    Condition2: we only accept data that is after 2009.1.1
    Condition3: Bleaching is already happening.

    Precondition:
    - the data input should be of simplified version
    """
    if len(data[0]) == 5:
        regions = set(info[0] for info in data)
        good_region = {place for place in regions if count_data(data, place) >= 400}
        filtered_data =\
            [info for info in data if info[0] in good_region
             and info[1] >= 20090101]
        return filtered_data
    else:
        raise NotSimplifiedError


def polynomial_usage(data: list) -> list:
    """Here we import polynomial features to try out different powers of our variables,
    making it easier for future modeling"""
    poly = PolynomialFeatures(2)
    return poly.fit_transform(data).tolist()


def simpleimputer_usage(data_nan: list, data_complete: list) -> list:
    """Here we import simpleimputer features to impute values that is set as nan before
    we process them in the section of polynomial usage

    Precondition:
    - The input of data_nan expects a 2D-array"""
    imp = SimpleImputer(missing_values=math.nan, strategy='mean')
    imp.fit(data_complete)
    return imp.transform(data_nan)


def get_severity(bleaching: float) -> int:
    """"Return the severity of coral bleaching in the way discussed in the
    computational plan

    Precondition:
    - the input should be specific piece of information that has been processed
    """
    if bleaching == 0:
        return 0
    elif 0 < bleaching <= 10:
        return 1
    elif 10 < bleaching <= 50:
        return 2
    else:
        return 3


def apply_imputer(data: list) -> list:
    """Applying simple imputer to a specific data list that is 2D array and return a
    list of data without nan"""
    data_nan = []
    data_complete = []
    for info in data:
        if any([math.isnan(elements) for elements in info]):
            data_nan.append(info)
        else:
            data_complete.append(info)
    implemented_data = simpleimputer_usage(data_nan, data_complete)
    data_complete.extend(implemented_data)
    return data_complete


def get_processed_data(filename: str) -> list:
    """Return a summary of all the work done previously processing"""
    full_info = load_csv_file(filename)
    header, data = full_info[0], full_info[2]
    final_data = filter_data(simplify_data(header, data))
    frequency = apply_imputer([[info[3], info[2]] for info in final_data])
    dhw = apply_imputer([[info[4], info[2]] for info in final_data])
    data_frequency = polynomial_usage(frequency)
    data_dhw = polynomial_usage(dhw)
    return [data_frequency, data_dhw]


def get_unit(filename: str, item: str) -> Any:
    """Return the corresponding unit of of a specific datatype"""
    header, unit = load_csv_file(filename)[0], load_csv_file(filename)[1]
    for i in range(len(header)):
        if item == header[i]:
            return unit[i]
    return None


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
