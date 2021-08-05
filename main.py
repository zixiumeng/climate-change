"""CSC110 Fall 2020 Final Project: Global Warming and Coral Bleaching

Copyright and Usage Information
===============================

This file is one of the steps of project for CSC110. All forms of
distribution of this code, whether as given or with any changes, are
expressly prohibited.

This file is Copyright (c) 2020 Krystal Miao, Idris Sun, Qianning Lian, Zixiu Meng.

This is the main project for manipulating data in the dataset and visualizing our data
"""
import doctest
import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import python_ta
import chi_square
from ploting import get_xy_data, get_frequency_x_without_repeat, get_frequency_y_coords,\
    get_dhw_y_coords, get_dhw_x_without_repeat
import model_selection
import processing_data


def plot_frequency_points(filename: str, power_x: int, power_y: int) -> None:
    """Plot the frequency and average_bleaching using plotly. Display results in a web browser.
    """
    fig = go.Figure()
    sim_data = get_xy_data(filename)
    x_coords = get_frequency_x_without_repeat(sim_data, power_x)
    y_coords = get_frequency_y_coords(sim_data, x_coords, power_y)

    fig.add_trace(go.Scatter(x=x_coords, y=y_coords, mode='markers', name='Data'))

    fig.show()


def plot_dhw_points(filename: str, power_x: int, power_y: int) -> None:
    """Plot the given x-coordinates using plotly. Display results in a web browser"""
    fig = go.Figure()
    sim_data = get_xy_data(filename)
    x_coords = get_dhw_x_without_repeat(sim_data, power_x)
    y_coords = get_dhw_y_coords(sim_data, x_coords, power_y)

    fig.add_trace(go.Scatter(x=x_coords, y=y_coords, mode='markers', name='Data'))

    fig.show()


def linear_regression(filename: str, power_x: int, power_y: int) -> None:
    """Plot the given x- and y-coordinates and linear regression and
    return coefficient of the linear regression
    """
    sim_data = get_xy_data(filename)
    x_coords = get_frequency_x_without_repeat(sim_data, power_x)
    y_coords = get_frequency_y_coords(sim_data, x_coords, power_y)
    co = np.polyfit(x_coords, y_coords, 1)
    func = np.poly1d(co)
    plt.plot(x_coords, y_coords, 'yo', x_coords, func(x_coords), '--k')
    plt.xlim(0, 25)
    plt.ylim(0, 25)


if __name__ == '__main__':
    file = os.getcwd() + '\\bcodmo_dataset_773466_712b_5843_9069.csv'
    header, unit, data = processing_data.load_csv_file(file)
    whole_data = processing_data.simplify_data(header, data)
    data_unit = {'average_bleaching': processing_data.get_unit(file, 'Average_Bleaching'),
                 'TSA_Frequency': processing_data.get_unit(file, 'TSA_Frequency'),
                 'TSA_DHW': processing_data.get_unit(file, 'TSA_DHW')}

    result_chi = chi_square.cal_chi_square(chi_square.chi_square(whole_data, 0, 0))

    plot_frequency_points(file, 1, 1)
    plot_dhw_points(file, 1, 1)

    best_model = model_selection.model_selection(file, [1, 2], [1, 2])

    linear_regression(file, best_model[0], best_model[1])

    doctest.testmod()

    python_ta.check_all(config={
        'extra-imports': ['sklearn.impute', 'numpy', 'csv', 'matplotlib.pyplot', 'math',
                          'statistics', 'model_selection', 'ploting', 'processing_data',
                          'sklearn.preprocessing', 'os', 'plotly.graph_objects', 'chi_square'],
        'allowed-io': ['load_csv_file'],
        'max-line-length': 100,
        'disable': ['R1705', 'C0200']
    })
