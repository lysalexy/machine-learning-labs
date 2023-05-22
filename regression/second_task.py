import numpy as np
from pandas import read_csv
from sklearn.linear_model import LinearRegression, Ridge

from first_task import get_results_for_classifier


def second_task():
    data = read_csv('reglab.txt', delimiter='\t').to_numpy()
    vars = [0, 1, 2]
    variables = ["y", "x1", "x2", "x3", "x4"]

    for i in range(1, 5):
        x_train, x_test, y_train, y_test, res, accuracy = get_results_for_classifier(Ridge(),  0.4,data, 0, i, None, None)
        print("Accuracy for dependance " + str(variables[0]) + " from " + str(variables[i]) + " is " + str(
            accuracy))
        for j in range(1, 5):
            if i != j:
                x_train, x_test, y_train, y_test, res, accuracy = get_results_for_classifier(Ridge(), 0.4, data,  0, i, j, None)
                print("Accuracy for dependance " + str(variables[0]) + " from " + str(variables[i]) + " and " + str(
                    variables[j]) + " is " + str(
                    accuracy))
                for k in range(1, 5):
                    if j != k:
                        x_train, x_test, y_train, y_test, res, accuracy = get_results_for_classifier(Ridge(),  0.4, data,0, i, j, k)
                        print(
                            "Accuracy for dependance " + str(variables[0]) + " from " + str(variables[i]) + " and " + str(
                                variables[j]) + " and " + str(
                                variables[k]) + " is " + str(
                                accuracy))

    x_train, x_test, y_train, y_test, res, accuracy = get_results_for_classifier(Ridge(), 0.4, data,  0, None, None, None)
    print("Accuracy for " + str(variables[0]) + " is " + str(accuracy))
