import numpy as np
from pandas import read_csv
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def generate_dataset_depend_from_all_other_variables(data, testPercent, var):
    Y = data[:, var]
    X = np.delete(data, [var], axis=1)

    return train_test_split(X, Y, test_size=testPercent, random_state=42)


def generate_dataset_depend_from_one_variable(data, testPercent, numY, numX):
    Y = []
    X = []
    for i in range(0, len(data)):
        X.append([data[i][numX]])
        Y.append([data[i][numY]])

    return train_test_split(X, Y, test_size=testPercent)


def generate_dataset_depend_from_two_variables(data, testPercent, numY, numX1, numX2):
    Y = []
    X = []
    for i in range(0, len(data)):
        X.append([data[i][numX1], data[i][numX2]])
        Y.append([data[i][numY]])

    return train_test_split(X, Y, test_size=testPercent)


def generate_dataset_depend_from_three_variables(data, testPercent, numY, numX1, numX2, numX3):
    Y = []
    X = []
    for i in range(0, len(data)):
        X.append([data[i][numX1], data[i][numX2], data[i][numX3]])
        Y.append([data[i][numY]])

    return train_test_split(X, Y, test_size=testPercent, random_state=42)


def get_results_for_classifier(clf, testPercent, data, var, var2, var3, var4):
    if (var2 is not None) and (var3 is None) and (var4 is None):
        x_train, x_test, y_train, y_test = generate_dataset_depend_from_one_variable(data, testPercent, var, var2)
    elif (var2 is not None) and (var3 is not None) and (var4 is None):
        x_train, x_test, y_train, y_test = generate_dataset_depend_from_two_variables(data, testPercent, var, var2,
                                                                                      var3)
    elif (var2 is not None) and (var3 is not None) and (var4 is not None):
        x_train, x_test, y_train, y_test = generate_dataset_depend_from_three_variables(data, testPercent, var, var2,
                                                                                        var3, var4)
    else:
        x_train, x_test, y_train, y_test = generate_dataset_depend_from_all_other_variables(data, testPercent, var)

    clf.fit(x_train, y_train)
    print( clf.score(x_train, y_train))
    rss = np.sum(np.square(y_test - clf.predict(x_test)))
    print(rss)

    return  x_train, x_test, y_train, y_test, clf.predict(x_test), clf.score(x_test, y_test)


# %%
def first_task():
    data = read_csv('reglab1.txt', delimiter='\t').to_numpy()
    vars = [0, 1, 2]
    variables = ["x", "y", "z"]

    for i in range(0, 3):
        for j in range(0, 3):
            if i != j:
                x_train, x_test, y_train, y_test, res, accuracy = get_results_for_classifier(LinearRegression(), 0.4,data,
                i, j, None, None)
                print("Accuracy for dependance " + str(variables[i]) + " from " + str(variables[j]) + " is " + str(
                    accuracy))

    i = 0
    for v in vars:
        x_train, x_test, y_train, y_test, res, accuracy = get_results_for_classifier(LinearRegression(),0.4,  data, v, None, None, None)
        print("Accuracy for " + str(variables[i]) + " is " + str(accuracy))
        i += 1
