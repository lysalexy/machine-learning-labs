import numpy as np
from pandas import read_csv
from sklearn.metrics import mean_absolute_error

from first_task import get_results_for_classifier
from sklearn.linear_model import LinearRegression, Ridge
from neural_networks.first_task import draw_two_lines_graph


def fourth_task():
    data = read_csv('longley.csv.', delimiter=',').to_numpy()
    data = np.delete(data, [4], axis=1)

    x_train, x_test, y_train, y_test, res, accuracy = get_results_for_classifier(LinearRegression(), 0.5, data, 5, None,
                                                                                 None, None)
    print(accuracy)

    lambd = []
    acc_test = []
    acc_train = []

    for i in range(0, 25):
        lamb = 10 ** (-3 + 0.2 * i)
        clf = Ridge(alpha=lamb)
        lambd.append(lamb)
        x_train, x_test, y_train, y_test, res, accuracy = get_results_for_classifier(clf, 0.5, data, 5,
                                                                                     None, None, None)
        clf.fit(x_train, y_train)
        acc_test.append(mean_absolute_error(y_test, clf.predict(x_test)))
        acc_train.append(mean_absolute_error(y_train, clf.predict(x_train)))
    draw_two_lines_graph(lambd, acc_train, acc_test, "lambda", "mean absolute error",
                         "Dependance of mean absolute error from lambda",
                         "train", "test", True)
