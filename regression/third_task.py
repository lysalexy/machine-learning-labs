from pandas import read_csv
from sklearn.linear_model import LinearRegression

from neural_networks.first_task import draw_two_lines_graph
from first_task import get_results_for_classifier


def third_task():
    data = read_csv('cygage.txt', delimiter='\t').to_numpy()

    clf = LinearRegression()
    x_train, x_test, y_train, y_test, res, accuracy = get_results_for_classifier(LinearRegression(),  0.4,data, 0, 1, None, None)
    draw_two_lines_graph(x_test, y_test, res, "Depth", "calAge", "calAge dependance from Depth", "actual", "predicted",
                         True)

    x_train, x_test, y_train, y_test, res, accuracy = get_results_for_classifier(LinearRegression(),  0.4,data,  0, None, None, None)
    draw_two_lines_graph(x_test[:, 0], y_test, res, "Depth", "calAge", "calAge dependance from Depth", "actual",
                         "predicted", True)
