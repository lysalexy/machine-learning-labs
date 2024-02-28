from pandas import read_csv
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import  r2_score
from sklearn.preprocessing import StandardScaler

from neural_networks.first_task import draw_two_lines_graph
from first_task import get_results_for_classifier


def third_task():
    data = read_csv('cygage.txt', delimiter='\t').to_numpy()


    clf = LinearRegression()
    # x_train, x_test, y_train, y_test, res, accuracy = get_results_for_classifier(LinearRegression(),  0.25, data, 0, None, None, None)
    #

    Y = data[:, 0]
    X = data[:, 1]

    # x_train_2=[]
    # x_test_2=[]
    # for i in range(0, len(x_train)):
    #     x_train_2.append(x_train[i][0])
    # for i in range(0, len(x_test)):
    #     x_test_2.append(x_test[i][0])
    # y_train_2 = y_train
    X = np.array(X).reshape(-1, 1)
    Y = np.array(Y).reshape(-1, 1)
    # print(x_test_2)
    # print(res)

    # clf.fit(x_train_2, y_train_2)
    clf.fit(X,Y)
    print(clf.score(X, Y))

    res = clf.predict(X)
    X1 = np.delete(data, [0], axis=1)

    clf1=LinearRegression()

    clf1.fit(X1, Y)
    res2 = clf1.predict(X1)
    print(clf1.score(X1, Y))

    print(r2_score(Y, res))
    print(r2_score(Y, res2))


    plt.scatter(X, Y)
    plt.plot(X, res)
    plt.plot(X1[:,0],res2)
    plt.title( "calAge dependance from Depth")
    plt.xlabel("Depth")
    plt.ylabel("calAge")
    plt.legend(["actual", "predicted without weight", "predicted with weight"])
    plt.show()

