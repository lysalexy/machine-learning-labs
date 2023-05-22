import matplotlib.pyplot as plt
import numpy as np
from pandas import read_csv
from sklearn.linear_model import LinearRegression


def fifth_task():
    data = read_csv('eustock.csv', delimiter=',').to_numpy()
    names = ["DAX", "SMI", "CAC", "FTSE"]
    x = []
    for i in range(0, len(data)):
        x.append(i)
    x=np.array(x).reshape(-1, 1)

    plt.plot(x, data[:, [0]], scaley=True)
    plt.plot(x, data[:, [1]], scaley=True)
    plt.plot(x, data[:, [2]], scaley=True)
    plt.plot(x, data[:, [3]], scaley=True)
    plt.legend(names)
    plt.show()

    i = 0
    for name in names:
        clf = LinearRegression()
        clf.fit(x, data[:, [i]])
        plt.plot(x, clf.predict(x), scaley=True)
        i += 1

    plt.legend(["DAX", "SMI", "CAC", "FTSE"])
    plt.show()
