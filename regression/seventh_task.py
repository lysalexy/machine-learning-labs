import matplotlib.pyplot as plt
from pandas import read_csv
from sklearn.linear_model import LinearRegression


def seventh_task():
    data = read_csv('cars.csv', delimiter=',').to_numpy()



    X = data[:, 0]
    Y = data[:, 1]
    X = X.reshape(-1, 1)

    plt.plot(X, Y)

    clf = LinearRegression()
    clf.fit(X, Y)
    pred = clf.predict(X)

    plt.plot(X, pred)
    plt.xlabel('Speed')
    plt.ylabel('Distance')
    plt.legend(('Real', 'Regression'))

    plt.show()

    d2 = clf.predict([[40]])[0]
    print(d2)
