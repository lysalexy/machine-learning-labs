import matplotlib.pyplot as plt
import numpy as np
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
def eighth_task():
    data = read_csv('svmdata6.txt', delimiter='\t').to_numpy()
    X = data[:, 0].reshape(-1, 1)
    Y = data[:, 1]

    # %%

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)
    clf = SVR(C=1, kernel='rbf')

    epsilons = np.arange(0, 2, 0.1)

    errors = []

    for e in epsilons:
        clf.epsilon = e
        clf.fit(x_train, y_train)

        err = mean_squared_error(y_test, clf.predict(x_test))

        errors.append(err)

    plt.figure(figsize=(10, 6))

    plt.plot(epsilons, errors)

    plt.xlabel('Epsilon')

    plt.ylabel('Error')

    plt.show()
