import matplotlib.pyplot as plt
from pandas import read_csv
import numpy as np
from sklearn.linear_model import LinearRegression
def sixth_task():
    data = read_csv('JohnsonJohnson.csv', delimiter=',').to_numpy()

    qs = [[] for _ in range(4)]

    for i in data:
        qnum = int(i[0][-1])
        qs[qnum - 1].append(i)

    for i in range(len(qs)):
        qs[i] = np.array(qs[i])[:, 1].reshape(-1, 1)

    qs = np.array(qs)
    x_axis = range(len(qs[1]))
    years = np.arange(1960, 1981)

    all_years = np.average(np.concatenate(qs, axis=1), axis=1).reshape(-1, 1)


    plt.figure(figsize=(20, 10))

    for q in qs:
        plt.plot(x_axis, q)

    plt.plot(x_axis, all_years)

    plt.xticks(x_axis, years)

    plt.legend(('Q1', 'Q2', 'Q3', 'Q4', 'Average'))

    plt.show()


    plt.figure(figsize=(20, 10))

    preds_2016 = []

    clf = LinearRegression()
    yreshaped = years.reshape(-1, 1)
    for q in qs:
        clf.fit(yreshaped, q)
        pred = clf.predict(yreshaped)
        plt.plot(years, pred)
        preds_2016.append(clf.predict([[2016]])[0])

    plt.xticks(years, [str(i) for i in years])

    plt.legend(('Q1', 'Q2', 'Q3', 'Q4'))
    plt.show()

    for q in qs:
        clf.fit(yreshaped, q)
        pred = clf.predict(yreshaped)
        plt.plot(years, pred)
        plt.plot(years, q)
        plt.show()


    for p, i in zip(preds_2016, range(1, 5)):
        print('Q' + str(i), p, sep='\t')

    clf.fit(yreshaped, all_years)
    print(clf.predict([[2016]])[0])
