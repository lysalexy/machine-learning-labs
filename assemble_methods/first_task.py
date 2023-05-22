import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier


def prepare_data(source, map, first_element_is_needed):
    data = []
    target = []
    for row in source.values:
        new_row = []
        i=1
        if first_element_is_needed==True:
            i=0
        for value in range(i, len(row) - 1):
            val=row[value]
            if map != None:
                val = map(row[value])
            new_row.append(val)
        targ = row[len(row)- 1]
        if map!=None:
            targ = map(row[len(row) - 1])
        data.append(new_row)
        target.append(targ)
    return data, target

def split_data_for_classification(test_percent, source, map):
    data, target = prepare_data(source,map, True)
    training_data, test_data, training_target, test_target = train_test_split(data, target,
                                                                              test_size=test_percent,
                                                                              random_state=42)
    return training_data, test_data, training_target, test_target


def read_valuable_data(source):
    df = pd.read_csv(source)

    data = pd.DataFrame(columns=["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "type"])
    for row in df.values:
        value = pd.DataFrame.from_dict({
            "RI": [row[1]],
            "Na": [row[2]],
            "Mg": [row[3]],
            "Al": [row[4]],
            "Si": [row[5]],
            "K": [row[6]],
            "Ca": [row[7]],
            "Ba": [row[8]],
            "Fe": [row[9]],
            "type": [row[10]]
        })
        data = pd.concat([data, value], ignore_index=True)
    print(data)
    return data


def draw_graph(x, y, xlabel, ylabel, title, is_scaley_needed):
    plt.plot(x, y, scaley=is_scaley_needed)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def test_assemble_method(methodIsBagging, training_data, test_data, training_target, test_target, estimator, est_amount):
    clf=0
    if (methodIsBagging):
        clf = BaggingClassifier(estimator=estimator, n_estimators=est_amount, random_state=42).fit(training_data,
                                                                                               training_target)
    else:
        clf = AdaBoostClassifier(estimator=estimator, n_estimators=est_amount, random_state=42).fit(training_data,
                                                                                               training_target)
    return accuracy_score(test_target, clf.predict(test_data))


def test_accuracy_dependence_on_estimators_amount(source, map, methodIsBagging):
    test_percent = 0.6
    training_data, test_data, training_target, test_target = split_data_for_classification(test_percent, source,map)

    est_amount = [5, 10, 50, 100, 250, 500, 750, 1000]
    svc = SVC(probability=True,kernel='linear')

    estimators = [GaussianNB(), KNeighborsClassifier(), DecisionTreeClassifier()] #KNeighborsClassifier(), SVC() для 1, svc для 2 без KNeighborsClassifier()

    for estimator in estimators:
        test_accs = []

        for n in est_amount:
            accuracy_score = test_assemble_method(methodIsBagging, training_data, test_data, training_target, test_target, estimator, n)
            test_accs.append(accuracy_score)
        draw_graph(est_amount, test_accs, 'estimators amount', 'accuracy', '', True)


def first_task():
    source = read_valuable_data(r'files\glass.csv')
    test_accuracy_dependence_on_estimators_amount(source, None,True)
