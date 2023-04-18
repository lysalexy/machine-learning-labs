from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt


def draw_graph(x, y, xlabel, ylabel, title, is_scaley_needed):
    plt.plot(x, y,scaley=is_scaley_needed)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def spam_map(element):
    if (element == 'spam')or(element == 'y'):
        return 1
    elif (element == 'nonspam')or(element == 'n'):
        return 0
    else:
        return element


def tick_tack_map(element):
    if element == 'x':
        return 1
    elif element == 'o':
        return 0
    elif element == 'b':
        return -1
    elif element == 'positive':
        return 5
    else:
        return -5


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

def split_data_for_classification(train_percent,source,map,is_first_element_needed):
    data, target = prepare_data(source, map, is_first_element_needed)
    training_data, test_data, training_target, test_target = train_test_split(data, target,
                                                                              test_size=1 - train_percent,
                                                                              random_state=42)
    return training_data, test_data, training_target, test_target

def do_bayess_classification(training_data, test_data, training_target, test_target):
    gnb = GaussianNB()
    gnb.fit(training_data, training_target)
    pred = gnb.predict(test_data)
    return pred, gnb


def count_accuracy_bayess(train_percent, source, map):
    training_data, test_data, training_target, test_target=split_data_for_classification(train_percent,source,map,True)
    pred, gnb = do_bayess_classification(training_data, test_data, training_target, test_target)

    accuracy = accuracy_score(test_target, pred)
    return accuracy

def collect_accuracy_bayess(source,map):
    train_percentages = []
    accuracies = []

    for i in range(1, 10):
        train_percent = 0.1 * i
        train_percentages.append(train_percent)

        accuracies.append(count_accuracy_bayess(train_percent, source, map))
    return train_percentages,accuracies


def do_accuracy_graphs(x, y, xlabel, ylabel, title):
    draw_graph(x, y, xlabel, ylabel, title,True)
    draw_graph(x, y, xlabel, ylabel, title, False)


def spam():
    df = pd.read_csv(r'C:\Users\orang\Downloads\spam.csv')
    train_percentages, accuracies = collect_accuracy_bayess(df, spam_map)
    do_accuracy_graphs(train_percentages, accuracies, "training fraction", "accuracy",
                       "spam/nonspam classification")


def tic_tac():
    df = pd.read_fwf(r'C:\Users\orang\Downloads\tic_tac_toe.txt')
    series = df.squeeze()
    series_with_name = series.append(pd.Series(series.name), ignore_index=True)
    splited_series = series_with_name.str.split(pat=',', expand=True)

    train_percentages, accuracies = collect_accuracy_bayess(splited_series, tick_tack_map)

    do_accuracy_graphs(train_percentages, accuracies, "training fraction", "accuracy",
                      "positive/negative classification")


def first_task():
    tic_tac()
    spam()
