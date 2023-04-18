import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

from first_task import prepare_data
from first_task import spam_map
from third_task import read_valuable_data
from second_task import draw_confusion_matrix

def find_accuracies_depth(criterion,depths,training_data, test_data, training_target, test_target):
    accuracy_depth = []
    for depth in depths:
        dtc = DecisionTreeClassifier(criterion=criterion, max_depth=depth)
        dtc.fit(training_data, training_target)
        accuracy_depth.append(accuracy_score(test_target, dtc.predict(test_data)))
    return accuracy_depth

def do_decision_tree_classifier(data,target,test_percent, max_depth):
    training_data, test_data, training_target, test_target = train_test_split(data, target, test_size=test_percent)
    clf = DecisionTreeClassifier(max_depth=max_depth)
    clf.fit(training_data, training_target)
    print(clf.get_depth())
    pred = clf.predict(test_data)
    return training_data, test_data, training_target, test_target, pred, clf

def draw_decision_tree(clf):
    plt.figure(figsize=(35, 35))
    plot_tree(clf, filled=False, rounded=True, fontsize=10)
    plt.show()

def draw_accuracy_depth_dependance_on_parameter(parameter, depths, training_data, test_data, training_target, test_target):
    legend = (parameter)
    accuracies_depth = [find_accuracies_depth(parameter, depths, training_data, test_data, training_target, test_target)]
    plt.figure(figsize=(10, 10))
    plt.grid(True)
    for accuracy in accuracies_depth:
        plt.plot(depths, accuracy)

    plt.xlabel('Max tree depth')
    plt.ylabel('Accuracy')
    plt.legend(legend)
    plt.show()

def a(data_txt):
    source = read_valuable_data(data_txt)
    data, target = prepare_data(source, None, True)
    training_data, test_data, training_target, test_target, pred, clf_10= do_decision_tree_classifier(data, target,0.2,10)
    draw_decision_tree(clf_10)
    print(accuracy_score(test_target, pred))

    depths = [_ for _ in np.arange(1, 100)]
    leaf_nodes = [_ for _ in np.arange(3, 100)]
    draw_accuracy_depth_dependance_on_parameter('gini', depths, training_data, test_data, training_target, test_target)
    draw_accuracy_depth_dependance_on_parameter('entropy', depths, training_data, test_data, training_target, test_target)

    training_data, test_data, training_target, test_target, pred, clf_14 = do_decision_tree_classifier(data, target,
                                                                                                       0.2, 14)
    draw_decision_tree(clf_14)
    print(accuracy_score(test_target, pred))

    training_data, test_data, training_target, test_target, pred, clf_7 = do_decision_tree_classifier(data,target, 0.2, 7)

    print(clf_7.get_depth())
    print(accuracy_score(test_target, pred))
    draw_decision_tree(clf_7)

def b(source_txt):
    df = pd.read_csv(source_txt)
    data,target = prepare_data(df,spam_map,True)

    training_data, test_data, training_target, test_target, pred, clf= do_decision_tree_classifier(data,target,0.2,3)

    print(accuracy_score(test_target, pred))

    draw_confusion_matrix(test_target,pred,clf)
    draw_decision_tree(clf)

def fifth_task():
    a(r'C:\Users\orang\Downloads\glass.csv')
    b(r'C:\Users\orang\Downloads\spam7.csv')