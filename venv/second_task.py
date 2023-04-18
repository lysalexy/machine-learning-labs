from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve

from first_task import do_bayess_classification
from first_task import split_data_for_classification

def get_xvalue(classNumber, x1_mean, x2_mean, deviation):
    x=[]
    x =pd.DataFrame.from_dict({
        "x1": [norm.rvs(x1_mean, deviation)],
        "x2": [norm.rvs(x2_mean, deviation)],
        "class": [classNumber]
    })
    return x

def generate_data():
    data = pd.DataFrame(columns=["x1", "x2","class"])
    for i in range (0,30):
        data = pd.concat([data,get_xvalue(-1, 22, 2, 3)], ignore_index=True)
    for i in range (0,70):
        data = pd.concat([data, get_xvalue(1, 9, 14, 4)], ignore_index=True)
    return data

def draw_points_with_classes(source):
    point_class = source.groupby("class")
    for p_class, point in point_class:
        plt.plot(point.x1, point.x2, marker='o', linestyle='', markersize=5, label=p_class)
    plt.legend()
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('data illustration')
    plt.show()

def draw_confusion_matrix(test_target, pred, classifier):
    confusion_m = confusion_matrix(test_target, pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_m)
    disp.plot()
    plt.show()

def draw_roc(gnb, test_data,test_target):
    pred_prob = gnb.predict_proba(test_data)
    fpr, tpr, _ = roc_curve(test_target, pred_prob[:, 1])
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1])
    legend = ('ROC', 'Random guess')
    plt.legend(legend)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def draw_pr_curve(gnb, test_data, test_target):
    pred_prob = gnb.predict_proba(test_data)
    precision, recall, thresholds = precision_recall_curve(test_target, pred_prob[:, 1])
    plt.plot(recall, precision)
    legend = ('PR-Curve', '')
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.legend(legend)
    plt.show()

def second_task():
    source = generate_data()
    draw_points_with_classes(source)
    training_data, test_data, training_target, test_target=split_data_for_classification(0.4, source, None, True)
    pred, gnb = do_bayess_classification(training_data, test_data, training_target, test_target)

    accuracy = accuracy_score(test_target, pred)
    print('accuracy is '+ str(accuracy))

    draw_confusion_matrix(test_target, pred,gnb)

    draw_roc(gnb,test_data, test_target)

    draw_pr_curve(gnb,test_data, test_target)

