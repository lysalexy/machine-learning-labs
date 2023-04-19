import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import svm
from  sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import accuracy_score

from first_task import prepare_data
from first_task import do_accuracy_graphs
from second_task import draw_confusion_matrix

def svm_data_map(element):
    if element == 'red':
        return 1
    elif ((element == 'green')or(element == 'gree')):
        return 0
    else:
        return element

def read_svm_data_to_df(source):
    df = pd.read_fwf(source)
    series = df.squeeze()
    splited_series = series.str.split(pat='\t', expand=True)
    return splited_series

def get_test_and_train_data(train_txt,test_txt):
    train_df = read_svm_data_to_df(train_txt)
    test_df = read_svm_data_to_df(test_txt)
    training_data, training_target = prepare_data(train_df, svm_data_map,False)
    test_data, test_target = prepare_data(test_df, svm_data_map,False)
    return training_data, training_target, test_data, test_target

def do_svm(training_data, training_target, test_data, test_target, kernel_type, C_value, degree_value, gamma_value):
    clf = svm.SVC(kernel=kernel_type)
    if C_value!=None:
        clf = svm.SVC(kernel=kernel_type, C=C_value)
    if (degree_value!=None)and(gamma_value!=None):
        clf = svm.SVC(kernel=kernel_type, degree=degree_value, gamma=gamma_value)
    elif gamma_value!=None:
        clf = svm.SVC(kernel=kernel_type, gamma=gamma_value)
    elif degree_value!=None:
        clf = svm.SVC(kernel=kernel_type, degree=degree_value)
    clf.fit(training_data, training_target)
    pred = clf.predict(test_data)
    return pred, clf


def draw_model(clf, x_t, y_t, t, x_label, y_label):
    x0=[]
    for elem in x_t:
        x0.append(float(elem[0]))
    x0= np.asarray(x0)

    x1 = []
    for elem in x_t:
        x1.append(float(elem[1]))
    x1=np.asarray(x1)

    numeric_x=[]
    for row in x_t:
        new_row=[]
        for elem in row:
            new_row.append(float(elem))
        numeric_x.append(new_row)

    x_array = np.asarray(numeric_x)

    disp = DecisionBoundaryDisplay.from_estimator(
            clf,
            x_array,
            response_method="predict",
            cmap=plt.cm.coolwarm,
            alpha=0.8,
            xlabel=x_label,
            ylabel=y_label,
    )
    plt.scatter(x0, x1, c=y_t, cmap=plt.cm.coolwarm, s=10, edgecolors="k")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(t)

    plt.show()

def a(train_txt,test_txt):
    training_data, training_target, test_data, test_target=get_test_and_train_data(train_txt,test_txt)
    pred, clf =do_svm(training_data, training_target, test_data, test_target,'linear', None, None, None)
    draw_confusion_matrix(test_target,pred,clf)
    print("amount of support vectors =" + str(len(clf.support_vectors_)))
    draw_model(clf, test_data, test_target, t='Svm linear', x_label='X1', y_label='X2')


def count_accuracy_svm(training_data, training_target, test_data, test_target, kernel_type, C_value):
    pred, clf = do_svm(training_data,training_target,test_data,test_target,kernel_type,C_value, None, None)
    accuracy = accuracy_score(test_target, pred)
    return accuracy


def collect_accuracy_svm(training_data, training_target, test_data, test_target, kernel_type,max_C_value):
    accuracies=[]
    C_values = []
    for i in range (1, max_C_value):
        accuracies.append(count_accuracy_svm(training_data, training_target, test_data, test_target, kernel_type, i))
        C_values.append(i)
    return C_values, accuracies

def test_accuracy_dependance_on_C_value(training_data, training_target, test_data, test_target,kernel_type,max_C_value, title):
    C_values, accuracies = collect_accuracy_svm(training_data, training_target, test_data, test_target,kernel_type,max_C_value)
    do_accuracy_graphs(C_values, accuracies, "C value","accuracy", title)


def b(train_txt,test_txt):
    training_data, training_target, test_data, test_target = get_test_and_train_data(train_txt, test_txt)
    pred, clf = do_svm(training_data, training_target, test_data, test_target, 'linear', None, None, None)
    test_accuracy_dependance_on_C_value(training_data, training_target, training_data, training_target,'linear',
                                        1000, 'train data')
    test_accuracy_dependance_on_C_value(training_data, training_target, test_data, test_target, 'linear',
                                        1000, 'test data')
    draw_model(clf, test_data, test_target, t='Svm linear', x_label='X1', y_label='X2')

def get_svc_models(training_data, training_target, test_data, test_target,gamma_value):
    pred_linear, linear_clf = do_svm(training_data, training_target, test_data, test_target,'linear', None, None, gamma_value)
    pred_poly_1, poly_1_clf = do_svm(training_data, training_target, test_data, test_target, 'poly',  None, 1, gamma_value)
    pred_poly_2, poly_2_clf = do_svm(training_data, training_target, test_data, test_target, 'poly',  None, 2, gamma_value)
    pred_poly_3, poly_3_clf = do_svm(training_data, training_target, test_data, test_target, 'poly', None, 3, gamma_value)
    pred_poly_4, poly_4_clf = do_svm(training_data, training_target, test_data, test_target, 'poly', None, 4, gamma_value)
    pred_poly_5, poly_5_clf = do_svm(training_data, training_target, test_data, test_target, 'poly', None, 5, gamma_value)
    pred_rbf, rbf_clf = do_svm(training_data, training_target, test_data, test_target, 'rbf', None, None, gamma_value)
    pred_sigmoid, sigmoid_clf = do_svm(training_data, training_target, test_data, test_target, 'sigmoid', None, None, gamma_value)

    models=(linear_clf,poly_1_clf,
        poly_2_clf,poly_3_clf,
        poly_4_clf,poly_5_clf,
        rbf_clf,sigmoid_clf
    )
    return models


def test_different_kernel_types(training_data, training_target, test_data, test_target, gamma_value):
    models = get_svc_models(training_data, training_target, test_data, test_target, gamma_value)
    titles=['linear ', 'poly 1 ','poly 2 ','poly 3 ','poly 4 ','poly 5 ','rbf ', 'sigmoid ']
    if gamma_value!=None:
        for i  in range(0,len(titles)):
            titles[i]+=' gamma value is '+ str(gamma_value)

    for model,title in zip(models,titles):
        draw_model(model, test_data, test_target, title, x_label='X1',y_label='X2')


def c(train_txt,test_txt):
    training_data, training_target, test_data, test_target = get_test_and_train_data(train_txt, test_txt)

    test_different_kernel_types(training_data, training_target, test_data, test_target,None)


def d(train_txt, test_txt):
    training_data, training_target, test_data, test_target = get_test_and_train_data(train_txt, test_txt)

    test_different_kernel_types(training_data, training_target, test_data, test_target,None)

def e(train_txt, test_txt):
    training_data, training_target, test_data, test_target = get_test_and_train_data(train_txt, test_txt)

    gammas=[10,100,150,250]

    for gamma in gammas:
        test_different_kernel_types(training_data, training_target, test_data, test_target,gamma)


def fourth_task():
    a(r'files\svmdata_a.txt',r'files\svmdata_a_test.txt')
    b(r'files\svmdata_b.txt', r'files\svmdata_b_test.txt')
    c(r'files\svmdata_c.txt', r'files\svmdata_c_test.txt')
    d(r'files\svmdata_d.txt', r'files\svmdata_d_test.txt')
    e(r'files\svmdata_e.txt', r'files\svmdata_e_test.txt')
