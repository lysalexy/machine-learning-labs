import pandas as pd

from first_task import split_data_for_classification
from first_task import do_bayess_classification
from third_task import do_k_neighbours_classification
from fourth_task import do_svm

from second_task import draw_confusion_matrix

def prepare_bank_data(source):
    data = []
    target = []
    for row in source.values:
        new_row = []
        for value in range(1, len(row)):
            val=row[value]
            new_row.append(val)
        targ = row[0]
        data.append(new_row)
        target.append(targ)
    return data, target

def sixth_task():
    df_train = pd.read_csv(r'C:\Users\orang\Downloads\bank_scoring_train.csv',sep='\t')
    df_test = pd.read_csv(r'C:\Users\orang\Downloads\bank_scoring_test.csv',sep='\t')

    training_data, training_target = prepare_bank_data(df_train)
    test_data, test_target = prepare_bank_data(df_test)

    bayess_pred, bayess = do_bayess_classification(training_data,test_data,training_target,test_target)
    k_neighb_pred, k_neighb = do_k_neighbours_classification(training_data,test_data,training_target,test_target,20,'minkowski')
    svm_pred,svm = do_svm(training_data, training_target, test_data, test_target,'rbf',None,None,None)

    draw_confusion_matrix(test_target,bayess_pred,bayess)
    draw_confusion_matrix(test_target,k_neighb_pred,k_neighb)
    draw_confusion_matrix(test_target,svm_pred,svm)
