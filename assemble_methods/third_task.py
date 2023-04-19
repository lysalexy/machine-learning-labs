import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


from sklearn import preprocessing
from sklearn import utils
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

from first_task import prepare_data

def titanic_map(element):
    if element == 'S':
        return 1
    elif element == 'Q':
        return 0
    elif element == 'C':
        return -1
    elif element == 'male':
        return 5
    elif element == 'female':
        return -5
    else:
        return element

def third_task():
    df_train = pd.read_csv(r'files\titanic_train.csv', sep=',')
    df_test = pd.read_csv(r'files\titanic_test.csv', sep=',')

    df_train=df_train.drop('Survived', axis=1)

    df_train =df_train.drop('Name', axis=1)
    df_test = df_test.drop('Name', axis=1)

    df_train = df_train.drop('Ticket', axis=1)
    df_test = df_test.drop('Ticket', axis=1)

    df_train = df_train.drop('Sex', axis=1)
    df_test = df_test.drop('Sex', axis=1)

    df_train = df_train.drop('Embarked', axis=1)
    df_test = df_test.drop('Embarked', axis=1)

    df_train = df_train.drop('Cabin', axis=1)
    df_test = df_test.drop('Cabin', axis=1)

    df_train.dropna(inplace=True)
    indices_to_keep = ~ df_train.isin([np.nan, np.inf, -np.inf]).any(axis=1)
    df_train=df_train[indices_to_keep].astype(np.float64)

    df_test.dropna(inplace=True)
    indices_to_keep = ~ df_test.isin([np.nan, np.inf, -np.inf]).any(axis=1)
    df_test = df_test[indices_to_keep].astype(np.float64)

    train_data, train_target = prepare_data(df_train, titanic_map, False)
    test_data, test_target = prepare_data(df_test, titanic_map, False)

    lab = preprocessing.LabelEncoder()
    train_target = lab.fit_transform(train_target)
    test_target = lab.fit_transform(test_target)

    print(train_data)
    print(test_data)


    # estimators = [
    #     ('bayess',GaussianNB())]
    #
    # clf = StackingClassifier(estimators=estimators, final_estimator=GaussianNB())
    # clf.fit(train_data, train_target)
    clf = GaussianNB().fit(train_data, train_target)
    print(accuracy_score(test_target, clf.predict(test_data)))

