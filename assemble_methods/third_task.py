import pandas as pd
import numpy as np
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score

from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC

from first_task import prepare_data



def third_task():
    df_train = pd.read_csv(r'files\titanic_train.csv', sep=',')
    df_test = pd.read_csv(r'files\titanic_test.csv', sep=',')

    df_train=df_train.drop('Survived', axis=1)
    df_train = df_train.drop('PassengerId', axis=1)
    df_test = df_test.drop('PassengerId', axis=1)
    df_train = df_train.drop('Name', axis=1)
    df_test = df_test.drop('Name', axis=1)
    df_train = df_train.drop('Ticket', axis=1)
    df_test = df_test.drop('Ticket', axis=1)

    df_train['Age'].astype(float)
    df_test['Age'].astype(float)
    df_train['Fare'].astype(float)
    df_test['Fare'].astype(float)

    all=pd.concat([df_train,df_test])

    enc = OrdinalEncoder()
    enc.fit(all)
    df_train =pd.DataFrame(enc.transform(df_train))
    df_test = pd.DataFrame(enc.transform(df_test))

    all = pd.concat([df_train, df_test])

    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(all)
    df_train=pd.DataFrame(imp.transform(df_train))
    df_test = pd.DataFrame(imp.transform(df_test))


    train_data, train_target = prepare_data(df_train, None, False)
    test_data, test_target = prepare_data(df_test, None, False)

    lab = preprocessing.LabelEncoder()
    train_target = lab.fit_transform(train_target)
    test_target = lab.fit_transform(test_target)

    estimators = [
        ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
        ('bayess',GaussianNB()),
        ('svc and bayess', make_pipeline(StandardScaler(),LinearSVC(random_state=42))),
        ('svc',KNeighborsClassifier(35))
    ]

    clf = StackingClassifier(estimators=estimators, final_estimator=GaussianNB())
    clf.fit(train_data, train_target)

    print(accuracy_score(test_target, clf.predict(test_data)))

