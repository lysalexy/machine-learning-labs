import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from first_task import do_accuracy_graphs
from first_task import draw_graph

from first_task import split_data_for_classification

def read_valuable_data(source):
    df = pd.read_csv(source)

    data = pd.DataFrame(columns=["RI", "Na", "Mg","Al","Si","K","Ca","Ba","Fe","type"])
    for row in df.values:
        value = pd.DataFrame.from_dict({
            "RI":[row[1]],
            "Na":[row[2]],
            "Mg":[row[3]],
            "Al":[row[4]],
            "Si":[row[5]],
            "K":[row[6]],
            "Ca":[row[7]],
            "Ba":[row[8]],
            "Fe":[row[9]],
            "type":[row[10]]
        })
        data = pd.concat([data, value], ignore_index=True)
    print(data)
    return data



def do_k_neighbours_classification( training_data, test_data, training_target, test_target, neighbours_amount, metric):
    knc = KNeighborsClassifier(neighbours_amount, metric=metric)
    knc.fit(training_data, training_target)
    pred = knc.predict(test_data)
    return pred, knc

def count_accuracy_k_neighbours(source, neighbours_amount, metric, test_percent):
    training_data, test_data, training_target, test_target=split_data_for_classification(1-test_percent,source,None,True)
    pred, knc=do_k_neighbours_classification(training_data, test_data, training_target, test_target,
                    neighbours_amount,metric)
    accuracy = accuracy_score(test_target, pred)
    return accuracy

def collect_accuracy_k_neighbours(source, max_neighbours_amount,test_percent):
    accuracies=[]
    neighbours_amount = []
    for i in range (1, max_neighbours_amount):
        accuracies.append(count_accuracy_k_neighbours(source,i,"minkowski",test_percent))
        neighbours_amount.append(i)
    return neighbours_amount, accuracies

def research_metric_influence_on_accuracy(source, test_percent):
    metrics_to_accuracy = {'minkowski': 0, 'chebyshev': 0, 'euclidean': 0, 'manhattan': 0,
                           'braycurtis':0, 'correlation':0,'cosine':0}
    for metric in metrics_to_accuracy:
        metrics_to_accuracy[metric]=count_accuracy_k_neighbours(source,5,metric,test_percent)
        print(str(metric)+ " metric accuracy is " + str(metrics_to_accuracy[metric]))
    return metrics_to_accuracy

def classify_object(source, object, max_neighbours_amount,test_percent):
    neighbours_amount = []
    classes = []
    for i in range(1,max_neighbours_amount):
        training_data, test_data, training_target, test_target = split_data_for_classification(1 - test_percent, source,
                                                                                               None, True)
        pred, knc = do_k_neighbours_classification(training_data, test_data, training_target, test_target,
                                                   i, "minkowski")
        classes.append(knc.predict(object)[0])
        neighbours_amount.append(i)
    draw_graph(neighbours_amount, classes,'amount of neighbours','predicted class','',True)


def third_task():
    test_percent = 0.6
    source = read_valuable_data(r'files\glass.csv')
    neighbours_amount, accuracies=collect_accuracy_k_neighbours(source,
                                                                round(len(source)*(1-test_percent)),test_percent)
    do_accuracy_graphs(neighbours_amount,accuracies,"neighbours amount","accuracy","type classification")

    research_metric_influence_on_accuracy(source, test_percent)

    classify_object(source,[[1.516, 11.7, 1.01, 1.19, 72.59, 0.43, 11.44, 0.02, 0.1]],
                    round(len(source)*(1-test_percent)),test_percent)