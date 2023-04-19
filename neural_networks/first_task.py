import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow import keras
import random as python_random

from sklearn.model_selection import train_test_split

from sklearn import metrics

def draw_graph(x, y, xlabel, ylabel, title, is_scaley_needed):
    plt.plot(x, y,scaley=is_scaley_needed)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def draw_two_lines_graph( x,y_1, y_2, xlabel, ylabel, title,y_1_name,y_2_name, is_scaley_needed):
    plt.plot(x, y_1,scaley=is_scaley_needed)
    plt.plot(x,y_2,scaley=is_scaley_needed)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend([y_1_name,y_2_name])
    plt.show()


def draw_points_with_classes(source):
    point_class = source.groupby("class")
    for p_class, point in point_class:
        plt.plot(point.X1, point.X2, marker='o', linestyle='', markersize=5, label=p_class)
    plt.legend()
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('data illustration')
    plt.show()


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
                                                                              random_state=100)
    return training_data, test_data, training_target, test_target

def build_model(amount_of_neurons,optimizer,activation):
    np.random.seed(1234)
    python_random.seed(123)
    tf.random.set_seed(1234)
    print(amount_of_neurons)
    inputs = tf.keras.Input(shape=(2,))
    outputs = tf.keras.layers.Dense(amount_of_neurons, activation=activation, input_shape=(2,))(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())
    return model;

def create_one_layer_neuron_network(training_data,test_data,training_target,test_target,amount_of_neurons,epochs_amount,optimizer,activation):
    clf = keras.wrappers.scikit_learn.KerasClassifier(build_fn=build_model,amount_of_neurons=amount_of_neurons, optimizer=optimizer,
                                                      activation=activation, epochs=epochs_amount)
    history = clf.fit(np.asarray(training_data), np.asarray(training_target))
    predict=clf.predict(test_data);
    accuracy = metrics.accuracy_score(test_target, predict)
    return history, accuracy

def test_one_neuron_network_with_different_optimizers_and_activations(training_data,test_data,training_target,test_target,epochs_amount):
    optimizers = ['SGD', 'adam',
                  'adadelta',
                  'adagrad', 'adamax',
                  'nadam', 'ftrl',
                  'RMSprop']
    activations = ['selu', 'softmax', 'elu', 'gelu',
                   'softplus', 'relu',
                   'sigmoid', 'softsign', 'swish', 'tanh']
    optims = []
    accs = []

    for activation in activations:
        optims = []
        train_accs=[]
        test_accs = []
        for optimizer in optimizers:
            ir = [2, 4, 8, 16, 32, 63, 128, 256, 512]
            for i in ir:
                history, test_accuracy = create_one_layer_neuron_network(training_data, test_data, training_target,
                                                                         test_target, i, epochs_amount, optimizer,
                                                                         activation)

                optims.append(optimizer)
                train_accs.append(history.history['accuracy'][epochs_amount - 1])
                test_accs.append(test_accuracy)

        draw_two_lines_graph(optims, train_accs,test_accs, 'optimizer', 'accuracy', activation, 'train accuracy','test accuracy', True)

def test_nn(source):
    epochs_amount=100
    epochs = []
    for i in range(1, epochs_amount+1):
        epochs.append(i)
    nn = pd.read_csv(source)
    draw_points_with_classes(nn)
    training_data, test_data, training_target, test_target = split_data_for_classification(0.8, nn, None,
                                                                                                  True)
    history, accuracy = create_one_layer_neuron_network(training_data, test_data, training_target, test_target, 1,epochs_amount,'SGD',
                                                  'sigmoid')

    draw_graph( epochs,history.history['accuracy'],'epochs amount', 'train accuracy', 'accuracy to epochs', True)
    test_one_neuron_network_with_different_optimizers_and_activations(training_data, test_data, training_target,
                                                                      test_target,epochs_amount)

def first_task():
    test_nn(r'files\nn_0.csv')
    test_nn(r'files\nn_1.csv')

