import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
import random as python_random
import tensorflow as tf
import numpy as np

from sklearn import metrics

from neural_networks.first_task import split_data_for_classification


def build_model(shape, amount_of_neurons,optimizer,activation,loss):
    np.random.seed(1234)
    python_random.seed(123)
    tf.random.set_seed(1234)
    print(amount_of_neurons)
    inputs = tf.keras.Input(shape=shape)
    x = tf.keras.layers.Dense(amount_of_neurons, activation=activation, input_shape=shape)(inputs)
    outputs=tf.keras.layers.Dense(2, activation=activation)(x)#нет для первого задания
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    #sgd = keras.optimizers.SGD(weight_decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy'])
    print(model.summary())
    return model;

def create_one_layer_neuron_network(training_data,test_data,training_target,test_target,amount_of_neurons,epochs_amount,optimizer,activation, shape, loss):
    clf = keras.wrappers.scikit_learn.KerasClassifier(build_fn=build_model,amount_of_neurons=amount_of_neurons, optimizer=optimizer,
                                                      activation=activation, epochs=epochs_amount, shape = shape, loss=loss)
    history = clf.fit(np.asarray(training_data), np.asarray(training_target))
    predict=clf.predict(test_data);
    accuracy = metrics.accuracy_score(test_target, predict)
    return clf, history, accuracy

def draw_optimizator_activation_amount_of_neurons_graphs(x,y_1,y_2,title,subtitles,xlabel,ylabel,y_1_name, y_2_name):
    # plt.title(title)
    fig, subs = plt.subplots(4, 2)
    title = 0
    for sub in subs:
        for i in range(0,2):
            sub[i].set_title(subtitles[title])
            sub[i].plot(x, y_1[title], scaley=False)
            sub[i].plot(x, y_2[title], scaley=False)
            sub[i].set_xlabel(xlabel)
            sub[i].set_ylabel(ylabel)
            sub[i].legend([y_1_name, y_2_name])
            title += 1;
    plt.show()


def test_one_layer_network_with_different_optimizers_and_activations(training_data,test_data,training_target,test_target,epochs_amount,shape,loss):
    optimizers = ['SGD', 'adam',
                  'adadelta',
                  'adagrad', 'adamax',
                  'nadam', 'ftrl',
                  'RMSprop']
    activations = [
        'selu',  'elu', 'gelu',
                   'softplus', 'relu',
                    'softsign', 'swish', 'tanh']
    optims = []
    accs = []

    for activation in activations:
        optims = []
        train_accs=[]
        test_accs = []
        for optimizer in optimizers:
            ir = [2,4, 8, 16, 32, 63, 128, 256, 512]
            opt_train_accs=[]
            opt_test_accs=[]
            for i in ir:
                clf, history, test_accuracy = create_one_layer_neuron_network(training_data, test_data, training_target,
                                                                         test_target, i, epochs_amount, optimizer,
                                                                         activation,shape,loss)

                opt_train_accs.append(history.history['accuracy'][epochs_amount - 1])
                opt_test_accs.append(test_accuracy)
            train_accs.append(opt_train_accs)
            test_accs.append(opt_test_accs)
        draw_optimizator_activation_amount_of_neurons_graphs(ir,test_accs,train_accs,activation,optimizers,'amount of neurons',
                                                             'accuracy','test accuracy','train accuracy')

def second_task():
    nn = pd.read_csv(r'files\nn_1.csv')
    training_data, test_data, training_target, test_target = split_data_for_classification(0.8, nn, None,
                                                                                           True)
    test_one_layer_network_with_different_optimizers_and_activations(training_data, test_data, training_target, test_target,50,(2,),'binary_crossentropy')

    clf, history,accuracy = create_one_layer_neuron_network(training_data, test_data, training_target, test_target,34,50,'RMSprop','gelu',(2,),'binary_crossentropy')
    print(accuracy)
