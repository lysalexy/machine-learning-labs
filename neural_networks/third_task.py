from tensorflow import keras
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def draw_confusion_matrix(test_target, pred, classifier):
    confusion_m = confusion_matrix(test_target.argmax(axis=1), pred.argmax(axis=1))
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_m)
    disp.plot()
    plt.show()

def build_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    # compile model
    opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def third_task():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')
    x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')

    number_class = 10
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    clf = keras.wrappers.scikit_learn.KerasClassifier(build_fn=build_model, epochs=15,batch_size=128,verbose=1)

    clf.fit(x_train, y_train, batch_size=128, epochs=15, verbose=1)
    pred_x_test=clf.predict(x_test)
    pred_x_test=tf.keras.utils.to_categorical(pred_x_test)
    draw_confusion_matrix(y_test,pred_x_test,clf)


