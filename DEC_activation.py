from keras import Model, Input, optimizers
from keras.datasets import mnist
from keras.engine import Layer, InputSpec
import keras.backend as K
#https://www.dlology.com/blog/how-to-do-unsupervised-clustering-with-keras/

from keras.layers import Activation, Lambda
from keras.losses import kullback_leibler_divergence
from keras.metrics import categorical_accuracy
from keras.optimizers import SGD
from keras.utils import get_custom_objects
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

import metrics
from model import get_encoder, n_hidden_layers
import numpy as np

# http://louistiao.me/posts/implementing-variational-autoencoders-in-keras-beyond-the-quickstart-tutorial/
# http://krasserm.github.io/2018/07/27/dfc-vae/
(x_train, labels_train), (x_test, labels_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
sae_autoencoder, encoder = get_encoder()
n_clusters = 10
input_dim = 10
original_dim = 28*28
hidden_dim_1 = 500
hidden_dim_2 = 500
hidden_dim_3 = 2000
encoded_dim = 10
dimentions = [original_dim, hidden_dim_1, hidden_dim_2, hidden_dim_3, encoded_dim]

class SoftAssignment(Layer):

    def __init__(self, n_clusters, alpha, **kwargs):
        """

        :param input_shape: size of embedded vector
        :param output_dim: n_clusters
        :param kwargs:
        """
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        self.n_clusters = n_clusters
        self.alpha = alpha
        super(SoftAssignment, self).__init__(**kwargs)

    def build(self, input_shape):
        print(input_shape)
        input_dim = input_shape[1]
        self.clusters = self.add_weight((n_clusters, input_dim), initializer='glorot_uniform', name='clusters', trainable=True)
        super(SoftAssignment, self).build(input_shape)

    def call(self, inputs, **kwargs):
        print(f'shape: {inputs.shape}')
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q
    # def output_shape(self):
    #     return (self.input_shape[0], self.out)


def distribution_divergence_loss(y_true, y_pred):
        print(f'q shape: {y_pred.shape}')
        p_distribution = get_target_distr(y_pred, alpha=1)
        # p_distribution = K.transpose(p_distribution)
        print(f'p shape: {p_distribution.shape}')
        return kullback_leibler_divergence(p_distribution, y_pred)

# def class_prediction_accuracy(y_true, y_pred):
#     """
#
#     :param y_true:
#     :param y_pred: is a vector of soft assignments. actual y_pred = max(q)
#     :return:
#     """
#     q_pred = y_pred
#     # y_pred =  q_pred.argmax(1) # TODO
#     y_pred = K.argmax(q_pred, axis=1)
#     return categorical_accuracy(y_true, y_pred)

def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_pred = y_pred.argmax(1)
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size



# class SoftAssignment(Layer):
#
#     def __init__(self, activation, n_clusters, input_dim, **kwargs):
#         super(SoftAssignment, self).__init__(activation, **kwargs)
#         self.__name__ = 'SoftAssignment'
#         self.clusters = self.add_weight((n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
#
# def cluster_assignment(z, alpha, clusters):
#     """
#
#     :param z:
#     :return:
#     """
#
#     q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(z, axis=1) - clusters), axis=2) / alpha))
#     q **= (alpha + 1.0) / 2.0
#     q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
#
#
# get_custom_objects().update({'SoftAssignment': SoftAssignment(cluster_assignment, n_clusters, input_dim)})

def get_p_distribution(q_distr, alpha):
    print(f'q d sh : {q_distr.shape}')
    weight = q_distr ** 2 / q_distr.sum(axis=0)
    print(f'weight {weight.shape}')

    return (weight.T / weight.sum(axis=1)).T

def get_target_distr(q_distr, alpha):
    weight = q_distr ** 2 / K.sum(q_distr, axis=0)
    return K.transpose(K.transpose(weight)/ K.sum(q_distr, axis=1))
# input_ = Input(shape=(original_dim,))

q = SoftAssignment( n_clusters, alpha=1)(encoder.output)
# target_distr = Lambda(lambda q_distr: get_p_distribution(q_distr, alpha=1))(q)
# custom loss
# custom accuracy

print(encoder.summary())
dec = Model(encoder.input, q)
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# dec.compile(optimizer='sgd', loss='kld',  metrics=[])
dec.compile(optimizer=sgd, loss=distribution_divergence_loss)
for i in range(30):
    history = dec.fit(x_train, labels_train, epochs=10, batch_size=128, validation_data=(x_test, labels_test),
                      verbose=2)
    predicted = dec.predict(x_test)
    acc_ = acc(labels_test, predicted)
    print(f'acc: {acc_}')

#
# maxiter = 20000
# update_interval = 140
# batch_size = 128
# index = 0
# index_array = np.arange(x_train.shape[0])
# print(dec.summary())
# history = {'acc': []}
# for ite in range(int(maxiter)):
#     if ite % update_interval == 0:
#         q = dec.predict(x_train, verbose=0)
#         p = get_p_distribution(q, alpha=1)  # update the auxiliary target distribution p
#         # evaluate the clustering performance
#         y_pred = q.argmax(1)
#         if labels_train is not None:
#             acc = np.round(metrics.acc(labels_train, y_pred), 5)
#             history['acc'].append(acc)
#
#     idx = index_array[index * batch_size: min((index+1) * batch_size, x_train.shape[0])]
#     loss = dec.train_on_batch(x=x_train[idx], y=p[idx])
#     index = index + 1 if (index + 1) * batch_size <= x_train.shape[0] else 0



# history = dec.fit(x_train, labels_train, epochs=80, batch_size=128, validation_data=(x_test, labels_test), verbose=2)

# plt.plot(history['acc'])
# # plt.plot(history['val_class_prediction_accuracy'])
# plt.title('DEC loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train'], loc='upper left')
# plt.show()

