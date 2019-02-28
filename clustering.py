from keras import Model
from keras.datasets import mnist
from keras.engine import Layer, InputSpec
import keras.backend as K
from keras.optimizers import SGD
from sklearn.cluster import KMeans

import metrics
from model import get_encoder
import numpy as np

(x_train, labels_train), (x_test, labels_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
sae_autoencoder, encoder = get_encoder()
class ClusteringLayer(Layer):


    def __init__(self, n_clusters: int=10,
                 weights=None, alpha=1, input_shape=None, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        # set clusters as a weights with dim: (10 - for mnist, # embedded input )
        self.clusters = self.add_weight((self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """
        calculates q distribution as a t-distribution of embedded vectors and clusters
        :param inputs:
        :param kwargs:
        :return:
        """
        # assert isinstance(inputs, list)
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class DEC(object):
    def __init__(self, dims, n_clusters=10, alpha=1.0, init='glorot_unifrom'):
        super(DEC, self).__init__()

        self.dims = dims
        self.input_dim = dims[0]
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.encoder = encoder

        clustering_layer = ClusteringLayer(self.n_clusters, name='clustering')(self.encoder.output)
        self.model = Model(inputs=encoder.input, outputs=clustering_layer)

    def load_weights(self, weights):
        self.model.load_weights(weights)

    def predict(self, input):
        q_distr = self.model.predict(input, verbose=0)
        return q_distr.argmax(1)

    @staticmethod
    def target_distribution(q_distr):
        weight = q_distr ** 2 / q_distr.sum(0)
        return (weight.T / weight.sum(1)).T

    def compile(self, optimizer='sgd', loss='kld'):
        self.model.compile(optimizer=optimizer, loss=loss)

    def fit(self, x, y=None, maxiter=2e6, batch_size=256, tol=1e-3,
            update_interval=140):
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        save_interval = int(x.shape[0] / batch_size) * 5  # 5 epochs
        y_pred = kmeans.fit_predict(self.encoder.predict(x))
        y_pred_last = np.copy(y_pred)
        self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])
        loss = 0
        index = 0
        index_array = np.arange(x.shape[0])
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                q = self.model.predict(x, verbose=0)
                p = self.target_distribution(q)  # update the auxiliary target distribution p

                # evaluate the clustering performance
                y_pred = q.argmax(1)

                # check stop criterion
                # delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = np.copy(y_pred)
                if y is not None:
                    acc = np.round(metrics.acc(y, y_pred), 5)
                    nmi = np.round(metrics.nmi(y, y_pred), 5)
                    ari = np.round(metrics.ari(y, y_pred), 5)
                    loss = np.round(loss, 5)
                    # logdict = dict(iter=ite, acc=acc, nmi=nmi, ari=ari, loss=loss)
                    print('Iter %d: acc = %.5f, nmi = %.5f, ari = %.5f' % (ite, acc, nmi, ari), ' ; loss=', loss)

                # check stop criterion
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]

                if ite > 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print('Reached tolerance threshold. Stopping training.')
                    break
            idx = index_array[index * batch_size: min((index + 1) * batch_size, x.shape[0])]
            loss = self.model.train_on_batch(x=x[idx], y=p[idx])
            index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0

            # save intermediate model
            if ite % save_interval == 0:
                print('saving model to:', 'DEC_model_' + str(ite) + '.h5')
                self.model.save_weights('DEC_model_' + str(ite) + '.h5')

        ite += 1

        # save the trained model

        print('saving model to:', 'DEC_model_final.h5')
        self.model.save_weights('DEC_model_final.h5')
        return y_pred


if __name__ == '__main__':
    n_clusters = 10
    init = 'glorot_uniform'
    dec = DEC(dims=[x_train.shape[-1], 500, 500, 2000, 10], n_clusters=n_clusters, init=init)
    dec.model.summary()
    dec.compile(optimizer=SGD(0.01, 0.9), loss='kld')
    y_pred = dec.fit(x_train, y=labels_train, maxiter=100)
    kek = 0