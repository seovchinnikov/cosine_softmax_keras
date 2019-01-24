from keras import backend as K, regularizers, initializers
from keras.layers import Layer


class CosineSoftmax(Layer):
    def __init__(self, output_dim, kernel_initializer=initializers.glorot_uniform(), kernel_regularizer=None,
                 k_initializer=initializers.Constant(value=0.), k_regularizer=regularizers.l2(1e-1), **kwargs):
        super(CosineSoftmax, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.k_initializer = k_initializer
        self.k_regularizer = k_regularizer
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer

    def build(self, input_shape):
        self.k = self.add_weight(name='k',
                                 shape=(),
                                 initializer=initializers.get(self.k_initializer),
                                 regularizer=regularizers.get(self.k_regularizer),
                                 trainable=True)
        self.kernel = self.add_weight(name='w_i',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer=initializers.get(self.kernel_initializer),
                                      regularizer=regularizers.get(self.kernel_regularizer),
                                      trainable=True)
        super(CosineSoftmax, self).build(input_shape)

    def call(self, x, **kwargs):
        return K.softmax(K.softplus(self.k) * K.dot(K.l2_normalize(x, axis=1), K.l2_normalize(self.kernel, axis=0)))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'k_regularizer': regularizers.serialize(self.k_regularizer),
            'k_initializer': initializers.serialize(self.k_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'kernel_initializer': initializers.serialize(self.kernel_initializer)
        }
        base_config = super(CosineSoftmax, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
