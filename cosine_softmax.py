from keras import backend as K, regularizers, initializers
from keras.layers import Layer


class CosineSoftmax(Layer):
    def __init__(self, output_dim, kernel_initializer=initializers.glorot_uniform(), kernel_regularizer=None,
                 k_initializer=initializers.Constant(value=1), k_regularizer=regularizers.l2(1e-1), **kwargs):
        self.output_dim = output_dim
        self.k_initializer = k_initializer
        self.k_regularizer = k_regularizer
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        super(CosineSoftmax, self).__init__(**kwargs)

    def build(self, input_shape):
        self.k = self.add_weight(name='k',
                                 shape=(),
                                 initializer=self.k_initializer,
                                 regularizer=self.k_regularizer,
                                 trainable=True)
        self.kernel = self.add_weight(name='w_i',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      trainable=True)
        super(CosineSoftmax, self).build(input_shape)

    def call(self, x, **kwargs):
        return K.softmax(self.k * K.dot(K.l2_normalize(x, axis=1), K.l2_normalize(self.kernel, axis=0)))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
