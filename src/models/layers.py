import config
import tensorflow as tf

from tensorflow.keras.layers import Layer, BatchNormalization
from keras import regularizers


class Dense(Layer):
    """Module applying dense operation to pointset."""

    def __init__(self, out_dim, bias=True, activation=tf.nn.relu, batchnorm=False,
                 regularizer=config.WEIGHTS_REG, **kwargs):
        super(Dense, self).__init__(**kwargs)

        self.normalizer = None
        self.b = None
        self.w = None
        self.output_dim = out_dim
        self.activation = activation
        self.bias = bias
        self.batchnorm = batchnorm
        self.regularizer = regularizers.get(regularizer)

    def build(self, input_shape):
        with tf.compat.v1.variable_scope(self.name + '_vars'):
            # input shape: [filter_height, filter_width, in_channels, out_channels]
            # here, 1x1 conv. over input with channel (3,), filter_width=1 because <npoints> changes per batch
            self.w = self.add_weight(
                shape=(input_shape[-1], self.output_dim),
                initializer='glorot_uniform',
                regularizer=self.regularizer,
                trainable=True, name='weights')
            
            self.b = self.add_weight(
                shape=(self.output_dim,),
                initializer='random_normal',
                regularizer=self.regularizer,
                trainable=True, name='bias')
            
        self.normalizer = BatchNormalization()

    def call(self, inputs, training=True):
        output = tf.compat.v1.matmul(inputs, self.w)
        if self.bias:
            output = tf.add(output, self.b)
        if self.batchnorm is not False:
            output = self.normalizer(output, training=training)
        if self.activation is not False:
            output = self.activation(output)
        return output


class SharedConv(Layer):
    """Module applying 1D convolution to pointset."""

    def __init__(self, filters, strides=None, bias=True, activation=tf.nn.relu, padding='VALID', batchnorm=False,
                 regularizer=config.WEIGHTS_REG, **kwargs):
        super(SharedConv, self).__init__(**kwargs)

        self.normalizer = None
        self.b = None
        self.w = None
        if strides is None:
            strides = [1]

        self.filters = filters
        self.strides = strides
        self.bias = bias
        self.activation = activation
        self.padding = padding
        self.batchnorm = batchnorm
        self.regularizer = regularizers.get(regularizer)

    def build(self, input_shape):
        with tf.compat.v1.variable_scope(self.name + '_vars'):
            self.w = self.add_weight(
                shape=(1, input_shape[-1], self.filters),
                initializer='glorot_uniform',
                regularizer=self.regularizer,
                trainable=True, name='weights')
            
            self.b = self.add_weight(
                shape=(self.filters,),
                initializer='random_normal',
                regularizer=self.regularizer,
                trainable=True, name='bias')
            
        self.normalizer = BatchNormalization()

    def call(self, inputs, training=True):
        output = tf.nn.conv1d(inputs, filters=self.w, stride=self.strides, padding=self.padding, data_format='NWC')
        if self.bias:
            output = tf.add(output, self.b)
        if self.batchnorm is not False:
            output = self.normalizer(output, training=training)
        if self.activation is not False:
            output = self.activation(output)
        return output


class EdgeConvolution(Layer):
    """Module applying convolution over the node edges.
    See details in: https://arxiv.org/abs/1704.02901."""

    def __init__(self, units, bias=True, strides=None, activation=tf.nn.relu, padding='VALID', batchnorm=False,
                 regularizer=config.WEIGHTS_REG, **kwargs):
        super(EdgeConvolution, self).__init__(**kwargs)

        self.w = None
        self.b = None
        self.normalizer = None
        if strides is None:
            strides = [1]

        self.strides = strides
        self.padding = padding
        self.units = units
        self.bias = bias
        self.activation = activation
        self.batchnorm = batchnorm
        self.regularizer = regularizers.get(regularizer)

    def build(self, input_shape):
        with tf.compat.v1.variable_scope(self.name + '_vars'):
            self.w = self.add_weight(shape=(1, input_shape[-1], self.units),
                                     initializer='glorot_uniform',
                                     regularizer=self.regularizer,
                                     trainable=True, name='weights')
            self.b = self.add_weight(shape=(self.units,),
                                     initializer='random_normal',
                                     regularizer=self.regularizer,
                                     trainable=True, name='bias')
            self.normalizer = BatchNormalization()

    def call(self, inputs, adj_matrix=None, xidx=None, training=True):
        # Get points neighborhood.
        tiled_mask = tf.tile(tf.expand_dims(adj_matrix, axis=-1),
                             multiples=[1, 1, 1, tf.shape(inputs)[-1]])  # (B, P, P, ?)
        tiled_inputs = tf.tile(tf.expand_dims(inputs, axis=-2),
                               multiples=[1, 1, tf.shape(inputs)[1], 1])  # (B, P, P, ?)
        masked_inputs = tf.math.multiply(tiled_mask, tiled_inputs)  # (B, P, P, ?)

        # Edging.
        xi = tf.gather_nd(masked_inputs, tf.expand_dims(xidx, axis=2), batch_dims=2)  # (B, P, 1, ?)
        xi = tf.expand_dims(xi, axis=2)
        edging = tf.math.multiply(xi - masked_inputs, tiled_mask)  # (B, P, P, ?)
        edging = tf.concat([masked_inputs, edging], axis=-1)  # (B, P, P, ?)

        # Apply linear transform.
        output = tf.nn.conv1d(edging, filters=self.w, stride=self.strides, padding=self.padding, data_format='NWC')
        if self.bias: 
            output = tf.add(output, self.b)
        if self.batchnorm: 
            output = self.normalizer(output, training=training)
        output = self.activation(output)  # (B, P, P, K)

        # Reduce dimension.
        maxp = tf.reduce_max(output, axis=-2, keepdims=True)  # (B, P, 1, ?)
        nonzero = tf.math.reduce_any(tf.not_equal(output, 0.0), axis=-1, keepdims=True)  # True/False on last axis
        n = tf.reduce_sum(tf.cast(nonzero, 'float32'), axis=2, keepdims=True)  # how many True/False on axis -2
        avgpool = tf.reduce_sum(edging, axis=2, keepdims=True) / n  # (B, P, 1, ?)

        return tf.concat([maxp, avgpool], axis=-1)[:, :, 0, :]  # (B, P, ?)


class FullyConnected(Layer):
    """Module applying fully connected operations to pointset."""

    def __init__(self, filters, strides=None, bias=True, activation=tf.nn.relu, padding='VALID',
                 batchnorm=False, regularizer=config.WEIGHTS_REG, **kwargs):
        super(FullyConnected, self).__init__(**kwargs)

        self.w = None
        self.b = None
        self.normalizer = None
        if strides is None:
            strides = 1
        self.filters = filters
        self.bias = bias
        self.strides = strides
        self.activation = activation
        self.padding = padding
        self.batchnorm = batchnorm
        self.regularizer = regularizers.get(regularizer)

    def build(self, input_shape):
        with tf.compat.v1.variable_scope(self.name + '_vars'):
            # input shape: [filter_height, filter_width, in_channels, out_channels]
            # here, 1x1 convolution over input with channel (3,)
            self.w = self.add_weight(
                shape=(input_shape[-1], self.filters),
                initializer='glorot_uniform',
                regularizer=self.regularizer,
                trainable=True, name='weights')
            
            self.b = self.add_weight(
                shape=(self.filters,),
                initializer='random_normal',
                regularizer=self.regularizer,
                trainable=True, name='bias')
            
            self.normalizer = BatchNormalization()
            
            super(FullyConnected, self).build(input_shape)

    def call(self, inputs, training=True):
        output = tf.matmul(inputs, self.w)
        if self.bias:
            output = tf.add(output, self.b)
        if self.batchnorm is not False:
            output = self.normalizer(output, training=training)
        if self.activation is not False:
            output = self.activation(output)
        return output
