import tensorflow as tf

from abc import ABC
from layers import EdgeConvolution, FullyConnected, SharedConv, Dense
from utils.dataset import squared_dist


class Model(object):
    def __init__(self, **kwargs):
        """Define model scheme."""
        allowed_kwargs = {'name'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')

        if not name:
            name = self.__class__.__name__.lower()

        self.name = name
        self.vars = {}
        self.layers = []
        self.sequence = []
        self.inputs = None
        self.outputs = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """Wrapper for _build()."""
        with tf.compat.v1.variable_scope(self.name):
            self._build()

        # Sequential layer model.
        self.sequence.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.sequence[-1])  # call last layer of sequence by <layer>
            self.sequence.append(hidden)
        self.outputs = self.sequence[-1]

        # Model variables.
        variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}


class GCN(Model, ABC):
    """Module of the Graph Convolution Network (GCN) with edge convolution
    component (see https://arxiv.org/abs/1704.02901)."""

    def __init__(self, units, bias=True, activation=tf.nn.relu, batchnorm=False, **kwargs):
        super(GCN, self).__init__(**kwargs)
        self.gcn1 = EdgeConvolution(units[0], bias=bias, activation=activation, batchnorm=batchnorm)
        self.conv1 = SharedConv(units[1], activation=activation, bias=bias, batchnorm=batchnorm)
        self.fc1 = Dense(units[1], activation=activation, bias=bias, batchnorm=batchnorm)
        self.fc2 = Dense(units[2], activation=activation, bias=bias, batchnorm=batchnorm)

    def __call__(self, features, xyz, radius, istraining=True):
        # Get adjacency matrix.
        dist_matrix = squared_dist(xyz, xyz)  # (B, P, P)
        adj_matrix = tf.cast((dist_matrix <= radius), tf.float32)  # (B, P, P)

        # Get vertex center.
        x0 = tf.argmin(dist_matrix, axis=-1)  # (B, P, 1)

        # Hidden ops.
        hidden = self.gcn1(features, adj_matrix, x0, istraining)  # (B, P, ?)
        hidden = self.conv1(hidden)  # (B, P, ?)

        return hidden


class VotingModule(Model, ABC):
    """Voting module to estimate object centers from pointset."""

    def __init__(self, units, activation=tf.nn.relu, bias=True, batchnorm=False, **kwargs):
        super(VotingModule, self).__init__(**kwargs)

        self.layers = []
        self.activation = activation
        self.batchnorm = batchnorm

        # Build ops sequentially.
        for i, unit in enumerate(units):
            if i == len(units) - 1:
                tmp_act = lambda x: x  # does nothing, no activation
                tmp_bn = lambda x: x
            else:
                tmp_act = self.activation
                tmp_bn = self.batchnorm
            offset = SharedConv(
                unit,
                activation=tmp_act,
                bias=bias,
                batchnorm=tmp_bn,
                name='voting{}'.format(i))
            self.layers.append(offset)

    def __call__(self, inputs, training):
        offset = inputs
        for layer in self.layers:
            offset = layer(offset, training)
        return offset[0, :, :]  # (feat_dim+3, offset_dim)


class FullyConnectedSeq(Model, ABC):
    """Module applying sequence of fully-connected operations."""

    def __init__(self, units, bias=True, activation=tf.nn.relu, batchnorm=False, **kwargs):
        super(FullyConnectedSeq, self).__init__(**kwargs)

        self.activation = activation
        self.batchnorm = batchnorm
        self.bias = bias
        self.layers = []

        # Build ops sequentially.
        for i, unit in enumerate(units):
            if i == (len(units) - 1):
                tmp_act = lambda x: x  # does nothing, no activation
                tmp_bn = lambda x: x
            else:
                tmp_act = self.activation
                tmp_bn = self.batchnorm
            output = FullyConnected(unit, bias=self.bias, batchnorm=tmp_bn, activation=tmp_act,
                                    name='fc{}'.format(i))
            self.layers.append(output)

    def __call__(self, inputs, training):
        output = inputs
        for layer in self.layers:
            output = layer(output, training)
        return output
