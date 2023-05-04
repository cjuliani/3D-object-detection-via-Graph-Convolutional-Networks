import sys

sys.path.append("..")  # Adds higher directory to python modules path.

import tensorflow as tf
import config


def sequential_convolution(features, filters, istraining, activated, batchnorm=config.BATCH_NORM):
    """Returns outputs from sequential convolution operations."""
    output = None
    for i, flt in enumerate(filters):
        output = tf.keras.layers.Conv1D(filters=flt, kernel_size=1, strides=1, padding='valid')(features)
        if batchnorm:
            output = tf.keras.layers.BatchNormalization()(output, training=istraining)
        if activated is False:
            if i < (len(filters) - 1):
                output = tf.keras.layers.ReLU()(output)
            else:
                pass  # do not activate
        else:
            output = tf.keras.layers.ReLU()(output)
    return output


def average_aggregation(inputs):
    """Returns normalized sampled points with respective
    hidden geometric features."""
    # Get average coordinates (over non-zero entries).
    nonzero = tf.math.reduce_any(tf.not_equal(inputs, 0.0), axis=-1, keepdims=True)  # (2,P,NP,3)
    n = tf.reduce_sum(tf.cast(nonzero, 'float32'), axis=2, keepdims=True)
    xyz = tf.reduce_sum(inputs, axis=2, keepdims=True) / n  # (B,P,1,3)

    # Normalization of xyz.
    xyz_max = tf.reduce_max(xyz, axis=1, keepdims=True)
    xyz_min = tf.reduce_min(xyz, axis=1, keepdims=True)
    xyz_factor = (xyz_max - xyz_min)

    # Get (euclidean) distance from centroid.
    offset = xyz[:, :1, :, :] - xyz  # (2,P,1,3)
    dist = tf.norm(offset, ord='euclidean', keepdims=True, axis=-1)  # (2,P,1,1)

    # Get height offset.
    h = xyz[:, :, :, -1:]
    hmin = tf.reduce_min(h, axis=1, keepdims=True)
    hmax = tf.reduce_max(h, axis=1, keepdims=True)
    hnorm = h / (hmax - hmin)  # (2,P,1,1)

    # Concatenate the offsets and distance as hidden
    # feature of points.
    hidden = tf.concat([hnorm, offset, dist], axis=-1)

    return hidden[:, :, 0, :], xyz[:, :, 0, :], xyz_factor[:, :, 0, :]  # (B,P,?)
