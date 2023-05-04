import config
import tensorflow as tf

from src.models.models import GCN
from src.utils.dataset import squared_dist
from src.models.utils import sequential_convolution


class PyramidGraphConvNet:
    def __init__(self, xyz, features, istraining):
        self.xyz = xyz
        self.features = features
        self.istraining = istraining

    def __call__(self):
        # Define GCN nodes.
        ptnet1 = GCN([128, 64, 32, 32], batchnorm=config.BATCH_NORM)
        ptnet2 = GCN([256, 128, 64, 64], batchnorm=config.BATCH_NORM)
        ptnet3 = GCN([512, 256, 128, 128], batchnorm=config.BATCH_NORM)
        ptnet4 = GCN([1024, 512, 256, 256], batchnorm=config.BATCH_NORM)
        ptnet6 = GCN([128, 64, 32, 32], batchnorm=config.BATCH_NORM)
        ptnet5 = GCN([256, 128, 64, 64], batchnorm=config.BATCH_NORM)

        # --- Stage 1
        points_feat = ptnet1(
            features=self.features,
            xyz=self.xyz,
            radius=config.GROUP_RADIUS[0],
            istraining=self.istraining)

        points_xyz1, points_feat1 = group_pooling(
            self.xyz, points_feat,
            radius=config.GROUP_RADIUS[0])

        # --- Stage 2
        points_feat1 = ptnet2(
            features=points_feat1,
            xyz=points_xyz1,
            radius=config.GROUP_RADIUS[1],
            istraining=self.istraining)

        points_xyz2, points_feat2 = group_pooling(
            points_xyz1, points_feat1,
            radius=config.GROUP_RADIUS[1])

        # --- Stage 3
        points_feat2 = ptnet3(
            features=points_feat2,
            xyz=points_xyz2,
            radius=config.GROUP_RADIUS[2],
            istraining=self.istraining)

        points_xyz3, points_feat3 = group_pooling(
            points_xyz2, points_feat2,
            radius=config.GROUP_RADIUS[2])

        # --- Stage 4 (bottleneck)
        points_feat0 = ptnet4(
            features=points_feat3,
            xyz=points_xyz3,
            radius=config.GROUP_RADIUS[3],
            istraining=self.istraining)

        # --- Stage 5
        points_feat5 = unpooling(
            ref_feat=points_feat2,
            feat=points_feat0,
            filters=256,
            istraining=self.istraining)

        points_feat5 = ptnet5(
            features=points_feat5,
            xyz=points_xyz2,
            radius=config.GROUP_RADIUS[2],
            istraining=self.istraining)

        # --- Stage 6
        points_feat6 = unpooling(
            ref_feat=points_feat1,
            feat=points_feat5,
            filters=128,
            istraining=self.istraining)

        points_feat6 = ptnet6(
            features=points_feat6,
            xyz=points_xyz1,
            radius=config.GROUP_RADIUS[1],
            istraining=self.istraining)

        # --- Stage 7
        return unpooling(
            ref_feat=points_feat,
            feat=points_feat6,
            filters=64 + 3,
            istraining=self.istraining,
            activated=False)


def unpooling(ref_feat, feat, istraining, filters=None, activated=True):
    """Returns up-sampled features."""
    upsampled = tf.keras.layers.UpSampling1D(size=2)(feat)  # (B, P*2, ?)

    # Make sure shapes are equal between reference feature and the
    # up-sampled one.
    shp1, shp2 = tf.shape(ref_feat)[1], tf.shape(upsampled)[1]
    upsampled = tf.cond(tf.not_equal(shp1, shp2), lambda: upsampled[:, :-1, :], lambda: upsampled)

    # Apply convolution.
    output = tf.concat([ref_feat, upsampled], axis=-1)
    if filters:
        output = sequential_convolution(output, filters, istraining, activated)

    return output


def group_pooling(pointset_xyz, pointset_feat, radius):
    """Returns pointset per given voxel given a distance radius on xy coordinates.
    Nodes nearby the voxel on xy plane, and within a given radius (in meters),
    are extracted."""
    assert radius > 0

    dist_matrix = squared_dist(pointset_xyz, pointset_xyz)  # (B, P, P)
    mask = tf.cast((dist_matrix <= radius), tf.int32)  # (B, P, P) of [0,1]

    # Make sure the shape is even (not odd) because the up-sampling is
    # done via kernel of 2, hence final tensor shape will be even.
    shp = tf.math.floordiv(tf.shape(mask)[1], 2)
    shp = tf.cond(tf.not_equal(tf.math.floormod(shp, 2), 0), true_fn=lambda: shp + 1, false_fn=lambda: shp)
    batch_indices = tf.range(
        start=0,
        limit=tf.shape(mask)[1],
        delta=2)
    batch_indices = tf.expand_dims(tf.expand_dims(batch_indices, axis=0), axis=-1)  # (1, P//2, 1)
    batch_indices = tf.tile(batch_indices, multiples=[tf.shape(mask)[0], 1, 1])  # (B, P//2, 1)

    cluster_xyz = tf.gather_nd(pointset_xyz, batch_indices, batch_dims=1)  # (B, P//2, 3)
    cluster_mask = tf.gather_nd(mask, batch_indices, batch_dims=1)  # (B, P//2, P)

    # Get masked features (assign by multiplication).
    shp = [1, 1, 1, tf.shape(pointset_feat)[-1]]
    cluster_mask = tf.tile(tf.expand_dims(
        tf.cast(cluster_mask, 'float32'), axis=-1), multiples=shp)  # (B, P//2, P, ?)

    shp = [1, tf.shape(cluster_mask)[1], 1, 1]
    pointset_feat_tiled = tf.tile(tf.expand_dims(pointset_feat, axis=1), multiples=shp)  # (B, P//2, P, ?)
    masked_hidden = tf.math.multiply(cluster_mask, pointset_feat_tiled)  # (B, P//2, P, ?)

    # Get pooling.
    maxpool = tf.math.reduce_max(masked_hidden, axis=2, keepdims=True)  # (B, P//2, 1, ?)
    nonzero = tf.math.reduce_any(tf.not_equal(masked_hidden, 0.0), axis=-1, keepdims=True)
    n = tf.reduce_sum(tf.cast(nonzero, 'float32'), axis=2, keepdims=True)
    avgpool = tf.reduce_sum(masked_hidden, axis=2, keepdims=True) / n  # (B, P//2, 1, ?)
    merge = tf.concat([maxpool, avgpool], axis=-1)[:, :, 0, :]  # (B, P//2, ?)

    # Residuals.
    output = tf.gather_nd(pointset_feat, batch_indices, batch_dims=1)  # (B, P//2, ?)
    output = tf.concat([merge, output], axis=-1)

    return cluster_xyz, output
