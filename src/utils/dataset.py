import sys
import time
import os
import config
import copy
import geopandas as gpd
import numpy as np
import random
import tensorflow as tf

from scipy.spatial import distance_matrix


def get_vertices(folder, file_name, save=True, has_objects=True):
    """Returns vertices of data point cloud in given folder."""
    print('\n∎ Collecting vertices')
    print('\t▹ Reading file <{}.shp> from <{}>'.format(file_name, folder))
    folder_path = config.DATA_ROOT + folder

    # Extract file given file name.
    files = os.walk(folder_path).__next__()[-1]
    mask = np.array([(file_name in word) and ('.shp' in word) and (not '.shp.' in word) for word in files])
    file = files[np.where(mask == True)[0][0]]

    # Read file.
    data = gpd.read_file(os.path.join(folder_path, file))

    print('\t▹ Organizing as array')
    # Setup toolbar.
    toolbar_width = 10
    sys.stdout.write("\t[%s]" % (" " * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width + 1))  # return to start of line, after '['
    tot_it = 1 if data.shape[0] < toolbar_width else data.shape[0] // 10

    vertx = np.zeros([data.shape[0], 3])  # define array of XYZ point data
    labels, objects_ids = [], []

    cnt = 0
    for index, geom in data.iterrows():
        if cnt % tot_it == 0:
            time.sleep(0.1)
            sys.stdout.write("∙")
            sys.stdout.flush()
        cnt += 1

        vertx[index] = np.array([geom.geometry.x, geom.geometry.y, geom.geometry.z])

        if has_objects:
            # make sure that a vertex with obj_id has a label (>0,
            # indicating an object, not background).
            if (geom.objects < 0) and (geom.labels > 0):
                # obj_id not know, but label is known, so consider the
                # vertex as background (0).
                labels.append(0)
                objects_ids.append(geom.objects)
            else:
                labels.append(geom.labels)
                objects_ids.append(geom.objects)

    labels = np.array(labels)
    objects_ids = np.array(objects_ids)

    sys.stdout.write("]\n")  # this ends the progress bar
    print('○ {} vertices collected.'.format(data.shape[0]))

    if has_objects:
        # Reformat labels to make it starts from 1 (not 0) and
        # to replace empty values (-9999) by 0.
        objects_ids[objects_ids > -1] = objects_ids[objects_ids > -1] + 1
        objects_ids[objects_ids < 0] = 0.

    if save is True:
        np.save(os.path.join(folder_path, 'vertices'), vertx)
        np.save(os.path.join(folder_path, 'object_labels'), labels)
        np.save(os.path.join(folder_path, 'object_ids'), objects_ids)

    return vertx, labels, objects_ids


def get_edges(folder, file_name, save=True):
    """Return geometries in array form."""
    print('\n∎ Collecting edges')
    print('\t▹ Reading file <{}.shp> from <{}>'.format(file_name, folder))
    folder_path = config.DATA_ROOT + folder

    # Extract file given file name.
    mask = np.array([(file_name in word) and ('.shp' in word) and (not '.shp.' in word) for word in
                     os.walk(folder_path).__next__()[-1]])
    files = os.walk(folder_path).__next__()[-1]
    file = files[np.where(mask == True)[0][0]]

    # Read file.
    data = gpd.read_file(os.path.join(folder_path, file))

    print('\t▹ Organizing as array')
    # Setup toolbar.
    toolbar_width = 10
    sys.stdout.write("\t[%s]" % (" " * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width + 1))  # return to start of line, after '['
    tot_it = 1 if data.shape[0] < toolbar_width else data.shape[0] // 10

    # Define array of XYZ point data.
    array = np.zeros([data.shape[0], 4])

    cnt = 0
    for index, geom in data.iterrows():
        if cnt % tot_it == 0:
            time.sleep(0.1)
            sys.stdout.write("∙")
            sys.stdout.flush()
        cnt += 1

        tmp = geom.geometry.coords.xy
        tmp = np.array([tmp[0][0], tmp[1][0], tmp[0][1], tmp[1][1]])
        array[index] = tmp

    sys.stdout.write("]\n")  # this ends the progress bar
    print('○ {} edges collected.'.format(data.shape[0]))

    if save is True:
        path = os.path.join(folder_path, 'edges')
        np.save(path, array)
    return array


def reload_data(folder, data):
    root_path = config.DATA_ROOT + folder
    return [np.load(os.path.join(root_path, name + '.npy'), allow_pickle=True) for name in data]


def format_edges(vertices, edges, name, folder, save=True):
    """Return geometries in array form."""
    assert edges.shape[1] == 4

    # Convert data as <int32> type.
    vertices_f = vertices.astype(np.float32)
    edges_f = edges.astype(np.float32)

    print('\n∎ Collecting edge vertices')
    # Setup toolbar.
    toolbar_width = 10
    sys.stdout.write("\t[%s]" % (" " * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width + 1))  # return to start of line, after '['
    tot_it = 1 if vertices_f.shape[0] < toolbar_width else vertices_f.shape[0] // 10

    # Define new edges with corresponding vertices.
    array = np.zeros(shape=(edges_f.shape[0], 2), dtype=np.int32)

    for vindex, vcoords in enumerate(vertices_f):
        # Vertex id <vindex> must start from 0 to process
        # sparse adjacency matrix.
        if vindex % tot_it == 0:
            time.sleep(0.1)
            sys.stdout.write("∙")
            sys.stdout.flush()
        x, y, _ = vcoords

        # Increment over edge columns.
        for cindex, i in enumerate(range(0, 3, 2)):
            # check and replace edge tips whose coordinates match given vertex' coordinates
            xmatch = np.equal(x, edges_f[:, i]).astype(np.int8)
            ymatch = np.equal(y, edges_f[:, i + 1]).astype(np.int8)
            match = (xmatch * ymatch) * vindex
            array[:, cindex] = array[:, cindex] + match

    sys.stdout.write("]\n")  # this ends the progress bar
    print('○ {} formated edges.'.format(array.shape[0]))

    # Make sure no edge contains same vertex twice.
    if np.sum(np.equal(array[:, 0], array[:, 1]).astype(np.int8)) > 0:
        print("!!! Some edges have same vertex twice.")

    if save is True:
        path = os.path.join(config.DATA_ROOT + folder, name)
        np.save(path, array)
    return array


def get_batch_pointset(vertices, point_sets, batch_keys, batch_size):
    """Returns formatted pointset batch as input to the
    network in the format [B,H,W,C] or (batch_size, 1, npoints, 3)
    where npoints is the maximum number of points extracted per
    voxel."""
    point_set = [vertices[point_sets[key]] for key in batch_keys]
    non_zero_limits = [pset.shape[0] for pset in point_set]
    npoints_max = max(non_zero_limits)  # get maximum npoints

    sparse_point_set = np.zeros(shape=(batch_size, 1, npoints_max, 3))
    for i, pset in enumerate(point_set):
        sparse_point_set[i, 0, :pset.shape[0], :] = pset
    return sparse_point_set


def get_batch_pointset_per_batch(vertices, point_sets, batch_keys):
    """Returns formatted pointset batch as input to the
    network in the format [B,H,W,C] or (batch_size, 1, npoints, 3)
    where npoints is the maximum number of points extracted per voxel."""
    output, npoints_max, points = [], [], []
    for row in batch_keys:
        point_set = [vertices[point_sets[key]] for key in row]
        non_zero_limits = [pset.shape[0] for pset in point_set]
        npoints_max.append(max(non_zero_limits))  # get maximum npoints
        points.append(point_set)

    NP_max = max(npoints_max)
    for pts in points:
        sparse_point_set = np.zeros(shape=(batch_keys.shape[1], NP_max, 3))
        for i, p in enumerate(pts):
            sparse_point_set[i, :p.shape[0], :] = p
        output.append(sparse_point_set)

    return np.array(output)  # (B, P, NP, C)


def transform_geom(points, coord_reference, name, folder, save=True):
    """Transform points coordinates given <coord_reference>"""
    # get point cloud reference origin
    coord_min = np.amin(coord_reference, axis=0).tolist()
    coord_offset = np.asarray(coord_min[:points.shape[1]])

    # re-define points coords. given their distance to origin, and normalize by voxel size
    coord_trans = (points - coord_offset)
    if save is True:
        path = os.path.join(config.DATA_ROOT + folder, name)
        np.save(path, coord_trans)

    text = ', '.join([str(i) for i in np.amax(coord_trans, axis=0)])
    print('\n○ Max dimensions of transformed vertices (xyz):', text)
    return coord_trans


def voxelize(points, voxel_size):
    """Returns volexized point cloud over XY dimensions; not Z
    because of data type (topography). It is used to make input
    sets of vertices homogeneous in space (plane XY)."""
    xyz = points // voxel_size
    xy = xyz[:, :2].astype(np.int32)

    voxels, voxel_keys = {}, []
    for pidx in range(len(xy)):
        # Assign a key name given xy coord. normalized.
        key = ''.join([str(j) for j in xy[pidx]])  # (9172)
        if key in voxels:
            # Group points with same key name.
            voxels[key].append(pidx)
        else:
            voxels[key] = [pidx]
            voxel_keys.append(key)  # to return keys

    # Get representative xyz center of voxel.
    voxel_centers = []
    for key in voxels:
        # Get random point index from voxel (which represents
        # its center).
        center_idx = random.choice(voxels[key])
        voxel_centers.append(center_idx)
    return np.array(voxel_centers)


def get_centers(labels, vertices, folder, file_name, save=True):
    """Returns center geometries in array form."""
    path = os.path.join(config.DATA_ROOT + folder, file_name + '.shp')
    data = gpd.read_file(path)
    array = np.zeros([data.shape[0], 2])  # define array of XYZ point data

    for index, geom in data.iterrows():
        tmp = np.array([geom.geometry.coords.xy[0][0], geom.geometry.coords.xy[1][0]])
        array[index] = tmp
    print('○ {} centers collected.'.format(data.shape[0]))

    # Add 1 more column.
    padded = np.pad(array, ((0, 0), (0, 1)))  # (npoints, 3)
    for row, lbl in enumerate(np.unique(labels)[1:]):
        # Get Z mean.
        idx = np.where(labels == lbl)[0]
        obj_points = vertices[idx]
        padded[row][-1] = obj_points.mean(axis=0)[-1]

    if save is True:
        path = os.path.join(config.DATA_ROOT + folder, 'centers')
        np.save(path, padded)
    return padded


def get_sets(edges, ids, voxel_sets, key, hops):
    """Recursively retrieves connections between vertices at
    successive steps for a given key."""
    if hops > 0:
        tmp = []
        for i in ids:
            # Get 1st level.
            rows = np.where(edges == i)[0]
            rows = np.unique(edges[rows])
            voxel_sets[key] = np.hstack([voxel_sets[key], rows[rows != i]]).astype(np.int32)
            tmp.append(rows)
        get_sets(edges=edges, ids=np.hstack(tmp), voxel_sets=voxel_sets, key=key, hops=hops - 1)
    else:
        return voxel_sets


def normalize_point_number_in_batch(inkeys):
    """Makes the size of point batches same."""
    # Option (1) Repeat data by random selection from existing
    # batch.
    maxp = np.max([len(i) for i in inkeys])
    outkeys = []
    for array in inkeys:
        if len(array) < maxp:
            # missing data size
            margin = maxp - len(array)

            # Method (1): mirroring
            # WARNING: prone to errors because <array> possibly less
            # than margin in size.
            # extra = array[-margin:][::-1]

            # Method (2): random selection
            # WARNING: NOTE THAT <sample_group_by_radius> GETS <batch_indices>
            # GIVEN BATCH ORDERING, SO THIS METHOD MAY AFFECT LEARNING.
            extra = np.random.choice(array, size=margin, replace=True)

            # stack
            extra = np.hstack([array, extra])
            outkeys.append(extra)
        else:
            outkeys.append(array)

    return np.vstack(outkeys)


def get_graph_sets_by_squared_zoning(voxel_keys, vertices_xy, spl_radius, win_radius, center_xy):
    """Returns pointset per given voxel given a squared zone"""
    tmp_keys, tmp_sets = [], []
    for cent in center_xy:
        voxel_xy = vertices_xy[voxel_keys]

        # --- (1) Get new batch keys.
        # Get origin and max from object center coord. (i.e. origin_xy).
        win_min_xy = cent - win_radius
        win_max_xy = cent + win_radius

        origin, max_xy = np.squeeze(win_min_xy), np.squeeze(win_max_xy)

        # Assign color to vertices within box predicted.
        new_batch_keys = []
        for i, xy in enumerate(voxel_xy):
            if (all(xy <= max_xy) is True) and (all(xy >= origin) is True):
                new_batch_keys += [voxel_keys[i]]
        new_batch_keys = np.array(new_batch_keys)

        # --- (2) Get pointset by radius per batch key-
        voxel_sets = {}

        # Calculate distance between a voxel coords and those of
        # other nodes.
        voxel_xy = vertices_xy[new_batch_keys]
        distances = distance_matrix(voxel_xy, voxel_xy)

        # Extract nodes withing a given distance from the
        # voxel (neighborhood).
        mask = (distances < spl_radius).astype('int32')
        mask[mask == 0] = -1  # make 0s negative
        indices = (mask * new_batch_keys) + mask  # ...then add <mask> to make any 0 in <batch_keys> negative

        for i, row in enumerate(indices):
            pointset = row[row >= 0] - 1  # ...because here, we keep non-negative indices (hence 0 would always be kept)
            voxel_sets[new_batch_keys[i]] = pointset

        tmp_keys += [new_batch_keys]
        tmp_sets += [voxel_sets]

    # Merge dictionaries.
    sets = {key: val for d in tmp_sets for key, val in d.items()}

    # Normalize batch sizes.
    norm_keys = normalize_point_number_in_batch(tmp_keys)

    return norm_keys, sets  # (B, P), dict


def point_batching(centers, voxel_xy, batch_size=2):
    """Returns point cloud batches."""
    BS = batch_size - 2  # batch size minus 2 (because 2 is standard for positives + negatives)

    # always select positive & negative indices.
    pos_idx = np.random.choice(centers, size=1)[0]

    neg_centers = np.arange(voxel_xy.shape[0])
    neg_idx = np.random.choice(neg_centers, size=1)[0]

    # Then, add extra batches (positive and/or negative) if required.
    out_centers = [voxel_xy[pos_idx], voxel_xy[neg_idx]]
    if BS != 0:
        for i in range(BS):
            tmp_data = np.random.choice([centers, neg_centers], size=1)
            tmp_idx = np.random.choice(tmp_data, size=1)[0]
            out_centers.append(voxel_xy[tmp_idx])

    return out_centers


def get_graph_sets_by_squared_zoning2(voxel_keys, voxel_centers, voxel_xy,
                                      vertices_xy, spl_radius, win_radius, ratio_thresh):
    """Returns pointset per given voxel given a squared zone."""
    while True:
        # Get random voxel centers XY.
        centers_xy = point_batching(voxel_centers, voxel_xy, batch_size=2)

        tmp_keys, tmp_sets = [], []
        max_point_set_size = 0
        is_looping = False
        for cent in centers_xy:
            # --- (1) Get new batch keys
            # Get origin and max from object center coordinates (i.e. origin_xy).
            win_min_xy = cent - win_radius
            win_max_xy = cent + win_radius

            origin, max_xy = np.squeeze(win_min_xy), np.squeeze(win_max_xy)

            # Assign color to vertices within box predicted.
            new_batch_keys = []
            for i, xy in enumerate(voxel_xy):
                if (all(xy <= max_xy) is True) and (all(xy >= origin) is True):
                    new_batch_keys += [voxel_keys[i]]
            new_batch_keys = np.array(new_batch_keys)

            # Check size of point set.
            max_point_set_size = len(new_batch_keys) if (
                    len(new_batch_keys) > max_point_set_size) else max_point_set_size
            ratio = len(new_batch_keys) / max_point_set_size
            if ratio < ratio_thresh:
                # if the size of current point set is less than threshold, stop the loop and restart sampling
                is_looping = True
                break

            # --- (2) Get pointset by radius per batch key.
            voxel_sets = {}

            # Calculate distance between a voxel coords and those of other nodes.
            voxel_xy = vertices_xy[new_batch_keys]
            distances = distance_matrix(voxel_xy, voxel_xy)

            # Extract nodes withing a given distance from the voxel (neighborhood).
            mask = (distances < spl_radius).astype('int32')
            mask[mask == 0] = -1  # make 0s negative
            indices = (mask * new_batch_keys) + mask  # ...then add <mask> to make any 0 in <batch_keys> negative

            for i, row in enumerate(indices):
                pointset = row[
                               row >= 0] - 1  # ...because we keep non-negative indices (hence 0 would always be kept)
                voxel_sets[new_batch_keys[i]] = pointset

            tmp_keys += [new_batch_keys]
            tmp_sets += [voxel_sets]

        if not is_looping:
            break  # break out of outer loop, do not repeat sample selection

    # Merge dictionaries.
    sets = {key: val for d in tmp_sets for key, val in d.items()}

    # Normalize batch sizes.
    norm_keys = normalize_point_number_in_batch(new_batch_keys)

    return norm_keys, sets  # (B, P), dict


def squared_dist(A, B):
    expanded_a = tf.expand_dims(A, 2)  # (B, 256, 1, 32)
    expanded_b = tf.expand_dims(B, 1)  # (B, 1, 256, 32)
    distances = tf.reduce_sum(tf.compat.v1.squared_difference(expanded_a, expanded_b), 3)
    return tf.math.sqrt(distances)


def squared_dist_np(A, B):
    expanded_a = np.expand_dims(A, 2)  # (B,256,1,32)
    expanded_b = np.expand_dims(B, 1)  # (B,1,256,32)
    distances = np.sum(np.square(expanded_a - expanded_b), axis=3)
    return np.sqrt(distances)


def convert_angle_to_class(angle, eps=1e-10):
    """Returns convert angles to discrete class and residuals.
    eps: to avoid reaching MAX_ANGLE and getting 10 classes, instead
    of 9 (as indicated by angle bins)"""
    angle_increment = config.MAX_ANGLE / float(config.ANGLE_BINS)
    angle_classes = (angle - eps) / angle_increment
    angle_classes = angle_classes.astype(np.int32)
    residual_angles = angle - (angle_classes * angle_increment)
    return angle_classes, residual_angles


def rotate_point_cloud(points, angles):
    """Returns pointset rotated at given angles."""
    # Rotate in-place around Z axis.
    rotation_angle = np.random.choice(angles) * (np.pi / 180)
    sinval, cosval = np.sin(rotation_angle), np.cos(rotation_angle)
    rotation_matrix = np.array([[cosval, sinval, 0],
                                [-sinval, cosval, 0],
                                [0, 0, 1]])
    ctr = points.mean(axis=0)
    rotated_data = np.dot(points - ctr, rotation_matrix) + ctr
    return rotated_data


def flip_xy_axes(points):
    """Returns pointset flipped around x and y axes."""
    def flip(data):
        new_points = copy.deepcopy(data)
        x, y = copy.deepcopy(new_points[:, 0]), copy.deepcopy(new_points[:, 1])
        new_points[:, 0] = y
        new_points[:, 1] = x
        return new_points

    def no_flip(data):
        return data

    func = np.random.choice([flip, no_flip])
    return func(points)
