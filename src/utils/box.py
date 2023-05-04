import config
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import object_extraction as ext
import matplotlib.patches as patches

from shapely.geometry import Polygon
from shapely import affinity
from src.utils.dataset import convert_angle_to_class


def get_boxes_2D(vertices, object_ids, kernel=1, display=True):
    """Returns center points, box parameters and vertices as defined
    from a 2D map_.

    Get 2D bounding boxes parameters by projecting points over a map_.
    On this map_, we iteratively project each object with unique_ids to
    get a binary map_ (0s correspond to background, 1s to object). Then,
    we find contours and bbox of the projected object.
    """
    # Get unique object ids (except 0, which is background).
    unique_ids = np.unique(object_ids)[1:]

    # Build map_ given 1m resolution.
    coord_max = np.ceil(vertices.max(axis=0))
    row, clmn = np.arange(0., coord_max[1], 1.), np.arange(0., coord_max[0], 1.)  # minimum distance is 1m
    map_ = np.zeros(shape=(row.shape[0] + 1, clmn.shape[0] + 1))

    # Display map_ in new figure.
    if display is True:
        fig, ax = plt.subplots()

    # Look for contours per object via opencv by plotting it
    # alone (we reset map_ after an iteration).
    contours, maps = [], []
    for i in unique_ids:
        idx = np.where(object_ids == i)[0]
        obj_xy = vertices[idx]
        for j, coords in enumerate(obj_xy):
            x, y, _ = np.ceil(coords).astype(np.int32)
            # y = row_max - y
            map_[y - kernel:y + kernel, x - kernel:x + kernel] = 1.

        # Get object contours.
        con = ext.get_contours(map_, (3, 3), 120)
        contours.append(con)
        maps.append(map_)  # collect maps with single object

        # Reset map_.
        map_ = np.zeros(shape=(row.shape[0] + 1, clmn.shape[0] + 1))

    # Get map_ with all objects by take maximum over 1st axis.
    if display is True:
        map_ = np.max(np.array(maps), axis=0)
        ax.imshow(map_)

    # Then, plot the contours on this aggregated map_.
    centers_xy, box_params, box_vertices = [], [], []
    for i, con in enumerate(contours):
        # Take contour with the highest shape size (due to big, but also
        # small objects detected, because of map_ noise)
        if len(con[0]) > 1:
            sizes = [con[0][j].size for j in range(len(con))]
            idx = np.argmax(sizes)
        else:
            idx = 0

        # box: vertices of rectangle, box_coords: ((center), (size), angle)
        box, box_coords = ext.get_bounding_box_properties(con[0][idx], config.ROTATED_BOXES)

        # Find major axis given reference point.
        # Note: <p1> is always the lowest point of rectangle (does not matter
        # left or right) and is always the 1st sub-list of box array. Next
        # sub-lists represent points of rectangle taken in the clockwise direction
        # i.e. next point p2 (after p1) always represents the point that you first
        # get when you move in the clockwise direction.
        p1, p2, p3, p4 = box

        # Convert coordinates to centers.
        w = np.linalg.norm(p1 - p2)
        l = np.linalg.norm(p1 - p4)
        angle = np.abs(box_coords[2])  # in degree, in clockwise direction, given horizontal and p1-p4 segment
        xc, yc = box_coords[0]

        centers_xy.append([xc, yc])
        box_params.append([l, w, angle])
        box_vertices.append(box)  # always in [p1, p2, p3, p4] order

        if display is True:
            # Display corners.
            for bcoord in box:
                plt.plot(bcoord[0], bcoord[1], 'ro')

            # Display polygons.
            rect = patches.Polygon(box, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

            # Display object number.
            plt.text(xc, yc, str(i + 1), fontsize=12, color='red')
            plt.plot(xc, yc, 'ro', markersize=3)

    if display is True:
        plt.gca().invert_yaxis()  # invert map_ on the y-axis to get map_ view starting from (0,0)
        plt.show()

    return np.array(centers_xy), np.array(box_params), np.array(box_vertices)


def get_boxes_3D(centers_xy, box_vertices, object_ids, vertices, box_params):
    """Returns centroid, and box vertices, size and residuals."""
    # Get unique labels to look for unique objects.
    unique_ids = np.unique(object_ids)[1:]

    centers_xyz = np.pad(centers_xy, ((0, 0), (0, 1)))
    box_vertices_3D = np.zeros(shape=(box_vertices.shape[0], 8, 3))  # (nobjects, 8 box corners, xyz)

    new_box_params = []
    for cnt, i in enumerate(unique_ids):
        # NOTE: <i> starts from 1 because 0 is background in <unique_labels>, so we use <cnt> for iterating
        idx = np.where(object_ids == i)[0]
        obj_xyz = vertices[idx]
        zmax, zmin = np.max(obj_xyz, axis=0)[-1], np.min(obj_xyz, axis=0)[-1]

        h = zmax - zmin  # box height, plus 1 because the point at summit not accounted in bow otherwise
        centers_xyz[cnt][-1] = zmin + (h / 2)  # assign box center z coord

        new_params = box_params[cnt].tolist()
        new_params.insert(2, h)  # add z coord to 2nd position, s.t. (w, l, h, angle)
        new_box_params.append(new_params)

        # Add 3rd dimension of boxes given 2D coordinates and calculate
        # height (h).
        box_vertices_3D[cnt][:4, :2] = box_vertices[cnt]
        box_vertices_3D[cnt][4:, :2] = box_vertices[cnt]

        box_vertices_3D[cnt][:4, -1] = zmax
        box_vertices_3D[cnt][4:, -1] = zmin

    # Separate (l,w,h) sizes from angles (radian scalars).
    box_sizes, box_angles = np.array(new_box_params)[:, :-1], np.array(new_box_params)[:, -1]

    # get amgle classes and residuals
    box_angles_cls, box_angles_res = convert_angle_to_class(box_angles)

    # Make first row as 0s, which correspond to reference (unknown
    # center) for non-object points (background). Used to define ground
    # truth batches later via <gt_obj_ids>
    centers_xyz = np.pad(centers_xyz, ((1, 0), (0, 0)))
    box_vertices_3D = np.pad(box_vertices_3D, ((1, 0), (0, 0), (0, 0)))
    box_sizes = np.pad(box_sizes, ((1, 0), (0, 0)))
    box_angles_cls = np.pad(box_angles_cls, (1, 0))
    box_angles_res = np.pad(box_angles_res, (1, 0))

    return centers_xyz, box_vertices_3D, box_sizes, box_angles_cls, box_angles_res


def get_element(A, indices):
    """Return the ith element of indices from ith row of A."""
    one_hot_mask = tf.one_hot(indices, A.shape[1], on_value=True, off_value=False, dtype=tf.bool)
    return tf.boolean_mask(A, one_hot_mask)


def get_3D_box_corners(box_size, angle_cls, angle_res, center):
    """Returns box corner coordinates."""
    pos_size = tf.shape(box_size)[0]

    # Retrieve angle in degrees from class and residual.
    angle_increment = config.MAX_ANGLE / config.ANGLE_BINS

    # Counter-clockwise angles between horizontal and [p1, p4] axis.
    heading_angles = (angle_cls * angle_increment) + angle_res

    radian_fact = np.pi / 180
    c = tf.cos(heading_angles * radian_fact)  # (num_pos, )
    s = tf.sin(heading_angles * radian_fact)
    zeros = tf.zeros_like(c)
    ones = tf.ones_like(c)

    rotation = tf.reshape(tf.stack([c, s, zeros,
                                    -s, c, zeros,
                                    zeros, zeros, ones], -1),
                          tf.stack([pos_size, 3, 3]))

    l, w, h = box_size[..., 0], box_size[..., 1], box_size[..., 2]  # lwh(xzy) order!!!

    corners = tf.reshape(tf.stack([-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2,
                                   w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2,
                                   h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2], -1),
                         shape=tf.stack([pos_size, 3, 8]))

    rotated = tf.einsum('ijk,ikm->ijm', rotation, corners) + tf.expand_dims(center, axis=2)

    return tf.transpose(rotated, perm=[0, 2, 1])


def get_mean_iou_3d(gt_bbox, pred_bbox, return_mean=True):
    """Return iou of two 3d bbox."""
    mean_iou, ious = 0., []
    for b1, b2 in zip(gt_bbox, pred_bbox):
        # Get xy plane intersection.
        gt_poly_xz = Polygon(np.stack([b1[:4, 0], b1[:4, 1]], -1))  # takes coords. x, y of rect1
        pred_poly_xz = Polygon(np.stack([b2[:4, 0], b2[:4, 1]], -1))

        # It is possible that predicted corners are at the opposite side
        # of those from ground truth, so we rotate the predicted polygon.
        # Then, calculate intersection iteratively, and keep the best <iou>.
        tmp = []
        for angle in [0, 90, 180]:
            rot_poly = affinity.rotate(pred_poly_xz, angle, 'center')
            inter_area = gt_poly_xz.intersection(rot_poly).area

            # Calculate volume of intersecting plane from height (h).
            zmin = max(min(b1[:8, 2]), min(b2[:8, 2]))
            zmax = min(max(b1[:8, 2]), max(b2[:8, 2]))
            inter_h = max(0.0, zmax - zmin)
            inter_vol = inter_area * inter_h
            iou = inter_vol / (gt_poly_xz.area * (b1[0, 2] - b1[4, 2])
                               + pred_poly_xz.area * (b2[0, 2] - b2[4, 2]) - inter_vol)
            tmp.append(iou)

        # Add best iou for current bbox analyzed.
        mean_iou += max(tmp) / gt_bbox.shape[0]
        ious.append(max(tmp))

    if return_mean:
        return mean_iou  # mean IoU
    else:
        return ious  # list of IoUs
