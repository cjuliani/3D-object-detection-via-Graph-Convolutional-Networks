import cv2
import numpy as np

from PIL import Image


def remove_transparency(im, bg_colour=(255, 255, 255)):
    """Remove alpha channel from RGBA image.
    Only process if image has transparency (http://stackoverflow.com/a/1963146)"""
    if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):
        # Need to convert to RGBA if LA format due to a bug in PIL (see http://stackoverflow.com/a/1963146)
        alpha = im.convert('RGBA').split()[-1]

        # Create a new background image of our matt color
        # Must be RGBA because paste requires both images have the same format
        # (see http://stackoverflow.com/a/8720632  and  http://stackoverflow.com/a/9459208)
        bg = Image.new("RGBA", im.size, bg_colour + (255,))
        bg.paste(im, mask=alpha)
        return bg.convert('RGB')

    else:
        # return image if no alpha channel
        return im


def get_contours(layer_ref, blur_coef, binary_thresh):
    """Get object contours after transforming <layer_ref> (i.e. segmentation or binary annotated map).
    :param layer_ref: feature map representation (with blob) to be cropped
    :param blur_coef: gaussian blur coefficient used prior to thresholding image
    :param binary_thrs: threshold value for image processing of predicted object
    :return: contour list <contours>, <binary_map> and <rgb_map>"""
    # set black and white image
    im = np.uint8(layer_ref * 255)

    # get image (with 3 bands) from array
    im = Image.fromarray(im)
    rgb_map = remove_transparency(im, bg_colour=(255, 255, 255))
    rgb_map = np.array(rgb_map)

    grayM = cv2.cvtColor(rgb_map, cv2.COLOR_GRAY2RGB)
    grayM = cv2.cvtColor(grayM, cv2.COLOR_RGB2GRAY)

    blur = cv2.GaussianBlur(grayM, blur_coef, 0)
    ret, binary_map = cv2.threshold(blur, binary_thresh, 255, 0)

    # Get contours / clusters defined from thresholding
    # in Opencv4 cv2.findContour() only returns 2 values: contours, hierachy
    contours = cv2.findContours(binary_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    return contours, binary_map


def get_bounding_box_properties(contours, rotated=False):
    """Return the bounding rectangle of contours."""
    if rotated:
        #  ( center(x,y), (width, height), angle of rotation )
        box_coords = cv2.minAreaRect(contours)
    else:
        x1, y1, w, l = cv2.boundingRect(contours)  # x1,y1 = the top left coordinate of rect.
        xc = x1 + (w / 2)  # center x
        yc = y1 + (l / 2)
        box_coords = ((xc, yc), (w, l), -0.0)

    # get the box vertices
    box = cv2.boxPoints(tuple(box_coords))  # Find four vertices of rectangle from above rect
    box = np.int0(np.around(box))  # Round the values and make it integers

    return box, box_coords
