# import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
import skimage.draw as draw


def vis_polys(ax, img, poly, title=''):
    h, w = img.shape[:2]
    ax.imshow(img, aspect='equal')
    patch_poly = patches.Polygon(poly, alpha=0.6, color='blue')
    ax.add_patch(patch_poly)
    poly = np.append(poly, [poly[0, :]], axis=0)
    #
    ax.plot(poly[:, 0] * w, poly[:, 1] * h, '-o', linewidth=2, color='orange')
    # first point different color
    ax.plot(poly[0, 0] * w, poly[0, 1] * h, '-o', linewidth=3, color='blue')
    ax.set_title(title)
    ax.axis('off')


def draw_poly(mask, poly):
    """
    Draw a polygon in the img.

    Args:
    img: np array of type np.uint8
    poly: np array of shape N x 2
    """
    cv2.fillPoly(mask, [poly], 255)

    return mask


def polygon_perimeter(polygon, img_side=28):
    """
    Generate the perimeter of a polygon including the vertices.
    """
    # Create empty image
    img_shape = [img_side, img_side]
    img = np.zeros(img_shape, dtype=np.float32)

    prev_idx, cur_idx = -1, 0
    poly_len = len(polygon)
    while cur_idx < poly_len:
        # Get vertices
        prev_vertex = polygon[prev_idx]
        cur_vertex = polygon[cur_idx]

        # Get line pixels
        prev_rr, prev_cc = draw.line(
            prev_vertex[1], prev_vertex[0],
            cur_vertex[1], cur_vertex[0]
        )
        # Draw lines
        img[prev_rr, prev_cc] = 1.

        # Increment prev_idx and cur_idx
        prev_idx += 1
        cur_idx += 1

    return img
