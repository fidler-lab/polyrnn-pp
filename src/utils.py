import numpy as np

_MAX_POLY_LEN = 142


def _poly0g_to_poly01(polygon, grid_side=28):
    """
    [0, grid_side] coordinates to [0, 1].

    Note: we add 0.5 to the vertices so that the lie in the middle of the cell.
    """
    result = (polygon.astype(np.float32) + 0.5) / grid_side

    return result


def _mask_polys(polys, masks):
    """
    Return masked polys.
    """
    new_polys = []
    for poly, mask in zip(polys, masks):
        cur_poly = poly[mask.astype(np.bool)]
        new_polys.append(cur_poly)

    return new_polys


def _poly01_to_index(polygon, grid_side=112):
    """
    Return poly index in a flat array.
    """
    result = []
    for item in polygon:
        result.append(item[0] + item[1] * grid_side)

    return result


def preprocess_ggnn_input(pred_01_poly):
    """
    Prepare data for GGNN
    """

    enhanced_poly = []
    for i in range(len(pred_01_poly)):
        if i < len(pred_01_poly) - 1:
            enhanced_poly.append(pred_01_poly[i])

            enhanced_poly.append(
                np.array(
                    [(pred_01_poly[i][0] + pred_01_poly[i + 1][0]) / 2,
                     (pred_01_poly[i][1] + pred_01_poly[i + 1][1]) / 2])
            )
        else:
            enhanced_poly.append(pred_01_poly[i])
            enhanced_poly.append(
                np.array(
                    [(pred_01_poly[i][0] + pred_01_poly[0][0]) / 2,
                     (pred_01_poly[i][1] + pred_01_poly[0][1]) / 2])
            )

    poly_for_feature_index = np.floor(np.array(enhanced_poly) * 112).astype(np.int32)
    feature_indexs = _poly01_to_index(poly_for_feature_index, 112)
    feature_indexs = np.array(feature_indexs)
    fwd_poly = np.floor(np.array(enhanced_poly) * 112).astype(np.int32)
    poly_len = len(fwd_poly)

    array_feature_indexs = np.ones(_MAX_POLY_LEN, np.float32) * 0.
    arr_fwd_poly = np.ones((_MAX_POLY_LEN, 2), np.float32) * -1.
    arr_mask = np.zeros(_MAX_POLY_LEN, np.int32)
    arr_fwd_poly[:poly_len] = fwd_poly
    arr_mask[:poly_len] = 1
    array_feature_indexs[:poly_len] = feature_indexs

    return np.array([array_feature_indexs]), np.array([arr_fwd_poly]), np.array([arr_mask])
