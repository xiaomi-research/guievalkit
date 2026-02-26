import math
import numpy as np


_SWIPE_DISTANCE_THRESHOLD = 0.04


def is_tap_action(normalized_start_yx, normalized_end_yx):
    distance = np.linalg.norm(np.array(normalized_start_yx) - np.array(normalized_end_yx))
    return distance <= _SWIPE_DISTANCE_THRESHOLD


def _get_direction(point1, point2):
    """
    Calculate the cardinal direction from point1 to point2.
    Uses magnitude comparison for efficiency.
    """
    try:
        vx = point2["x"] - point1["x"]
        vy = point2["y"] - point1["y"]
    except (KeyError, TypeError):
        return "no direction"

    if vx == 0 and vy == 0:
        return "no direction"

    # Compare magnitude of change to determine primary axis
    if abs(vx) > abs(vy):
        return "right" if vx > 0 else "left"
    else:
        return "down" if vy > 0 else "up"


def get_direction(point, to):

    if isinstance(to, str):
        if to in ["up", "down", "left", "right"]:
            return to
        else:
            return "no direction"
    elif isinstance(to, list):
        try:
            point1 = {"x": point[0], "y": point[1]}
            point2 = {"x": to[0], "y": to[1]}
            return _get_direction(point1, point2)
        except Exception:
            return "no direction"


def _resize_annotation_bounding_boxes(annotation_position, width_factor=1.2, height_factor=1.2):

    def _resize(box):
        y, x, h, w = box
        h_delta = (height_factor - 1) * h
        w_delta = (width_factor - 1) * w
        y = max(0, y - h_delta / 2)
        x = max(0, x - w_delta / 2)
        h = min(1, h + h_delta)
        w = min(1, w + w_delta)
        return [y, x, h, w]

    if not annotation_position:
        return []

    if isinstance(annotation_position[0], list):
        return [_resize(b) for b in annotation_position]

    return _resize(annotation_position)


def check_inside(x, y, bbox_list):
    """
    Check if point (x, y) is inside any bounding box in bbox_list.
    Optimized with vectorized NumPy operations.
    """
    if not bbox_list:
        return False, None
        
    bbox_array = np.asarray(bbox_list)
    # bbox format: [y_min, x_min, height, width]
    y_min, x_min = bbox_array[:, 0], bbox_array[:, 1]
    y_max, x_max = y_min + bbox_array[:, 2], x_min + bbox_array[:, 3]

    mask = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
    
    if np.any(mask):
        return True, bbox_array[mask]
    return False, None


def obtain_gt_bbox(coordinate, bbox_list, eval_android_control=False):
    """
    Retrieve ground truth bounding boxes relevant to the given coordinate.
    """
    if not bbox_list:
        return []
        
    x, y = coordinate['x'], coordinate['y']

    if not eval_android_control:
        is_inside, bbox_inside = check_inside(x, y, bbox_list)
        return bbox_inside.tolist() if is_inside else []
    
    # For Android Control, return top 5 closest boxes by center distance
    bbox_array = np.asarray(bbox_list)
    centers_y = bbox_array[:, 0] + bbox_array[:, 2] / 2
    centers_x = bbox_array[:, 1] + bbox_array[:, 3] / 2
    
    distances_sq = (centers_y - y)**2 + (centers_x - x)**2
    closest_indices = np.argsort(distances_sq)[:5]
    
    return bbox_array[closest_indices].tolist()
