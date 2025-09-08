import math
import numpy as np


_SWIPE_DISTANCE_THRESHOLD = 0.04


def is_tap_action(normalized_start_yx, normalized_end_yx):
    distance = np.linalg.norm(np.array(normalized_start_yx) - np.array(normalized_end_yx))
    return distance <= _SWIPE_DISTANCE_THRESHOLD


def _get_direction(point1, point2):

    try:
        x1, y1 = point1["x"], point1["y"]
        x2, y2 = point2["x"], point2["y"]
        assert x1 is not None
        assert x2 is not None
        assert y1 is not None
        assert y2 is not None
        vx, vy = (x2 - x1, y2 - y1)
    except Exception:
        return "no direction"

    directions = {
        "up": (0, -1),
        "down": (0, 1),
        "left": (-1, 0),
        "right": (1, 0)
    }

    vector_length = math.sqrt(vx ** 2 + vy ** 2)
    if vector_length == 0:
        return "no direction"
    unit_vector = (vx / vector_length, vy / vector_length)

    max_cosine = -float('inf')
    closest_direction = None
    for direction, dir_vector in directions.items():
        dx, dy = dir_vector
        dir_length = math.sqrt(dx ** 2 + dy ** 2)
        cos_theta = (unit_vector[0] * dx + unit_vector[1] * dy) / dir_length
        if cos_theta > max_cosine:
            max_cosine = cos_theta
            closest_direction = direction

    return closest_direction


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
    bbox_array = np.array(bbox_list)
    y_min, x_min, height, width = bbox_array[:, 0], bbox_array[:, 1], bbox_array[:, 2], bbox_array[:, 3]
    y_max, x_max = y_min + height, x_min + width

    within_x = (x_min <= x) & (x <= x_max)
    within_y = (y_min <= y) & (y <= y_max)
    within_bbox = within_x & within_y

    if np.any(within_bbox):
        within_bbox_coords = bbox_array[within_bbox]
        return True, within_bbox_coords
    else:
        return False, None


def obtain_gt_bbox(coordinate, bbox_list, eval_android_control=False):
    x, y = coordinate['x'], coordinate['y']
    if len(bbox_list) == 0:
        return []

    if not eval_android_control:
        is_inside, bbox_inside = check_inside(x, y, bbox_list)
        if is_inside:
            return bbox_inside.tolist()
        else:
            return []
    else:
        def get_center_distance(box):
            ymin, xmin, h, w = box
            center_y = ymin + h / 2
            center_x = xmin + w / 2
            return ((center_y - y) ** 2 + (center_x - x) ** 2) ** 0.5

        sorted_boxes = sorted(bbox_list, key=get_center_distance)
        return sorted_boxes[:5]
