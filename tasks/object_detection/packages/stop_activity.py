from typing import List, Tuple

Detection = Tuple[Tuple[int, int, int, int], float, int]

class_names = {0: 'duckie', 1: 'truck', 2: 'sign'}


def should_stop(detections: List[Detection], img_size: int) -> Tuple[bool, str]:
    """
    Stop when a filtered duckie is close ahead.
    y2 (bottom of box) large -> object is low in the image -> near the robot.
    """
    if not detections:
        return False, ''

    # img_size = camera frame height in pixels (e.g. 480).
    close_y_threshold = img_size * 0.55

    for bbox, score, _cls_id in detections:
        xmin, ymin, xmax, ymax = bbox
        width = xmax - xmin
        height = ymax - ymin
        area = width * height

        if ymax >= close_y_threshold and area > 500:
            return True, f'duckie ahead (score={score:.2f})'

    return False, ''
