from typing import Tuple

# Path to the trained model weights (.onnx file).
# Relative paths resolve from the project root.
MODEL_PATH = "tasks/object_detection/models/best.onnx"


def NUMBER_FRAMES_SKIPPED() -> int:
    # Run inference every (skip + 1) frames. Increase if FPS is too low.
    return 1


def filter_by_classes(pred_class: int) -> bool:
    """Keep duckies (0) and trucks (1). Signs (2) are filtered out."""
    return pred_class in (0, 1)


def filter_by_scores(score: float) -> bool:
    """Drop low-confidence predictions."""
    return score >= 0.5


def filter_by_bboxes(bbox: Tuple[int, int, int, int]) -> bool:
    """Drop boxes that are too small to matter (likely far away)."""
    xmin, ymin, xmax, ymax = bbox
    width = xmax - xmin
    height = ymax - ymin
    area = width * height
    return area > 500
