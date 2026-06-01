from typing import Dict, Tuple
import logging
logger = logging.getLogger(__name__)

# Match the notebook "How It Works" table values.
SPEED = 0.5
TURN = 0.3


def get_motor_speeds(keys_pressed: Dict[str, bool]) -> Tuple[float, float]:

    up = bool(keys_pressed.get("up", False))
    down = bool(keys_pressed.get("down", False))
    left_key = bool(keys_pressed.get("left", False))
    right_key = bool(keys_pressed.get("right", False))

    left = 0.0
    right = 0.0

    # Forward / backward
    if up and not down:
        left += SPEED
        right += SPEED
    elif down and not up:
        left -= SPEED
        right -= SPEED

    # Turning (differential)
    if left_key and not right_key:
        left -= TURN
        right += TURN
    elif right_key and not left_key:
        left += TURN
        right -= TURN

    # Clamp to valid range
    left = max(-1.0, min(1.0, left))
    right = max(-1.0, min(1.0, right))

    # logger.debug(
    #     "keys=%s -> speeds=(%.2f, %.2f)",
    #     keys_pressed,
    #     left,
    #     right,
    # )

    return left, right
