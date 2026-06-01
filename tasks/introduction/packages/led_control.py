import colorsys
from typing import List


def set_turning_leds(direction: str) -> dict:
    """Set LEDs to indicate turning direction."""
    direction = (direction or "").strip().lower()

    OFF = [0.0, 0.0, 0.0]
    YELLOW = [1.0, 1.0, 0.0]
    WHITE = [1.0, 1.0, 1.0]
    RED = [1.0, 0.0, 0.0]

    if direction == "left":
        return {0: YELLOW, 2: OFF, 3: OFF, 4: YELLOW}
    if direction == "right":
        return {0: OFF, 2: YELLOW, 3: YELLOW, 4: OFF}
    if direction == "forward":
        return {0: WHITE, 2: WHITE, 3: OFF, 4: OFF}
    if direction == "stop":
        return {0: OFF, 2: OFF, 3: RED, 4: RED}

    raise ValueError("direction must be one of: 'left', 'right', 'forward', 'stop'")
