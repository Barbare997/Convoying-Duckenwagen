from typing import List, Tuple
import numpy as np


def detect_curve(yellow_xs: List[int],white_xs:  List[int],curve_threshold: int = 350,
    ) -> Tuple[bool, int]:
    # Compare line positions near the robot (index 0) vs farther ahead (index -1).
    # Large horizontal shift means we are entering a curve.
    shifts = []

    # Use yellow line shift if enough points were detected.
    if len(yellow_xs) >= 2:
        shifts.append(yellow_xs[-1] - yellow_xs[0])  # far minus near

    # Use white line shift too, when available.
    if len(white_xs) >= 2:
        shifts.append(white_xs[-1] - white_xs[0])

    # No reliable shift estimate from either line.
    if not shifts:
        return False, 0

    # Average both shifts to reduce single-line noise.
    avg_shift = int(np.mean(shifts))

    # Return curve flag + signed shift (sign indicates turn side).
    if abs(avg_shift) > curve_threshold:
        # positive shift = lines moved right in image = road curves left
        # negative shift = lines moved left = road curves right
        return True, avg_shift

    return False, 0
