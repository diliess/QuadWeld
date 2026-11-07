"""Binary-mask skeletons used to order weld centerlines."""

import numpy as np


def thin_mask(stroke):
    """Zhang-Suen thinning without an OpenCV contrib dependency."""
    image = np.pad((stroke > 0).astype(np.uint8), 1)
    for _ in range(128):
        changed = False
        for phase in (0, 1):
            center = image[1:-1, 1:-1]
            p2, p3, p4 = image[:-2, 1:-1], image[:-2, 2:], image[1:-1, 2:]
            p5, p6, p7 = image[2:, 2:], image[2:, 1:-1], image[2:, :-2]
            p8, p9 = image[1:-1, :-2], image[:-2, :-2]
            neighbours = (p2, p3, p4, p5, p6, p7, p8, p9)
            count = sum(neighbours)
            transitions = np.zeros_like(center)
            for first, second in zip(neighbours, neighbours[1:] + neighbours[:1]):
                transitions += ((first == 0) & (second == 1)).astype(np.uint8)
            if phase == 0:
                preserve = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                preserve = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            remove = (center == 1) & (count >= 2) & (count <= 6) & (transitions == 1) & preserve
            if np.any(remove):
                center[remove] = 0
                changed = True
        if not changed:
            break
    return image[1:-1, 1:-1]
