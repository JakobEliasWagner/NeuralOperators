from enum import IntEnum


class Direction(IntEnum):
    """Direction in a 3-dimensional space.

    """
    positive_x = 1
    positive_y = 2
    positive_z = 3
    negative_x = -1
    negative_y = -2
    negative_z = -3