from enum import Enum


class ADStatus(Enum):
    not_calibrated = 1
    calibrating = 2
    calibrated = 3
