from enum import Enum


class DepthMode(Enum):
    OFF = 0 << 2
    NFOV_2X2BINNED = 1 << 2
    NFOV_UNBINNED = 2 << 2
    WFOV_2X2BINNED = 3 << 2
    WFOV_UNBINNED = 4 << 2
    PASSIVE_IR = 5 << 2


class ColorResolution(Enum):
    OFF = 0
    RESOLUTION_720P = 720
    RESOLUTION_1080P = 1080
    RESOLUTION_1440P = 1140
    RESOLUTION_1536P = 1536
    RESOLUTION_2160P = 2160
    RESOLUTION_3072P = 3072
