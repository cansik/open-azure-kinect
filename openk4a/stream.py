from dataclasses import dataclass


@dataclass
class OpenK4AStream:
    index: int


@dataclass
class OpenK4AVideoStream(OpenK4AStream):
    codec_name: str

    width: int
    height: int

    frame_rate: float
    title: str


class OpenK4AStreams():

    def __init__(self):
        pass
