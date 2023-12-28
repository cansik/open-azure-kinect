from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

OpenK4AColorStreamName = "COLOR"
OpenK4ADepthStreamName = "DEPTH"
OpenK4AInfraredStreamName = "IR"

OpenK4AStreamPixelFormatMapping = {
    OpenK4AColorStreamName: "rgb24",
    OpenK4ADepthStreamName: "gray16le",
    OpenK4AInfraredStreamName: "gray16le",
}


@dataclass
class OpenK4AStream(ABC):
    index: int

    @property
    @abstractmethod
    def stream_name(self) -> str:
        pass


@dataclass
class OpenK4AVideoStream(OpenK4AStream):
    codec_name: str

    width: int
    height: int

    frame_rate: float
    title: str

    stream_name: Optional[str] = None
    pixel_format: Optional[str] = None

    def __post_init__(self):
        if self.stream_name is None:
            self.stream_name = f"v:{self.index}"

        if self.pixel_format is None:
            self.pixel_format = OpenK4AStreamPixelFormatMapping[self.title]
