from typing import Optional

import numpy as np


class OpenK4ACapture:

    def __init__(self, color: Optional[np.ndarray] = None,
                 depth: Optional[np.ndarray] = None,
                 ir: Optional[np.ndarray] = None):
        self.color = color
        self.depth = depth
        self.ir = ir

    @property
    def has_color(self) -> bool:
        return self.color is not None

    @property
    def has_depth(self) -> bool:
        return self.depth is not None

    @property
    def has_ir(self) -> bool:
        return self.ir is not None
