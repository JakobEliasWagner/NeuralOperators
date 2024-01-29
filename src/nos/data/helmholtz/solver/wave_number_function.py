from abc import ABC, abstractmethod

import numpy as np


class WaveNumberFunction(ABC):
    """Abstract function that modifies the wave number on a given domain"""

    @abstractmethod
    def eval(self, x: np.array) -> np.array:
        pass
