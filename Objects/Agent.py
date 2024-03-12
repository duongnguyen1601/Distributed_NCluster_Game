from Objects.Parameters import DistributionParameters
import numpy as np


class Agent():

    def __init__(self, id: int, h: int, params: DistributionParameters, rng: np.random.RandomState) -> None:
        self.h = h
        self.id = id
        self.params = params
        pass
