from Objects.Agent import Agent
from Objects.Parameters import DistributionParameters
import numpy as np


class Generator(Agent):
    def __init__(self, genData, id: int, h: int, params: DistributionParameters, rng: np.random.RandomState) -> None:
        super().__init__(id, h, params, rng)

        self.type = "Generator"

        if genData:
            idx = rng.randint(0, len(genData))
            self.a = float(genData[idx]['Cost_a'])
            self.b = float(genData[idx]['Cost_b'])
            self.c = float(genData[idx]['Cost_c'])
            self.min = float(genData[idx]['Pmin'])
            self.max = float(genData[idx]['Pmax'])
        elif params:
            self.a = rng.choice(params.G_a_set)
            self.b = rng.choice(params.G_b_set)
            self.c = rng.choice(params.G_c_set)
            self.min = rng.choice(params.G_min_set)
            self.max = rng.choice(params.G_max_set)

        pass
