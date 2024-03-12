from Objects.Agent import Agent
from Objects.Parameters import DistributionParameters
import numpy as np


class Battery(Agent):
    def __init__(self, id: int, h: int, params: DistributionParameters, rng: np.random.RandomState) -> None:
        super().__init__(id, h, params, rng)

        self.type = "Battery"

        if params:
            self.a = rng.uniform(params.S_a_range[0], params.S_a_range[1])
            self.b = rng.uniform(params.S_b_range[0], params.S_b_range[1])
            self.c = rng.uniform(params.S_c_range[0], params.S_c_range[1])
            self.max_capacity = rng.uniform(
                params.S_max_cap_range[0], params.S_max_cap_range[1])
            self.max_charge = rng.uniform(
                params.S_max_range[0], params.S_max_range[1]) * self.max_capacity
            self.min_charge = rng.uniform(
                params.S_min_range[0], params.S_min_range[1]) * self.max_capacity
            self.leakage = rng.uniform(
                params.S_leakage_range[0], params.S_leakage_range[1])
            self.initial_charge = rng.uniform(
                params.S_initial_range[0], params.S_initial_range[1]) * self.max_charge
            self.epsilon = params.epsilon

        pass
