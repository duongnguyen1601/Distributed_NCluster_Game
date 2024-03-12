from dataclasses import dataclass
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class DistributionParameters:
    TestName: str  # Name of test
    Max_Iter: int  # Maximum number of iterations to run the algo
    Thres: float  # Convergence threshold
    ExactThres: float  # Convergence threshold for exact NE
    alpha: float  # Step size
    alpha_NE: float  # NE Step size
    gamma: float  # weight

    T: int  # Number of timesteps
    N: int  # Number of agents
    H: int  # Number of clusters
    percent_G: float  # Percent chance an agent is a generator
    cluster_membership: list  # How many agents in each cluster
    extra_connections: int  # How many extra connections to add to the graph
    # How many extra connections within clusters to add to the graph
    extra_cluster_connections: int

    demand_set: list  # Set of possible demand levels
    demand_range: list  # Range of possible demand levels

    G_a_set: list  # Set of possible a values for generators
    G_b_set: list  # Set of possible b values for generators
    G_c_set: list  # Set of possible c values for generators
    G_min_set: list  # Set of possible min values for generator rates
    G_max_set: list  # Set of possible max values for generator rates

    S_a_range: list  # Range of possible a values for batteries
    S_b_range: list  # Range of possible b values for batteries
    S_c_range: list  # Range of possible c values for batteries
    S_min_range: list  # Range of possible min values for battery rates
    S_max_range: list  # Range of possible max values for battery rates
    S_min_cap_range: list  # Range of possible min values for battery capacity
    S_max_cap_range: list  # Range of possible max values for battery capacity
    S_leakage_range: list  # Range of possible battery leakage rates
    S_initial_range: list  # Range of possible initial battery charges
    epsilon: float  # Charge difference

    p_scaling_factor: float  # Grid scaling factor
    sell_back_rate: float  # Sell back price as percent of purchase price
