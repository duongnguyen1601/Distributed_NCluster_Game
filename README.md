<a name="readme-top"></a>

<h3 align="center">Distributed Nash Equilibrium Seeking in N-Cluster Game: Day-Ahead Microgrid Energy Management</h3>


### About The Project
Welcome to the Nash Equilibrium simulation project for solving day-ahead microgrid energy management problems. This project provides tools to simulate and analyze the Nash Equilibrium for decision-making among multiple agents and clusters of microgrids over a predefined time horizon.

###  Features
**Nash Equilibrium Solver:** The core of the project contains a Python script designed to run simulations and compute the Nash Equilibrium for decision-making within the microgrid environment.

**MatPower Data Conversion:** A script is included to facilitate the conversion of MatPower data files, which define generators within the microgrid.

### Usage
To utilize the simulation:

1. Set parameters and input data for the microgrids in the **Test_params.json** file.
2. Utilize the **Data/GenData.csv** file to specify generator data. Note: this data originates from the MatPower **ACTIVSg2000.mat** file, processed using **MatPowerConvertor.py**.
3. Execute the simulation via **Main.py**. This will generate a **results.json** file containing the exact Nash Equilibrium values, alongside the history for the estimates Z and the errors. Additionally, convergence plots will be produced. To reload data for re-plotting, set the `LoadFromFile` parameter in **Main.py**, which will load data from **results.json** instead of rerunning the simulation.
4. The `save_state_` parameters determine the frequency of saving the simulation state. This feature is for resuming simulations for additional iterations or recovering from unexpected crashes.

### Data Source
For the dataset used in this project, please refer to the official repository of MATPOWER available at https://github.com/MATPOWER/matpower/tree/master/data
