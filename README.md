<a name="readme-top"></a>

<h3 align="center">Distributed Nash Equilibrium in N-Cluster Game</h3>


### About The Project
This project contains a simulation for finding the Nash Equilibrium in a day-ahead microgrid energy management problem.

The project contains a python script for running the simulation and a script for converting MatPower data files to define generators.

The simulation finds the Nash Equilibrium for the decisions of all agents and clusters of microgrids over a predefined time horizon.

### Usage

Set parameters for the simulation and input data for the microgrids in the Test_params.json file. Distributions of possible agents can be set in the file for batteries and generators. Generator data can also be set in the Data/GenData.csv file.
The provided data for GenData.csv was pulled from the MatPwer ACTIVSg2000.mat file using MatPowerConvertor.py

The simulation is run using the Main.py file. This will output a results.json which contains the values for the exact Nash Equilibrium as well as the complete history for V, Z, Y, and the error in Z and V.
The simulation will also about two plots of convergence. To reload the data to re-plot the parameter LoadFromFile can be set in Main.py which will load in the data from the results.json file instead of re-running the simulation.
