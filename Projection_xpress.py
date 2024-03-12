import numpy as np
import xpress as xp
from OutputFormatter import Formatter


class Projection():

    def GetProjection(self, vec, h, clusters, T, demandLevels):
        self.vec = vec
        self.h = h
        self.T = T
        self.clusters = clusters
        self.demandLevels = demandLevels
        self.model = xp.problem()
        xp.setOutputEnabled(False)
        self.SetIndexes()
        self.CreateVariables()
        self.CreateConstraints()
        self.CreateObjFunc()
        solvestatus, solstatus, results = self.SolveModel()

        return solvestatus, solstatus, results

    def SetIndexes(self):
        agents = self.clusters[self.h]
        self.R_set = []  # Indexes of generators
        self.B_set = []  # Indexes of batteries

        for idx, a in enumerate(agents):
            if a.type == "Generator":
                self.R_set.append(idx)
            else:
                self.B_set.append(idx)

    def CreateVariables(self):
        # Split up the agents into generator and battery
        # Make variables for both and the cluster decision
        agents = self.clusters[self.h]

        # Generators
        if self.R_set:
            self.PR = np.array([[xp.var(name=f"PR_{i}_{t}", vartype=xp.continuous, lb=agents[i].min, ub=agents[i].max)
                                 for t in range(self.T)] for i in self.R_set], dtype=xp.npvar)
            self.model.addVariable(self.PR)
        else:
            self.PR = []

        # Batteries
        if self.B_set:
            self.PB = np.array([[xp.var(name=f"PB_{i}_{t}", vartype=xp.continuous, lb=-1*agents[i].max_capacity, ub=agents[i].max_capacity)
                                 for t in range(self.T)] for i in self.B_set], dtype=xp.npvar)
            self.model.addVariable(self.PB)
        else:
            self.PB = []

        # Buy from grid
        self.PG = np.array([xp.var(name=f"PG_{t}", vartype=xp.continuous)
                            for t in range(self.T)], dtype=xp.npvar)
        self.model.addVariable(self.PG)

        # Sell back to grid
        self.PS = np.array([xp.var(name=f"PS_{t}", vartype=xp.continuous)
                            for t in range(self.T)], dtype=xp.npvar)
        self.model.addVariable(self.PS)

    def CreateConstraints(self):
        agents = self.clusters[self.h]

        if self.B_set:
            # Battery charge constraint
            PCL_constraints = [
                -1 * xp.Sum(
                    agents[self.B_set[i]].leakage**(t-s) * self.PB[i][s] for s in range(t)) >=
                -1 * agents[self.B_set[i]
                            ].leakage**(t-1) * agents[self.B_set[i]].initial_charge
                for i in range(len(self.B_set)) for t in range(1, self.T)]
            for i in range(len(self.B_set)):
                for t in range(self.T - 1):
                    PCL_constraints[i*(self.T-1) +
                                    t].name = f"PCL_{self.B_set[i]}_{t}"
            self.model.addConstraint(PCL_constraints)

            PCU_constraints = [
                -1 * xp.Sum(agents[self.B_set[i]].leakage**(t-s) * self.PB[i][s] for s in range(t)) <=
                agents[self.B_set[i]].max_charge -
                agents[self.B_set[i]].leakage**(t-1) *
                agents[self.B_set[i]].initial_charge
                for i in range(len(self.B_set)) for t in range(1, self.T)]
            for i in range(len(self.B_set)):
                for t in range(self.T - 1):
                    PCU_constraints[i*(self.T-1) +
                                    t].name = f"PCU_{self.B_set[i]}_{t}"
            self.model.addConstraint(PCU_constraints)

            # End battery level constraint
            PC_EndL_constraints = [
                agents[self.B_set[i]].leakage**(self.T-2) * agents[self.B_set[i]].initial_charge -
                xp.Sum(agents[self.B_set[i]].leakage**(self.T - 1 - s) * self.PB[i][s]
                       for s in range(self.T-2))
                - agents[self.B_set[i]
                         ].initial_charge <= agents[self.B_set[i]].epsilon * agents[self.B_set[i]].max_capacity
                for i in range(len(self.B_set))
            ]
            for i in range(len(self.B_set)):
                PC_EndL_constraints[i].name = f"PC_EndL_{self.B_set[i]}"
            self.model.addConstraint(PC_EndL_constraints)

            PC_EndU_constraints = [
                agents[self.B_set[i]].leakage**(self.T-2) * agents[self.B_set[i]].initial_charge -
                xp.Sum(agents[self.B_set[i]].leakage**(self.T - 1 - s) * self.PB[i][s]
                       for s in range(self.T-2))
                - agents[self.B_set[i]].initial_charge >= -
                1 * agents[self.B_set[i]].epsilon *
                agents[self.B_set[i]].max_capacity
                for i in range(len(self.B_set))
            ]
            for i in range(len(self.B_set)):
                PC_EndU_constraints[i].name = f"PC_EndU_{self.B_set[i]}"
            self.model.addConstraint(PC_EndU_constraints)

        # Demand constraint
        Demand_constraints = [
            xp.Sum(self.PR[i][t] for i in range(len(self.R_set))) +
            xp.Sum(self.PB[i][t] for i in range(len(self.B_set))) + self.PG[t]
            == self.demandLevels[self.h][t] + self.PS[t] for t in range(self.T)
        ]
        for t in range(self.T):
            Demand_constraints[t].name = f"Demand_{t}"
        self.model.addConstraint(Demand_constraints)

    def GetInputValue(self, i, t):
        agents = self.clusters[self.h]
        L = len(agents) + 2
        idx = (t * L) + i
        return self.vec[idx]

    def CreateObjFunc(self):
        agents = self.clusters[self.h]
        PR_input = np.array([[self.GetInputValue(i, t) for t in range(self.T)]
                             for i in self.R_set], dtype=float)

        PB_input = np.array([[self.GetInputValue(i, t) for t in range(self.T)]
                             for i in self.B_set], dtype=float)

        PG_input = np.array([self.GetInputValue(len(agents), t)
                            for t in range(self.T)], dtype=float)

        PS_input = np.array([self.GetInputValue(len(agents) + 1, t)
                            for t in range(self.T)], dtype=float)

        self.model.setObjective(xp.Sum(self.PR[i][t]*self.PR[i][t] - 2*PR_input[i][t]*self.PR[i][t] + PR_input[i][t]*PR_input[i][t] if len(self.PR) > 0 else 0 for i in range(len(self.PR)) for t in range(self.T))
                                + xp.Sum(self.PB[i][t]*self.PB[i][t] - 2*PB_input[i][t]*self.PB[i][t] +
                                         PB_input[i][t]*PB_input[i][t] if len(self.PB) > 0 else 0 for i in range(len(self.PB)) for t in range(self.T))
                                + xp.Sum(self.PG[t]*self.PG[t] - 2*PG_input[t]*self.PG[t] +
                                         PG_input[t]*PG_input[t] for t in range(self.T))
                                + xp.Sum(self.PS[t]*self.PS[t] - 2*PS_input[t]*self.PS[t] + PS_input[t]*PS_input[t] for t in range(self.T)), sense=xp.minimize)

    def SolveModel(self):

        solvestatus, solstatus = self.model.optimize()
        results = Formatter.format_output(
            self.model.getVariable(), self.model.getSolution(), self.clusters[self.h], self.T)

        if (np.sum(results) == 0):
            print("infeasible")

        return solvestatus, solstatus, results
