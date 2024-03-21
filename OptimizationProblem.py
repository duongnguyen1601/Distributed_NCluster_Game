from Objects.Parameters import DistributionParameters
from Objects.Battery import Battery
from Objects.Generator import Generator
import numpy as np
from collections import defaultdict
from Projection_xpress import Projection
import pickle
import bz2
import os


class OptimizationProblem:
    def __init__(self, params: DistributionParameters, genData) -> None:
        self.rng = np.random.RandomState(
            np.random.MT19937(np.random.SeedSequence(1610925)))
        self.params = params
        self.ProblemSetup(genData)
        pass

    def ProblemSetup(self, genData):
        self.T = self.params.T
        self.N = self.params.N
        self.H = self.params.H
        self.agents = {}
        self.clusters = defaultdict(list)
        self.membershipDict = {}
        h = 0
        for i in range(0, self.params.N):
            if len(self.clusters[h]) >= self.params.cluster_membership[h]:
                h += 1
            if self.rng.rand() > self.params.percent_G:
                agent = Battery(i, h, self.params, self.rng)
            else:
                agent = Generator(genData, i, h, self.params, self.rng)
            self.agents[i] = agent
            self.clusters[h].append(agent)
            self.membershipDict[i] = h
        self.SetDemandLevels()

    def SetDemandLevels(self):
        self.demandLevels = np.zeros((self.H, self.T))

        if self.params.equal_demand:
            for t in range(self.T):
                demand = self.rng.uniform(
                    self.params.demand_range[0], self.params.demand_range[1])
                for h in range(self.H):
                    self.demandLevels[h][t] = demand
        else:
            for h in range(self.H):
                for t in range(self.T):
                    self.demandLevels[h][t] = self.rng.uniform(
                        self.params.demand_range[0], self.params.demand_range[1])

    def GetCommGraph(self):
        # Create graph
        commGraph = np.zeros((self.N, self.N))

        # Add self loops
        for i in range(self.N):
            commGraph[i][i] = 1

        for h in range(self.H):
            # Add loop within each cluster
            cluster = self.clusters[h]
            commGraph[cluster[-1].id, cluster[0].id] = 1
            for i in range(len(cluster) - 1):
                commGraph[cluster[i].id, cluster[i + 1].id] = 1

            # Add random connections within cluster
            for _ in range(self.params.extra_cluster_connections):
                i = self.rng.randint(0, len(cluster))
                j = self.rng.randint(0, len(cluster))
                commGraph[cluster[i].id, cluster[j].id] = 1

        # Add loop between clusters
        for h in range(self.H - 1):
            i = self.rng.choice(self.clusters[h]).id
            j = self.rng.choice(self.clusters[h + 1]).id
            commGraph[i, j] = 1

        i = self.rng.choice(self.clusters[self.H - 1]).id
        j = self.rng.choice(self.clusters[0]).id
        commGraph[i, j] = 1

        # Add extra random connections
        for _ in range(self.params.extra_connections):
            h_1 = self.rng.randint(0, self.H)
            h_2 = self.rng.randint(0, self.H)
            if h_1 != h_2:
                i = self.rng.choice(self.clusters[h_1]).id
                j = self.rng.choice(self.clusters[h_2]).id
                commGraph[i, j] = 1

        return commGraph

    def GetWeightMatrix(self, commGraph):
        # Copy graph and make row stochastic with random weights
        weightMat = commGraph.copy()
        for i in range(self.N):
            row_sum = int(commGraph[i].sum())
            for j in range(self.N):
                weightMat[i, j] = weightMat[i, j] / row_sum
        return weightMat

    def GetClusterMatricies(self, commGraph):
        clusterMatricies = {}
        prev_idx = 0
        for h in range(self.H):
            end_idx = prev_idx+self.params.cluster_membership[h]
            clusterMatrix = commGraph[prev_idx:end_idx,
                                      prev_idx:end_idx].copy()
            self.MakeColStochasticMatrix(clusterMatrix)
            clusterMatricies[h] = clusterMatrix
            prev_idx = end_idx
        return clusterMatricies

    def MakeColStochasticMatrix(self, input_matrix):
        sum_arr = input_matrix.sum(axis=0)
        for i in range(len(input_matrix)):
            col_sum = sum_arr[i]
            for j in range(len(input_matrix)):
                input_matrix[j, i] = input_matrix[j, i] / col_sum

    def VUpdate(self, R, ZPrev):
        VCur = ZPrev.astype("float").copy()
        for i in range(self.N):
            VCur[i] = np.matmul(R[i], ZPrev)
        return VCur

    def getAgentDecision(self, V, i, j, t):
        idx = t*(self.N + 2*self.H) + j
        return V[i, idx]

    def getClusterDecision(self, V, i, h, t):
        idx = t*(self.N + 2*self.H) + self.N + h
        return V[i, idx]

    def getGrad(self, V):
        Grads = {}

        for h in range(self.H):
            cluster = self.clusters[h]
            L = len(cluster) + 2

            for i, agent in enumerate(cluster):
                agent_grad = np.zeros(self.T*(L))
                for t in range(self.T):
                    idx = (t * L) + i
                    Pr = self.getAgentDecision(V, agent.id, agent.id, t)
                    agent_grad[idx] = 2*agent.a*Pr + np.sign(Pr)*agent.b
                for t in range(self.T):
                    idx = t * L + len(cluster)

                    clusterDecSum = 0
                    for h2 in range(self.H):
                        clusterDecSum += self.getClusterDecision(
                            V, agent.id, h2, t)

                    agent_grad[idx] = (1/len(cluster)) * (self.params.p_scaling_factor * (
                        self.getClusterDecision(V, agent.id, h, t) - self.params.sell_back_rate * self.getClusterDecision(V, agent.id, self.H + h, t) + clusterDecSum))

                    idx_sb = t * L + len(cluster) + 1
                    agent_grad[idx_sb] = (1/len(cluster)) * self.params.p_scaling_factor * \
                        (-1 * self.params.sell_back_rate * clusterDecSum)

                Grads[agent.id] = agent_grad

        # This is a dictionary, key = agentId, value = vector of length, (# of agents in that same cluster + 1) * T
        return Grads

    def YUpdate(self, C_dict, VPrev, VCur, YPrev):
        YCur = {}
        for i, vec in YPrev.items():
            YCur[i] = vec.copy()

        fCur = self.getGrad(VCur)
        fPrev = self.getGrad(VPrev)

        for h in range(self.H):
            cluster = self.clusters[h]
            YPrev_h = np.zeros((len(cluster), self.T*(len(cluster)+2)))
            C_h = C_dict[h]
            for i, agent in enumerate(cluster):
                YPrev_h[i] = YPrev[agent.id]

            for i, agent in enumerate(cluster):
                YCur[agent.id] = np.matmul(
                    C_h[i], YPrev_h) + fCur[agent.id] - fPrev[agent.id]

        return YCur

    def getProject(self, vec, h):
        # Make optimization problem that includes every constraint, for every agent in the cluster
        # Minimize distance between vec and output (proj_vec)
        # Decision variable is proj_vec which is same dimension as the input vec
        # Input vector is 1xT*L where L is number of agents in cluster + 1
        Projector = Projection()
        solvestatus, solstatus, projected_vec = Projector.GetProjection(
            vec, h, self.clusters, self.T, self.demandLevels)

        return projected_vec

    def CollapseZ(self, L, agent, cluster, Cur, h):
        # Pick the values for the given cluster out of the full Z and collapse into smaller vector
        Vec_i_h = np.zeros(self.T*(L))
        for t in range(self.T):
            for i2, agent2 in enumerate(cluster):
                idx = (t * L) + i2
                Vec_i_h[idx] = self.getAgentDecision(
                    Cur, agent.id, agent2.id, t)
            idx = t * L + len(cluster)
            Vec_i_h[idx] = self.getClusterDecision(
                Cur, agent.id, h, t)

            idx_sb = t * L + len(cluster) + 1
            Vec_i_h[idx_sb] = self.getClusterDecision(
                Cur, agent.id, h + self.H, t)

        return Vec_i_h

    def ExpandZ(self, L, agent, cluster, Z, h, Vec_i_h):
        # Take the vector for the cluster and put the values back into the larger Z
        for t in range(self.T):
            for i2, agent2 in enumerate(cluster):
                idx_i_h = (t * L) + i2
                idx_i = t*(self.N + 2*self.H) + agent2.id
                Z[agent.id, idx_i] = Vec_i_h[idx_i_h]

            idx_h_h = t * L + len(cluster)
            idx_h = t*(self.N + 2*self.H) + self.N + h
            Z[agent.id, idx_h] = Vec_i_h[idx_h_h]

            idx_h_h_sb = t * L + len(cluster) + 1
            idx_h_sb = t*(self.N + 2*self.H) + self.N + h + self.H
            Z[agent.id, idx_h_sb] = Vec_i_h[idx_h_h_sb]
        return Z

    def ZUpdate(self, VCur, YCur):
        ZCur = VCur.copy()

        for h in range(self.H):
            cluster = self.clusters[h]
            L = len(cluster) + 2
            for agent in cluster:

                VCur_i_h = self.CollapseZ(L, agent, cluster, VCur, h)

                ZCur_i_h = (1 - self.params.gamma)*self.getProject(VCur_i_h, h) + (
                    self.params.gamma) * self.getProject(VCur_i_h - self.params.alpha * YCur[agent.id], h)

                ZCur = self.ExpandZ(L, agent, cluster, ZCur, h, ZCur_i_h)

        return ZCur

    def initializeZ(self):
        ZPrev = self.rng.rand(self.N, (2*self.H + self.N) * self.T)*2

        # Project Z
        for h in range(self.H):
            cluster = self.clusters[h]
            L = len(cluster) + 2
            for agent in cluster:

                ZPrev_i_h = self.CollapseZ(L, agent, cluster, ZPrev, h)

                ZPrev_i_h = self.getProject(ZPrev_i_h, h)

                ZPrev = self.ExpandZ(L, agent, cluster, ZPrev, h, ZPrev_i_h)

        return ZPrev

    def Solve(self):
        # Initilize random Z (estimate of answers), V (same as Z), Y (gradient, calculated later) for each agent
        # Get gradient function
        # make adjency matrix named A
        # A is row stochastic large weight matrix, B is the col stochastic matrix for each cluster (Rename to R and C)
        # Z update, V update, Y update
        # V update (same as x update, eq 16) VCur[i] = np.matmul(A[i],XPrev)
        # Y update (same as yUpdate but done for each cluster, eq 17)
        # Z update (in two parts, for in cluster and out, eq 18) second part is just copying the V, in cluster part is projecting using the optimizer
        ###
        # Check how close the estimates are to each other
        # Then either continue or terminate
        ############################

        # Create communication graph and weight matricies
        commGraph = self.GetCommGraph()
        R = self.GetWeightMatrix(commGraph)
        C_dict = self.GetClusterMatricies(commGraph)

        if self.params.save_state_freq > 0 and os.path.isfile("cSaveState.bz2"):
            iter, VPrev, YPrev, ZPrev, ZHist = self.LoadCompressedState()
            iter = iter + 1
            errorHistZ = []
            for i in range(len(ZHist) - 1):
                errorHistZ.append(np.linalg.norm(ZHist[i+1] - ZHist[i]))
        else:
            # Each row is all of the agent decisions, followed by cluster decisions, for each timestep
            ZPrev = self.initializeZ()
            VPrev = ZPrev.copy()
            # This is a dictionary, key = agentId, value = vector of length, # of agents in that same cluster + 1
            YPrev = self.getGrad(VPrev)

            # Initilize histories
            ZHist = []
            errorHistZ = []
            iter = 0

        for iter in range(iter, self.params.Max_Iter):
            VCur = self.VUpdate(R, ZPrev)
            YCur = self.YUpdate(C_dict, VPrev, VCur, YPrev)
            ZCur = self.ZUpdate(VCur, YCur)
            ZHist.append(ZCur)

            errorHistZ.append(np.linalg.norm(ZCur - ZPrev))

            if iter % 200 == 0:
                print(str(errorHistZ[-1]) + "," +
                      str(np.linalg.norm(VCur - VPrev)))

            if np.linalg.norm(ZCur) > 10**6:
                print('Iteration is diverging. Iam going to stop iteration.')
                break
            if np.linalg.norm(ZCur - ZPrev) < self.params.Thres and np.linalg.norm(VCur - VPrev) < self.params.Thres:
                print(f'The algorithm stops at iteration {iter}.')
                print(ZCur)
                break
            VPrev = VCur
            YPrev = YCur
            ZPrev = ZCur

            if self.params.save_state_freq > 0 and iter > 0 and iter % self.params.save_state_freq == 0:
                self.SaveCompressedState(iter, VPrev, YPrev, ZPrev, ZHist)

        return ZHist, errorHistZ

    def SaveCompressedState(self, iter, VPrev, YPrev, ZPrev, ZHist):
        result_dict = {}
        YPrevConvert = {}
        for key, item in YPrev.items():
            YPrevConvert[key] = item.tolist()

        result_dict["YPrev"] = YPrevConvert
        result_dict["ZPrev"] = ZPrev.tolist()
        result_dict["VPrev"] = VPrev.tolist()
        result_dict["iter"] = iter

        pickle.dump(result_dict, bz2.open("cSaveState.bz2", "wb"))

        with open("hisSaveState.pkl", "ab") as f:
            for entry in ZHist[len(ZHist) - self.params.save_state_freq:]:
                pickle.dump(entry.tolist(), f)

    def LoadCompressedState(self):
        with bz2.open("cSaveState.bz2", "rb") as f:
            result_dict = pickle.load(f)

        iter = result_dict["iter"]
        ZPrev = []
        for row in result_dict["ZPrev"]:
            ZPrev.append(np.array(row))
        VPrev = np.array(result_dict["VPrev"])
        ZPrev = np.array(result_dict["ZPrev"])
        YPrev = {}
        for key, item in result_dict["YPrev"].items():
            YPrev[int(key)] = np.array(item)

        ZHist = []
        with open("hisSaveState.pkl", "rb") as f:
            for _ in range(iter):
                z = pickle.load(f)
                ZHist.append(np.array(z))

        return iter, VPrev, YPrev, ZPrev, ZHist

    def GetCostCurves(self, ZHist):
        clusterCostCurves = {i: [] for i in range(self.H)}
        for i in range(0, len(ZHist)):
            clusterCosts = self.GetClusterCosts(ZHist[i])
            for key, val in clusterCosts.items():
                clusterCostCurves[key].append(val)
        return clusterCostCurves

    def GetClusterCosts(self, ZCur):
        Costs = {}

        # Get amount bought/sold total to the grid for every timestamp
        gridUsageAmounts = []

        for t in range(self.T):
            gridTotal = 0
            for h in range(self.H):
                cluster = self.clusters[h]
                clusterAmount = 0
                for i, agent in enumerate(cluster):
                    clusterAmount += self.getClusterDecision(
                        ZCur, agent.id, h, t)

                clusterAmount = clusterAmount / len(cluster)
                gridTotal += clusterAmount

            gridUsageAmounts.append(gridTotal)

        for h in range(self.H):
            cluster = self.clusters[h]

            clusterTotalCost = 0
            for t in range(self.T):
                # Get cluster cost and sell back
                Pg = 0
                Ps = 0

                for i, agent in enumerate(cluster):
                    Pr = self.getAgentDecision(ZCur, agent.id, agent.id, t)
                    Pg += self.getClusterDecision(ZCur, agent.id, h, t)
                    Ps += self.getClusterDecision(ZCur,
                                                  agent.id, self.H + h, t)
                    agentCost = agent.a*Pr*Pr + \
                        np.sign(Pr)*agent.b*Pr + agent.c
                    clusterTotalCost += agentCost

                Pg = Pg / len(cluster)
                Ps = Ps / len(cluster)
                gridCost = Pg * self.params.p_scaling_factor * \
                    gridUsageAmounts[t]
                gridSell = Ps * self.params.sell_back_rate * \
                    self.params.p_scaling_factor * gridUsageAmounts[t]
                clusterTotalCost += gridCost
                clusterTotalCost -= gridSell

            Costs[h] = clusterTotalCost

        return Costs


# Find the exact NE with full information

    def FindExactNE(self):

        if self.params.save_state_freq > 0 and os.path.isfile("cSaveState_E.bz2"):
            iter, ZPrev = self.LoadCompressedExactState()
            ZCur = ZPrev
            iter = iter + 1
        else:
            # Initialize and Project Z
            ZPrev = self.rng.rand((2*self.H + self.N) * self.T)*5

            for h in range(self.H):
                cluster = self.clusters[h]
                L = len(cluster) + 2

                ZPrev_h = self.CollapseExactZ(L, cluster, ZPrev, h)

                ZPrev_h = self.getProject(ZPrev_h, h)

                ZPrev = self.ExpandExactZ(L, cluster, h, ZPrev_h, ZPrev)
                iter = 0

        for iter in range(iter, 2 * self.params.Max_Iter):
            # Stack the Zprev into a N row array
            ZCur = np.zeros(((2*self.H + self.N) * self.T))
            ZCur_stack = np.zeros((self.N, (2*self.H + self.N) * self.T))
            for i in range(self.N):
                ZCur_stack[i] = ZPrev

            # Get all gradients
            fCur = self.getGrad(ZCur_stack)

            # Z Update
            for h in range(self.H):
                cluster = self.clusters[h]
                L = len(cluster) + 2

                # Get the gradient for the cluster
                FCur = np.zeros(self.T*(L))
                for agent in cluster:
                    FCur += fCur[agent.id]

                ZPrev_h = self.CollapseExactZ(L, cluster, ZPrev, h)

                temp = ZPrev_h - self.params.alpha_NE * FCur

                ZCur_h = self.getProject(
                    temp, h)

                ZCur = self.ExpandExactZ(L, cluster, h, ZCur_h, ZCur)

            # Check convergence
            print(np.linalg.norm(ZCur - ZPrev))
            if np.linalg.norm(ZCur) > 10**6:
                print('Iteration is diverging. Iam going to stop iteration.')
                break
            if np.linalg.norm(ZCur - ZPrev) < self.params.ExactThres:
                print(f'The algorithm stops at iteration {iter}.')
                print(ZCur)
                break
            ZPrev = ZCur

            if self.params.save_state_freq > 0 and iter % self.params.save_state_freq == 0:
                self.SaveCompressedExactState(iter, ZPrev)

        return ZCur

    def CollapseExactZ(self, L, cluster, ZPrev, h):
        # Pick the values for the given cluster out of the full Z and collapse into smaller vector
        ZPrev_h = np.zeros(self.T*(L))
        for t in range(self.T):
            # Get the values for each agent in the cluster at each timestep
            for i, agent in enumerate(cluster):
                idx = (t * L) + i
                prev_idx = t*(self.N + 2*self.H) + agent.id
                ZPrev_h[idx] = ZPrev[prev_idx]
            # Get the values for the cluster decision at each timestep
            idx = t * L + len(cluster)
            prev_idx = t*(self.N + 2*self.H) + self.N + h
            ZPrev_h[idx] = ZPrev[prev_idx]

            idx_sb = t * L + len(cluster) + 1
            prev_idx_sb = t*(self.N + 2*self.H) + self.N + h + self.H
            ZPrev_h[idx_sb] = ZPrev[prev_idx_sb]

        return ZPrev_h

    def ExpandExactZ(self, L, cluster, h, ZPrev_h, ZPrev):
        # Take the vector for the cluster and put the values back into the larger Z
        for t in range(self.T):
            for i, agent in enumerate(cluster):
                idx = (t * L) + i
                prev_idx = t*(self.N + 2*self.H) + agent.id
                ZPrev[prev_idx] = ZPrev_h[idx]

            prev_idx = t * L + len(cluster)
            idx = t*(self.N + 2*self.H) + self.N + h
            ZPrev[idx] = ZPrev_h[prev_idx]

            prev_idx_sb = t * L + len(cluster) + 1
            idx_sb = t*(self.N + 2*self.H) + self.N + h + self.H
            ZPrev[idx_sb] = ZPrev_h[prev_idx_sb]
        return ZPrev

    def SaveCompressedExactState(self, iter, ZPrev):
        result_dict = {}

        result_dict["ZPrev"] = ZPrev.tolist()
        result_dict["iter"] = iter

        pickle.dump(result_dict, bz2.open("cSaveState_E.bz2", "wb"))

    def LoadCompressedExactState(self):
        with bz2.open("cSaveState_E.bz2", "rb") as f:
            result_dict = pickle.load(f)

        iter = result_dict["iter"]
        ZPrev = []
        for row in result_dict["ZPrev"]:
            ZPrev.append(np.array(row))
        ZPrev = np.array(result_dict["ZPrev"])

        return iter, ZPrev
