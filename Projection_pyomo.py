import numpy as np
import pyomo.environ as pyo
import math


def ProjectMatrix(vec, h, agents):
    # Create problem
    # Create variables
    # Add constraints
    # Add objective function
    # Solve model
    # return variables, reformated back to vec

    model = pyo.ConcreteModel()
    AddVariables(model, vec, h, agents)
    AddConstraints(model, vec, h, agents)
    AddObjFunc(model, vec, h, agents)
    result = SolveModel(model)

    return result


def AddVariables(model, vec, h, agents):
    model.V = pyo.Var([i for i in range(len(vec))],
                      domain=pyo.Reals)
    return


def AddConstraints(model, vec, h, agents):
    return


def AddObjFunc(model, vec, h, agents):

    model.d = pyo.Var([i for i in range(len(vec))],
                      domain=pyo.Reals)

    model.d_constPos = pyo.Constraint(
        expr=model.d >= model.V - vec)
    model.d_constNeg = pyo.Constraint(
        expr=model.d >= -1 * (model.V - vec))

    model.OBJ = pyo.Objective(expr=model.d, sense=pyo.minimize)

    # model.OBJ = pyo.Objective(expr=sum(
    #     model.V[i] * model.V[i] - 2*vec[i]*model.V[i] + vec[i]*vec[i] for i in range(0, len(vec))), sense=pyo.minimize)
    return


def SolveModel(model):
    solvername = 'glpk'
    solverpath_exe = 'C:\\glpk-4.65\\w64\\glpsol'

    opt = pyo.SolverFactory(solvername, executable=solverpath_exe)
    solvedModel = opt.solve(model)
    results = []
    if solvedModel.Solver.Status == pyo.SolverStatus.ok:
        results = solvedModel.V

    return results
