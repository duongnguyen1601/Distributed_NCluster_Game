from Objects.Parameters import DistributionParameters
from OptimizationProblem import *
import matplotlib.pyplot as plt
import numpy as np
import csv
import os
import lzma
import shutil


def GetParameters(test_path):
    test_file = open(test_path, "r")
    params = DistributionParameters.from_json(test_file.read())
    return params


def GetData():
    genData = []
    genCsvPath = 'Data\\GenData.csv'
    if os.path.isfile(genCsvPath):
        with open(genCsvPath) as f:
            genData = [{k: v for k, v in row.items()}
                       for row in csv.DictReader(f, skipinitialspace=True)]

    return genData


def SaveResults(exactNE, ZHist, errorHistZ, name):
    results = {}
    results["ExactNE"] = exactNE
    results["ZHist"] = [arr.tolist() for arr in ZHist]
    results["errorHistZ"] = errorHistZ

    with lzma.open(name + "results.xz", "wb") as f:
        pickle.dump(results, f)


def LoadResults(name):
    with lzma.open(name + "results.xz", "rb") as f:
        results = pickle.load(f)

    exactNE = results["ExactNE"]
    errorHistZ = results["errorHistZ"]
    ZHist = []
    for row in results["ZHist"]:
        ZHist.append(np.array(row))

    return exactNE, ZHist, errorHistZ


def PlotConvergence(exactNE, ZHist, errorHistZ, name):
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

    # Format the data for plotting
    stackedNE = np.zeros((len(ZHist[-1]), len(ZHist[-1][-1])))
    for i in range(len(ZHist[-1])):
        stackedNE[i] = exactNE

    diffVec = []
    for i in range(len(ZHist)):
        diffVec.append(np.linalg.norm(
            ZHist[i]-stackedNE)/np.linalg.norm(stackedNE))

    errorHistZ_arr = np.asarray(errorHistZ)
    errorHistZ_norm = errorHistZ_arr/errorHistZ_arr[0]

    plt.rcParams.update({'font.size': 20})

    # Plot of difference from ideal NE
    f3 = plt.figure()
    ax3 = f3.add_subplot(111)
    ax3.plot(diffVec, linewidth=2)
    ax3.set_xlabel('Iterations')
    ax3.set_ylabel(r'$\|\boldsymbol{z} (k) - \boldsymbol{x}^*\|$')

    resolution_value = 1200
    f3.tight_layout()
    f3.savefig(name+"fig1.png", format="png", dpi=resolution_value)

    f1 = plt.figure()
    ax1 = f1.add_subplot(111)
    ax1.set_yscale('log')
    ax1.plot(diffVec, linewidth=2)
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel(r'$\|\boldsymbol{z} (k) - \boldsymbol{x}^*\|$')

    resolution_value = 1200
    f1.tight_layout()
    f1.savefig(name+"log_fig1.png", format="png", dpi=resolution_value)

    f5 = plt.figure()
    ax5 = f5.add_subplot(111)
    ax5.set_yscale('log')
    ax5.set_xscale('log')
    ax5.plot(diffVec, linewidth=2)
    ax5.set_xlabel('Iterations')
    ax5.set_ylabel(r'$\|\boldsymbol{z} (k) - \boldsymbol{x}^*\|$')

    resolution_value = 1200
    f5.tight_layout()
    f5.savefig(name+"loglog_fig1.png", format="png", dpi=resolution_value)

    # Plot of concensus
    f4 = plt.figure()
    ax4 = f4.add_subplot(111)
    ax4.plot(errorHistZ_norm, linewidth=2)
    ax4.set_xlabel(r'Iterations')
    ax4.set_ylabel(r'$\|\boldsymbol{z} (k) - \boldsymbol{z} (k-1)\|$')
    f4.tight_layout()
    f4.savefig(name+"fig2.png", format="png", dpi=resolution_value)

    f2 = plt.figure()
    ax2 = f2.add_subplot(111)
    ax2.set_yscale('log')
    ax2.plot(errorHistZ_norm, linewidth=2)
    ax2.set_xlabel(r'Iterations')
    ax2.set_ylabel(r'$\|\boldsymbol{z} (k) - \boldsymbol{z} (k-1)\|$')
    f2.tight_layout()
    f2.savefig(name+"log_fig2.png", format="png", dpi=resolution_value)

    f6 = plt.figure()
    ax6 = f6.add_subplot(111)
    ax6.set_yscale('log')
    ax6.plot(errorHistZ_norm, linewidth=2)
    ax6.set_xlabel(r'Iterations')
    ax6.set_ylabel(r'$\|\boldsymbol{z} (k) - \boldsymbol{z} (k-1)\|$')
    f6.tight_layout()
    f6.savefig(name+"loglog_fig2.png", format="png", dpi=resolution_value)

    plt.close(f1)
    plt.close(f2)
    plt.close(f3)
    plt.close(f4)
    plt.close(f5)
    plt.close(f6)


def PlotCostCurve(name, costCurves):
    x_axis = list(range(len(costCurves[0])))
    plt.rcParams.update({'font.size': 20})

    f1 = plt.figure()
    ax1 = f1.add_subplot(111)

    for key, val in costCurves.items():
        ax1.plot(x_axis, val, label="MG"+str(key+1))

    ax1.set_xlabel(r'Iterations')
    ax1.set_ylabel(r'Cost')
    ax1.legend(loc="best", ncol=3, fontsize="12")
    resolution_value = 1200
    f1.savefig(name+"Costs.png", format="png", dpi=resolution_value)
    plt.close(f1)


def Main():
    test_path = "Test_params.json"
    params = GetParameters(test_path)
    LoadFromFile = False
    fileName = f"Test\\{str(params.TestName)}_a_{str(params.alpha)}_aE_{str(params.alpha_NE)}_N_{params.N}_g_{params.gamma}_"
    if LoadFromFile:
        exactNE, ZHist, errorHistZ = LoadResults(
            fileName)
    else:
        genData = GetData()
        problem = OptimizationProblem(params, genData)
        exactNE = problem.FindExactNE()
        ZHist, errorHistZ = problem.Solve()

        # Save results
        SaveResults(exactNE, ZHist, errorHistZ, fileName)
        costCurves = problem.GetCostCurves(ZHist)
        PlotCostCurve(fileName, costCurves)

    PlotConvergence(exactNE, ZHist, errorHistZ, fileName)
    print("Done")


if __name__ == "__main__":
    Main()
