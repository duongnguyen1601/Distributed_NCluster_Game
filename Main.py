from Objects.Parameters import DistributionParameters
from OptimizationProblem import *
import matplotlib.pyplot as plt
import numpy as np
import json
import csv
import os


def GetParameters():
    test_path = "Test_params_small.json"
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


def SaveJson(exactNE, VHist, ZHist, YHist, errorHistZ, errorHistV):
    result_dict = {}
    result_dict["exactNE"] = exactNE.tolist()
    result_dict["VHist"] = [arr.tolist() for arr in VHist]
    result_dict["ZHist"] = [arr.tolist() for arr in ZHist]

    YHistConvert = []
    for YCur in YHist:
        YCurConvert = {}
        for key, item in YCur.items():
            YCurConvert[key] = item.tolist()
        YHistConvert.append(YCurConvert)

    result_dict["YHist"] = YHistConvert
    result_dict["errorHistZ"] = errorHistZ
    result_dict["errorHistV"] = errorHistV

    json_object = json.dumps(result_dict, indent=4)
    with open("results.json", "w") as outfile:
        outfile.write(json_object)


def LoadJson():
    result_dict = {}
    with open("results.json") as f:
        result_dict = json.load(f)

    exactNE = result_dict.get("exactNE", [])
    VHist = result_dict.get("VHist", [])
    ZHist = result_dict.get("ZHist", [])
    YHist = result_dict.get("YHist", [])
    errorHistZ = result_dict.get("errorHistZ", [])
    errorHistV = result_dict.get("errorHistV", [])

    return exactNE, VHist, ZHist, YHist, errorHistZ, errorHistV


def PlotConvergence(exactNE, ZHist, errorHistZ):
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

    # diffVec = diffVec/diffVec[0]

    errorHistZ_arr = np.asarray(errorHistZ)
    errorHistZ_norm = errorHistZ_arr/errorHistZ_arr[0]

    plt.rcParams.update({'font.size': 20})

    # Plot of difference from ideal NE
    f1 = plt.figure()
    ax1 = f1.add_subplot(111)
    ax1.set_yscale('log')
    ax1.plot(diffVec, linewidth=2)
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel(r'$\|\boldsymbol{z} (k) - \boldsymbol{x}^*\|$')

    resolution_value = 1200
    f1.tight_layout()
    f1.savefig("fig1.png", format="png", dpi=resolution_value)

    # Plot of concensus
    f2 = plt.figure()
    ax2 = f2.add_subplot(111)
    ax2.set_yscale('log')
    ax2.plot(errorHistZ_norm, linewidth=2)
    ax2.set_xlabel(r'Iterations')
    ax2.set_ylabel(r'$\|\boldsymbol{z} (k) - \boldsymbol{z} (k-1)\|$')
    f2.tight_layout()
    f2.savefig("fig2.png", format="png", dpi=resolution_value)


def Main():
    LoadFromFile = True

    if LoadFromFile:
        exactNE, VHist, ZHist, YHist, errorHistZ, errorHistV = LoadJson()
    else:
        params = GetParameters()
        genData = GetData()
        problem = OptimizationProblem(params, genData)
        exactNE = problem.FindExactNE()
        VHist, ZHist, YHist, errorHistZ, errorHistV = problem.Solve()

        # Save results
        SaveJson(exactNE, VHist, ZHist, YHist, errorHistZ, errorHistV)

    PlotConvergence(exactNE, ZHist, errorHistZ)
    print("Done")


if __name__ == "__main__":
    Main()
