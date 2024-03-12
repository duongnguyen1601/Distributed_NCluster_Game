import scipy.io
from Table_Idx import T_Idx
import csv
import os


class MatPowerConvertor:

    def LoadData(self, dataPath):
        mat = scipy.io.loadmat(dataPath)
        self.version = mat['mpc']['version'][0][0]
        self.baseMVA = mat['mpc']['baseMVA'][0][0][0][0]
        self.bus = mat['mpc']['bus'][0][0]
        self.gen = mat['mpc']['gen'][0][0]
        self.branch = mat['mpc']['branch'][0][0]
        self.gencost = mat['mpc']['gencost'][0][0]
        self.genfuel = mat['mpc']['genfuel'][0][0]
        self.nb = len(self.bus)          # number of buses
        self.nl = len(self.branch)       # number of lines
        self.ng = len(self.gen)          # number of generators

        self.T = range(1)           # Time period (1 day)
        self.U = range(self.ng)

    def SaveCSV(self, outputPath):
        header = ['ID', 'Type', 'Pg', 'Pmin',
                  'Pmax', 'Cost_a', 'Cost_b', 'Cost_c']
        lines = []
        lines.append(header)

        for i in range(len(self.gen)):
            line = [i, self.genfuel[i][0][0], self.gen[i, T_Idx.PG], self.gen[i, T_Idx.PMIN], self.gen[i, T_Idx.PMAX],
                    self.gencost[i, T_Idx.COST], self.gencost[i, T_Idx.COST + 1], self.gencost[i, T_Idx.COST + 2]]
            lines.append(line)

        with open(outputPath, 'w') as file:
            writer = csv.writer(file)
            writer.writerows(lines)


def Main():
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, 'Data\\ACTIVSg2000.mat')

    convertor = MatPowerConvertor()
    convertor.LoadData(filename)
    convertor.SaveCSV("Data\\GenData.csv")


if __name__ == "__main__":
    Main()
