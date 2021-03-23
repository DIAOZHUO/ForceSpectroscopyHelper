from ForceSpectroscopyHelper.formula import *
import matplotlib.pyplot as plt


class DulcineaMapCreator:

    def __init__(self):
        # row
        self.I = 1024
        # smooth dfを計算するために
        self.N = 1
        # nm
        self.Width = 10
        # FMD_sensitivity
        self.Sen = 72.24
        # A/V
        self.VolAmp = 40.9

        # xOrder ファイルに書いたxのレンジ
        self.xOrder = 1.5
        # yOrder ファイルに書いたyのレンジ
        self.yOrder = 0.575

    # ファイルの縦にYの値が並んでいる場合
    @staticmethod
    def ReadXYFromFile(file, index):
        lines = file.readlines()
        valuex = []
        valuey = []
        for i in range(0, 1024):
            valuex.append(i)
        for line in lines:
            valuey.append(float(line.split()[index]))
        return valuex, valuey

    def GetDulcineaParam(self, A, f0, k, height, data_count):
        param = measurement_param(height, data_count)
        param.amp = A * self.VolAmp
        param.f0 = f0
        param.k = k
        return param

    def CalcDfsMap(self, data):
        I = data.shape[0]
        J = data.shape[1]
        Sen = self.Sen
        N = self.N

        Df = np.asarray(data)
        # DfS = np.zeros((I,J))
        print("smooth", N)

        c = 0
        for j in range(0, J):
            c += Df[I-1][j]
        c = c / J * Sen
        """
        for j in range(0, J):
            for i in range(int((N - 1) / 2), int(I - (N - 1) / 2)):
                DfS[i][j] = 0
                for m in range(int(i - (N - 1) / 2), int(i + (N - 1) / 2 + 1)):
                    DfS[i][j] += Df[m][j]
                DfS[i][j] = -DfS[i][j] / N

            for i in range(0, int((N - 1) / 2)):
                DfS[i][j] = 0
                for m in range(0, int(i + (N - 1) / 2 + 1)):
                    DfS[i][j] += Df[m][j]
                DfS[i][j] = -DfS[i][j] / ((N - 1) / 2 + i + 1)

            for i in range(int(I - (N - 1) / 2), I):
                DfS[i][j] = 0
                for m in range(int(i - (N - 1) / 2), I):
                    DfS[i][j] += Df[m][j]
                DfS[i][j] = -DfS[i][j] / ((N - 1) / 2 + I - i)
        """

        return -SmoothMap(Df, N) * Sen + c

    def CalcFMap(self, data, param: measurement_param):
        I = data.shape[0]
        J = data.shape[1]

        F = np.zeros(shape=data.shape)

        print("Finish Loading File")
        # ===================================================
        for j in range(0, I):
            F[j] = CalcForceCurveMatrix(data[j], param)

            plt.plot(param.z, F[:,j])
            plt.show()
            print(j + 1, "/", J)
        # F *= 2 * k / f0 * 0.1
        print("Finish Calculating F")
        return F

    # xの単位をAngにする
    def GetIndex(self):
        array = np.zeros(self.I)
        for i in range(0, self.I):
            array[i] = ((self.xOrder / (self.I - 1)) * -i + self.xOrder) / self.xOrder * self.Width
        return array
