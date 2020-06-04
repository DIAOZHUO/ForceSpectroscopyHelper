import scipy.interpolate as interpolate
import matplotlib.pyplot as plt

from math import sqrt
from ForceSpectroscopyHelper.structures import *


def smooth_curve(x, y, s=1.0, k=3):
    f = interpolate.UnivariateSpline(x, y, s=s, k=k)
    return f(x)


def integral(m, n, h, f):
    if (n - m) % 2 == 0:
        S = (f[m] - f[n]) * h / 3
        for i in range(1, int((n - m) / 2) + 1):
            S += (4 * f[m + 2 * i - 1] + 2 * f[m + 2 * i]) * h / 3
    else:
        S = (f[m] - f[n - 1]) * h / 3
        for i in range(1, int((n - 1 - m) / 2) + 1):
            S += (4 * f[m + 2 * i - 1] + 2 * f[m + 2 * i]) * h / 3
        S += (f[n - 1] + f[n]) * h / 2
    return S


def CalcForceCurve(df_curve, param: measurement_param):
    Der = np.zeros(param.data_count)
    G = np.zeros(param.data_count)
    F = np.zeros(param.data_count)

    # der = d df_curve / dh
    Der[0] = (df_curve[1] - df_curve[0]) / param.dh
    Der[param.data_count - 1] = (df_curve[param.data_count - 1] - df_curve[param.data_count - 2]) / param.dh
    for i in range(1, param.data_count - 1):
        Der[i] = ((df_curve[i + 1] - df_curve[i]) / param.dh + (df_curve[i] - df_curve[i - 1]) / param.dh) / 2

    F[0] = 0
    F[1] = 0
    for i in range(2, param.data_count):
        # do int
        G[i] = 0
        for m in range(i+1, param.data_count):

            G[m] = df_curve[m] + df_curve[m] * sqrt(param.amp) / (8 * sqrt(3.1415926 * (param.z[m] - param.z[i]))) \
                   + Der[m] * param.amp * sqrt(param.amp) / sqrt(2 * (param.z[m] - param.z[i]))

        F[i] = integral(i, param.data_count - 1, param.dh, G)
        F[i] += df_curve[i] * param.dh + df_curve[i] * 2 * sqrt(param.amp * param.dh) / (8 * sqrt(3.1415926)) \
                + Der[i] * 2 * param.amp * sqrt(param.amp * param.dh / 2)
        F[i] *= -2 * param.k / param.f0 * 0.1
        # F[i][j] = integrate.simps(G[:I-i], z[:I-i])
    return F

#
# def CalcForceCurve(df_curve, param: measurement_param):
#     Der = np.zeros(param.data_count)
#     G = np.zeros(param.data_count)
#     F = np.zeros(param.data_count)
#
#     # der = d df_curve / dh
#     Der[0] = (df_curve[1] - df_curve[0]) / param.dh
#     Der[param.data_count - 1] = (df_curve[param.data_count - 1] - df_curve[param.data_count - 2]) / param.dh
#     for i in range(1, param.data_count - 1):
#         Der[i] = ((df_curve[i + 1] - df_curve[i]) / param.dh + (df_curve[i] - df_curve[i - 1]) / param.dh) / 2
#
#     F[0] = 0
#     F[1] = 0
#     for i in range(2, param.data_count):
#         # do int
#         G[i] = 0
#         for m in range(i + 1, param.data_count):
#             G[m] = df_curve[m] + df_curve[m] * sqrt(param.amp) / (8 * sqrt(3.1415926 * (param.z[m] - param.z[i]))) \
#                    + Der[m] * param.amp * sqrt(param.amp) / sqrt(2 * (param.z[m] - param.z[i]))
#
#         F[i] = integral(i, param.data_count - 1, param.dh, G)
#         F[i] += df_curve[i] * param.dh + df_curve[i] * 2 * sqrt(param.amp * param.dh) / (8 * sqrt(3.1415926)) \
#                 + Der[i] * 2 * param.amp * sqrt(param.amp * param.dh / 2)
#         F[i] *= -2 * param.k / param.f0 * 0.1
#         # F[i][j] = integrate.simps(G[:I-i], z[:I-i])
#     return F



def SmoothMap(data, N):
        I = data.shape[0]
        J = data.shape[1]
        data = np.array(data)
        ResultData = np.zeros((I,J))
        for j in range(0, J):
            for i in range(int((N - 1) / 2), int(I - (N - 1) / 2)):
                ResultData[i][j] = 0
                for m in range(int(i - (N - 1) / 2), int(i + (N - 1) / 2 + 1)):
                    ResultData[i][j] += data[m][j]
                ResultData[i][j] = ResultData[i][j] / N

            for i in range(0, int((N - 1) / 2)):
                ResultData[i][j] = 0
                for m in range(0, int(i + (N - 1) / 2 + 1)):
                    ResultData[i][j] += data[m][j]
                ResultData[i][j] = ResultData[i][j] / ((N - 1) / 2 + i + 1)

            for i in range(int(I - (N - 1) / 2), I):
                ResultData[i][j] = 0
                for m in range(int(i - (N - 1) / 2), I):
                    ResultData[i][j] += data[m][j]
                ResultData[i][j] = ResultData[i][j] / ((N - 1) / 2 + I - i)
        return ResultData


def FFTMap(data, lowPass, x):
        I = data.shape[0]
        J = data.shape[1]
        data = np.array(data)
        window = np.flipud(np.hamming(I))
        ResultData = np.zeros((I, J))
        for j in range(0, J):
            fft = np.fft.rfft(data[:, j] * window)
            fft[lowPass:] = 0
            y = np.fft.irfft(fft) / window
            ResultData[:, j] = y
            if j == 44:
                plt.plot(np.linspace(0, 513, 513), fft)
                plt.show()
                plt.plot(x, data[:, j])
                plt.plot(x, y)
                plt.show()

        # plt.imshow(data)
        # plt.colorbar()
        # plt.show()
        # plt.imshow(ResultData)
        # plt.colorbar()
        # plt.show()
        return ResultData


def FFT2Map(data, lowPass):
        I = data.shape[0]
        J = data.shape[1]
        data = np.array(data)
        window = np.flipud(np.hamming(I) * np.hamming(J))
        fft = np.fft.rfft2(data * window)
        plt.imshow(fft)
        plt.colorbar()
        plt.show()

        fft[lowPass:] = 0
        im = np.fft.irfft2(fft) / window

        plt.imshow(im)
        plt.colorbar()
        plt.show()
        return im
