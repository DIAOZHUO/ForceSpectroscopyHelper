import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.signal import savgol_filter

from math import sqrt
from ForceSpectroscopyHelper.structures import *


def spline_smooth(x, y, s=1.0, k=3):
    f = interpolate.UnivariateSpline(x, y, s=s, k=k)
    return f(x)


def average_smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def savitzky_golay_fliter(y, window_size=50, poly=3):
    return savgol_filter(y, window_size, poly)


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


def mldivide(A, B):
    num_vars = A.shape[1]
    rank = np.linalg.matrix_rank(A)
    if rank == num_vars:
        return np.linalg.lstsq(A, B, rcond=None)[0]  # not under-determined
    else:
        for nz in combinations(range(num_vars), rank):  # the variables not set to zero
            try:
                sol = np.zeros((num_vars, 1))
                sol[nz, :] = np.asarray(np.linalg.solve(A[:, nz], B))
                return sol
            except np.linalg.LinAlgError:
                raise ValueError("picked bad variables, can't solve")


def CalcForceCurveSadar(df_curve, param: measurement_param) -> np.ndarray:
    dh = param.dh / 10 # nm
    amp = param.amp / 10 # nm
    Der = np.zeros(param.data_count)
    G = np.zeros(param.data_count)
    F = np.zeros(param.data_count)

    # der = d df_curve / dh
    Der[0] = (df_curve[1] - df_curve[0]) / dh
    Der[param.data_count - 1] = (df_curve[param.data_count - 1] - df_curve[param.data_count - 2]) / dh
    for i in range(1, param.data_count - 1):
        Der[i] = ((df_curve[i + 1] - df_curve[i]) / dh + (df_curve[i] - df_curve[i - 1]) / dh) / 2

    F[0] = 0
    F[1] = 0
    for i in range(2, param.data_count):
        G[i] = 0
        for m in range(i+1, param.data_count):

            G[m] = df_curve[m] + df_curve[m] * sqrt(amp) / (8 * sqrt(3.1415926 * (param.z[m] - param.z[i]) / 10)) \
                   + Der[m] * amp * sqrt(amp) / sqrt(2 * (param.z[m] - param.z[i]) / 10)

        F[i] = integral(i, param.data_count - 1, dh, G)
        F[i] += df_curve[i] * dh + df_curve[i] * 2 * sqrt(amp * dh) / (8 * sqrt(3.1415926)) \
                + Der[i] * 2 * amp * sqrt(amp * dh / 2)
        F[i] *= -2 * param.k / param.f0 # nN
        # F[i][j] = integrate.simps(G[:I-i], z[:I-i])
    return F


def CalcForceCurveMatrix(df_curve, param: measurement_param) -> np.ndarray:
    alpha = round(param.amp / param.dh)
    df_curve = np.flipud(df_curve)
    W = np.zeros(shape=(len(df_curve), len(df_curve)))
    for i in range(0, param.data_count):
        for j in range(0, param.data_count):
            if 0 <= i - j < 2 * alpha:
                W[i, j] = (param.f0 / param.k) * (np.pi / param.amp) \
                          * 2 / (2 * alpha + 1) * \
                          (np.sqrt((2 * alpha + 1) * (i - j + 1) - (i - j + 1) ** 2) - np.sqrt(
                              (2 * alpha + 1) * (i - j) - (i - j) ** 2))
    F = mldivide(W, df_curve)
    return np.flipud(F)



def inflection_point_test(x, F, Amp, z0) -> list:
    x = x - np.min(x)
    dx = x[1]-x[0]
    d1F = np.gradient(F) / dx
    d2F = np.gradient(d1F) / dx
    d3F = np.gradient(d2F) / dx

    # get inflection point in d2F
    point_list = []
    for i in range(0, z0):
        if d2F[i] * d2F[i+1] < 0:
            point_list.append(i)

    point_list = np.asarray(point_list)
    if len(point_list) < 0:
        return None

    param = []
    for i in point_list:
        p = inflecion_point_param(i)
        p.s_factor = x[i] * x[i] / 4 * d3F[i] / d1F[i]
        if p.is_well_posed:
            continue
        if x[i] / np.sqrt(-p.s_factor) / 2 <= Amp < x[i] / 2 and x[i] - 2*Amp > 0:
            p.wel_posed_boundary = x[i] - 2*Amp

    return param





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
            # if j == 44:
            #     plt.plot(np.linspace(0, 513, 513), fft)
            #     plt.show()
            #     plt.plot(x, data[:, j])
            #     plt.plot(x, y)
            #     plt.show()

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
