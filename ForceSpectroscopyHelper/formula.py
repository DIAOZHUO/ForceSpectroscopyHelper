from ForceSpectroscopyHelper.structures import *





def line_proline(map, xy_point_from, xy_point_to):
    length = int(np.hypot(xy_point_to[0] - xy_point_from[0], xy_point_to[1] - xy_point_from[1]))
    x, y = np.linspace(xy_point_from[0], xy_point_to[0], length), np.linspace(xy_point_from[1], xy_point_to[1], length)
    return map[x.astype(np.int), y.astype(np.int)]




def frequency_shift_to_normalized_frequency_shift(delta_f, param: measurement_param) -> np.ndarray:
    """
     Large amplitudes A0>>d
    :param delta_f:
    :param param:
    :return:
    """
    return delta_f * param.k * (param.amp / 10) ** 1.5 / param.f0








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





