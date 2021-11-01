from ForceSpectroscopyHelper.structures import *


def spline_smooth(x, y, s=1.0, k=3):
    f = interpolate.UnivariateSpline(x, y, s=s, k=k)
    return f(x)


def average_smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def savitzky_golay_fliter(y, window_size=51, poly=3):
    return savgol_filter(y, window_size, poly)

