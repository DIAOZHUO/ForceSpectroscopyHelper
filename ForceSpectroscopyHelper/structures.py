import numpy as np
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.signal import savgol_filter
from math import sqrt



class measurement_param:

    def __init__(self, max_z, data_count, f0=150000, k=25, amp=17):
        # resonance frequency Hz
        self.f0 = f0
        # spring constant N/m
        self.k = k
        # amplitude(Ang)
        self.amp = amp
        # ang
        self.max_z = max_z
        # force curve data count
        self.data_count = data_count
        # ang
        self.dh = max_z / (data_count - 1)
        # ang
        self.z = np.zeros(data_count)
        for i in range(0, data_count):
            self.z[i] = self.dh * i


class force_curve:

    def __init__(self, x, y):
        self.x = x
        self.y = y



class inflecion_point_param:

    def __init__(self, point):
        self.point = point # index
        self.s_factor = -1
        self.is_well_posed = (self.s_factor >= -1)
        self.wel_posed_boundary = 0 # with unit


class Vector2:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return Vector2(self.x + other, self.y + other)
        else:
            return Vector2(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return Vector2(self.x - other, self.y - other)
        else:
            return Vector2(self.x - other.x, self.y - other.y)

    def __str__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"

    @staticmethod
    def zero():
        return Vector2(0., 0.)

    @staticmethod
    def distance(vec1, vec2) -> float:
        return np.sqrt(np.square(vec1.x - vec2.x) + np.square(vec1.y - vec2.y))

    @staticmethod
    def limited(value: float, range) -> bool:
        if range.x <= value <= range.y:
            return True
        else:
            return False


class Vector3:
    def __init__(self, _x, _y, _z):
        self.x = _x
        self.y = _y
        self.z = _z

    def __str__(self):
        return "(" + str(self.x) + "," + str(self.y) + "," + str(self.z) + ")"
