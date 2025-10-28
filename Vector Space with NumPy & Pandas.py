import numpy as np
import pandas as pd


class R2Vector:
    def __init__(self, x=0, y=0):
        self.vec = np.array([x, y])

    def __repr__(self):
        return f"R2Vector({self.vec[0]}, {self.vec[1]})"

    def __add__(self, other):
        return R2Vector(*(self.vec + other.vec))

    def __sub__(self, other):
        return R2Vector(*(self.vec - other.vec))

    def dot(self, other):
        return np.dot(self.vec, other.vec)

    def to_pandas(self):
        return pd.Series(self.vec, index=['x', 'y'])


class R3Vector:
    def __init__(self, x=0, y=0, z=0):
        self.vec = np.array([x, y, z])

    def __repr__(self):
        return f"R3Vector({self.vec[0]}, {self.vec[1]}, {self.vec[2]})"

    def __add__(self, other):
        return R3Vector(*(self.vec + other.vec))

    def __sub__(self, other):
        return R3Vector(*(self.vec - other.vec))

    def dot(self, other):
        return np.dot(self.vec, other.vec)

    def cross(self, other):
        return R3Vector(*np.cross(self.vec, other.vec))

    def to_pandas(self):
        return pd.Series(self.vec, index=['x', 'y', 'z'])


class RNVector:
    def __init__(self, *components):
        self.vec = np.array(components)

    def __repr__(self):
        return f"RNVector({', '.join(map(str, self.vec))})"

    def __add__(self, other):
        return RNVector(*(self.vec + other.vec))

    def __sub__(self, other):
        return RNVector(*(self.vec - other.vec))

    def dot(self, other):
        return np.dot(self.vec, other.vec)

    def norm(self):
        return np.linalg.norm(self.vec)

    def to_pandas(self):
        return pd.Series(self.vec, index=[f'x{i}' for i in range(len(self.vec))])


class PolarVector:
    def __init__(self, r, theta_rad):
        self.r = r
        self.theta = theta_rad

    def __repr__(self):
        return f"PolarVector(r={self.r}, θ={self.theta} rad)"

    def to_cartesian(self):
        x = self.r * np.cos(self.theta)
        y = self.r * np.sin(self.theta)
        return R2Vector(x, y)


class HomogeneousVector:
    def __init__(self, *components):
        self.vec = np.array(list(components) + [1])

    def __repr__(self):
        return f"HomogeneousVector({', '.join(map(str, self.vec))})"

    def to_cartesian(self):
        return RNVector(*(self.vec[:-1] / self.vec[-1]))


# 🧪 Example Usage
v2a = R2Vector(3, 4)
v2b = R2Vector(1, 2)
print(f'v2a + v2b = {v2a + v2b}')
print(f'v2a • v2b = {v2a.dot(v2b)}')
print(v2a.to_pandas())

v3a = R3Vector(2, 3, 1)
v3b = R3Vector(0.5, 1.25, 2)
print(f'v3a × v3b = {v3a.cross(v3b)}')
print(v3a.to_pandas())

vn1 = RNVector(1, 2, 3, 4)
vn2 = RNVector(4, 3, 2, 1)
print(f'vn1 • vn2 = {vn1.dot(vn2)}')
print(vn1.to_pandas())

polar = PolarVector(5, np.pi / 4)
print(f'Polar to Cartesian: {polar.to_cartesian()}')

hvec = HomogeneousVector(2, 4, 6)
print(f'Homogeneous to Cartesian: {hvec.to_cartesian()}')
