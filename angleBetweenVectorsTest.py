import numpy as np


def degreeToRad(x):
    return x * np.pi / 180


def radToDegree(x):
    return x * 180 / np.pi


v1 = np.array([1, 1])

angleBetween = degreeToRad(-90)
rotationMatrix = np.array([[np.cos(angleBetween), -np.sin(angleBetween)], [np.sin(angleBetween), np.cos(angleBetween)]])
v2 = np.matmul(rotationMatrix, v1)
print(v2)

ang1 = np.arctan2(v1[1], v1[0])
ang2 = np.arctan2(v2[1], v2[0])
print('ver1: ', radToDegree(ang1), radToDegree(ang2), radToDegree(ang2 - ang1))

ang1Mod = np.arctan2(v1[1], v1[0]) % (2*np.pi)
ang2Mod = np.arctan2(v2[1], v2[0]) % (2*np.pi)
print('ver2: ', radToDegree(ang1Mod), radToDegree(ang2Mod), radToDegree(ang2Mod - ang1Mod))

print('ver3: ', radToDegree(ang1), radToDegree(ang2), radToDegree((ang2 - ang1) % (2*np.pi)))

dot = v1[0] * v2[0] + v1[1] * v2[1]
det = v1[0] * v2[1] - v1[1] * v2[0]
print('ver4: ', radToDegree(np.arctan2(det, dot)))
if np.arctan2(det, dot) < 0:
    print('ver4 +2pi: ', radToDegree(np.arctan2(det, dot) + 2 * np.pi))