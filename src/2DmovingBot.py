import math

import matplotlib.pyplot as pl
import numpy as np

# Time vector
DT = 0.2
T = np.arange(0, 100, DT)

# Build the real values

# position
S = np.zeros((T.size, 2))
S[:, 0] = np.arange(0, 4 * math.pi, 4 * math.pi / T.size)
S[:, 1] = 2 * S[:, 0] * np.sin(S[:, 0])
minY = math.fabs(np.min(S[:, 1]))
S[:, 1] = S[:, 1] + minY
dydx = np.zeros((T.size, 1))
for n in range(1, T.size):
    dydx[n] = (S[n, 1] - S[n - 1, 1]) / (S[n, 0] - S[n - 1, 0])

dy2dx2 = np.zeros((T.size, 1))
for n in range(1, T.size):
    dy2dx2[n] = dydx[n] - dydx[n - 1]

# cheat
dydx[0] = dydx[1]
dy2dx2[0] = dy2dx2[1]

# orientation
theta = np.zeros((T.size, 1))
for n in range(1, T.size):
    theta[n] = math.atan(dydx[n])

# velocity on the X and Y axis
V_m = np.arange(0.1, 2.1, 2 / T.size)
V = np.zeros((T.size, 2))
V[:, 0] = np.multiply((np.cos(theta[:, 0])), V_m)
V[:, 1] = np.multiply((np.sin(theta[:, 0])), V_m)

# calculate the time vector
for n in range(1, T.size):
    T[n] = np.linalg.norm(S[n] - S[n - 1]) * np.linalg.norm(V[n - 1])
T = np.cumsum(T)
print(V)

# acceleration on the X and Y axis
rho = np.zeros((T.size, 1))
for n in range(T.size):
    rho[n] = (math.pow(1 + math.pow(dydx[n], 2), 3 / 2)) / dy2dx2[n]

# acceleration due to rotation
A_n = np.zeros((T.size, 2))
for n in range(T.size):
    A_n[n] = np.power(V[n], 2) / rho[n]

# acceleration due to speed increase
A_t = np.zeros((T.size, 2))
for n in range(1, T.size):
    A_t[n] = V[n] - V[n - 1]
A_t[0] = A_t[1]

# sum of acceleration measured by the accelerometer
A = A_n + A_t

pl.grid(True)
pl.quiver(S[:,0], S[:,1], A[:,0], A[:,1])
#pl.plot(S[:,0], S[:,1])
#pl.plot(T)
#pl.plot(T, np.sqrt(np.power(A[:,0],2)+np.power(A[:,1],2)))
#pl.plot(S[:, 0], theta, 'r')
#pl.plot(S[:,0], S[:,1], 'b')
#pl.plot(S[:,0], np.sqrt(np.power(V[:,0],2)+np.power(V[:,1],2)))
#pl.plot(S[:,0], np.sqrt(np.power(A[:,0],2)+np.power(A[:,1],2)))
# pl.plot(T, V[:, 1])
pl.show()
