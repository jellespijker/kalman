import matplotlib.pyplot as pl
import numpy as np

# Time vector
DT = 0.2
T = np.arange(0, 100, DT)

# Real values
S = np.zeros((T.size, 1))
V = np.zeros((T.size, 1))
A = np.ones((T.size, 1)) * 9.81

# Build the dataset
for t in range(1, T.size):
    dT = T[t] - T[t - 1]
    V[t] = V[t - 1] + A[t - 1] * dT
    S[t] = S[t - 1] + V[t - 1] * dT + 0.5 * A[t - 1] * dT ** 2

# Known errors
sigma_m = 2.5
sigma_v = 2.5

# Reserve space
x = np.zeros((T.size, 2, 1))
P = np.zeros((T.size, 2, 2))
K = np.zeros((T.size, 2, 2))
xhat = np.zeros((T.size, 2, 1))
z = np.zeros((T.size, 2, 1))
y = np.zeros((T.size, 2, 1))

# Fill measurement array
S_m = np.random.normal(1, 0.1, (T.size,))
S_m = np.multiply(S.transpose(), S_m)
y[:, 0] = S_m.transpose()

# Init Kalman
P[0] = np.matrix([[sigma_m ** 2, 0],
                  [0, sigma_v ** 2]])

u = A
F = np.matrix([[1, DT],
               [0, 1]])
B = np.matrix([[0.5 * DT ** 2],
               [DT]])
H = np.matrix([[1, 0],
               [0, 0]])
Q = np.matrix([[2.5, 0.1],
               [0.0001, 0.00001]])
R = np.matrix([[1000, 5],
               [10, 100]])
v = np.random.normal(0, 0.25, (T.size, 2, 1))
w = np.random.normal(0, 0.25, (T.size, 2, 1))
w[:, 1] = np.zeros((T.size, 1))

for k in range(1, T.size):
    # predict state
    xhat[k] = F * x[k - 1] + B * u[k] + v[k]
    P[k] = F * P[k - 1] * F.transpose() + Q

    # Measurement
    z[k] = H * y[k] + w[k]

    # Update and calculate gain
    K[k] = P[k] * H.transpose() * np.linalg.inv(H * P[k] * H.transpose() + R)
    x[k] = xhat[k] + K[k] * (z[k] - H * xhat[k])

    # current state
    P[k] = (np.eye(2, 2) - K[k] * H) * P[k]

# save csv
datPos = np.zeros((T.size, 3, 1))
datPos[:, 0] = S
datPos[:, 1] = y[:, 0]
datPos[:, 2] = x[:, 0]
np.savetxt('fallingBallPos.dat', datPos, delimiter=',')

datSpeed = np.zeros((T.size, 2, 1))
datSpeed[:, 0] = V
datSpeed[:, 1] = x[:, 1]
np.savetxt('fallingBallSpeed.dat', datSpeed, delimiter=',')

# draw estimates
pl.figure(1)
pl.subplot(211)
lines_true_s = pl.plot(T, S, 'b--')
lines_meas_s = pl.plot(T, y[:, 0], 'b+')
lines_filter_s = pl.plot(T, x[:, 0], 'r')

pl.subplot(212)
lines_true_v = pl.plot(T, V, 'b:')
lines_filter_v = pl.plot(T, x[:, 1], 'r')
pl.savefig('fallingBall.png')
pl.show()
