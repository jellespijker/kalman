import matplotlib.pyplot as pl

pl.rcParams['legend.loc'] = 'best'
import numpy as np
from numpy import dot


def build_real_values():
    num_of_time_steps = 100
    dt = 0.2
    t = np.linspace(start=0., stop=num_of_time_steps * dt, num=num_of_time_steps)
    s = np.zeros((num_of_time_steps, 1))
    v = np.zeros((num_of_time_steps, 1))
    a = np.zeros((num_of_time_steps, 1))

    a = np.ones((num_of_time_steps, 1)) * 9.81  # Build accelaration profile
    for i in range(1, num_of_time_steps):
        v[i] = v[i - 1] + a[i] * dt  # Build speed profile
        s[i] = s[i - 1] + v[i] * dt + 0.5 * a[i] * dt ** 2  # Build position profile

    return [t, dt, s, v, a]


def build_measurement_values(t, S):
    y = np.zeros((t.size, 2, 1))
    S_m = np.random.normal(1, 20, (len(S), 1)) + S
    y[:, 0] = S_m
    return y


def build_control_values(t, a):
    u = np.ones((t.size, 2, 1)) * a[0]
    return u


def init_kalman(t, dt):
    phi_s = 5e-3

    F = np.array([
        [1., dt],
        [0., 1.]
    ])

    B = np.array([
        [0.5 * dt ** 2, 0.],
        [0, dt]
    ])

    H = np.array([
        [1., 0.],
        [0., 0.0]
    ])

    Q = np.array([
        [(1 / 3) * dt ** 3, 0.5 * dt ** 2],
        [0.5 * dt ** 2, dt]
    ]) * phi_s

    R = np.array([
        [5.0, 0.0],
        [0.0, 5.0e-9]
    ])
    v = np.random.normal(0, 25e-3, (t.size, 2, 1))
    w = np.random.normal(0, 25e-3, (t.size, 2, 1))
    return [F, B, H, Q, R, v, w]


def kalman(t, kalman_values, u, z, error):
    x = np.zeros((t.size, 2, 1))
    P = np.zeros((t.size, 2, 2))
    P[0, :, :] = np.array([
        [error[0] ** 2, 0.],
        [0., error[0] ** 2]
    ])

    xhat = np.zeros((t.size, 2, 1))
    y = np.zeros((t.size, 2, 1))

    F = kalman_values[0]
    B = kalman_values[1]
    H = kalman_values[2]
    Q = kalman_values[3]
    R = kalman_values[4]
    v = kalman_values[5]
    w = kalman_values[6]

    K = np.zeros((t.size, 2, 2))

    for k in range(1, t.size):
        xhat[k] = dot(F, x[k - 1]) + dot(B, u[k]) + w[k]
        Phat = dot(F, dot(P[k - 1], F.T)) + Q

        y[k] = z[k] - dot(H.T, xhat[k])

        S = dot(H.T, dot(Phat, H)) + R
        S = np.linalg.inv(S)
        K[k] = dot(Phat, dot(H, S))
        x[k] = xhat[k] + dot(K[k], y[k])

        P[k] = dot(np.eye(2) - dot(K[k], H.T), Phat)
    return [x, K, P, xhat, z]


def plot_results(t, x, s, v, a, u, y, K, P, xhat, z):
    res_s = np.reshape(s[:, 0], (t.size, 1)) - x[:, 0]
    res_v = np.reshape(v[:, 0], (t.size, 1)) - x[:, 1]

    pl.figure()
    pl.subplot(311)
    pl.plot(s)
    pl.plot(xhat[:, 0], 'o')
    pl.plot(x[:, 0])
    pl.plot(y[:, 0], '.')
    pl.subplot(312)
    pl.plot(v)
    pl.plot(xhat[:, 1], 'o')
    pl.plot(x[:, 1])
    pl.subplot(313)
    pl.plot(res_s)
    pl.plot(res_v)
    pl.tight_layout()
    pl.savefig('fallingBall.png')
    pl.show()

    # save csv
    datPos = np.zeros((t.size, 4, 1))
    datPos[:, 0] = t.reshape((t.size, 1))
    datPos[:, 1] = s
    datPos[:, 2] = z[:, 0]
    datPos[:, 3] = x[:, 0]
    np.savetxt('fallingBallPos.dat', datPos, delimiter=',')

    datSpeed = np.zeros((t.size, 3, 1))
    datSpeed[:, 0] = t.reshape((t.size, 1))
    datSpeed[:, 1] = v
    datSpeed[:, 2] = x[:, 1]
    np.savetxt('fallingBallSpeed.dat', datSpeed, delimiter=',')

def main():
    [t, dt, s, v, a] = build_real_values()
    z = build_measurement_values(t, s)
    u = build_control_values(t, a)
    [F, B, H, Q, R, vv, w] = init_kalman(t, dt)
    error = [2.5, 2.5]
    kalman_values = [F, B, H, Q, R, vv, w]
    x, K, P, xhat, y = kalman(t, kalman_values, u, z, error)
    plot_results(t, x, s, v, a, u, y, K, P, xhat, y)


if __name__ == '__main__':
    main()
