import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
pl.rcParams['legend.loc'] = 'best'
import numpy as np


def build_real_values():
    num_of_time_steps = 702
    dt = 0.2
    t = np.linspace(start=0., stop=num_of_time_steps * dt, num=num_of_time_steps)
    s = np.zeros((num_of_time_steps, 2))
    v = np.zeros((num_of_time_steps, 2))
    a = np.zeros((num_of_time_steps, 2))

    a = np.ones((num_of_time_steps, 1)) * 9.81 # Build accelaration profile
    v = np.cumsum(a, axis=0)  # integrate to gain velocity
    s = np.cumsum(v, axis=0)  # integrate to gain position

    return [t, dt, s, v, a]


def build_measurement_values(t, S):
    y = np.zeros((t.size, 2, 1))
    S_m = S * np.random.normal(1, 0.02, (1, len(S))).transpose()
    y[:, 0] = S_m
    return y


def build_control_values(t, a):
    u = np.ones((t.size, 2, 1)) * a[0]
    return u


def init_kalman(t, dt):
    F = np.array([
        [1., 0.],
        [0., 1.]
    ])

    B = np.array([
        [dt, 0.5 * dt ** 2],
        [0, dt]
    ])

    H = np.array([
        [1., 0.],
        [0., 0.]
    ])

    Q = np.array([
        [1., .1],
        [.1, 1.]
    ])

    R = np.array([
        [1.2, 0.2],
        [0.2, 0.2]
    ])
    v = np.random.normal(0, 25e-3, (t.size, 2, 1))
    w = np.random.normal(0, 25e-3, (t.size, 2, 1))
    return [F, B, H, Q, R, v, w]


def kalman(t, kalman_values, u, y, error):
    x = np.zeros((t.size, 2, 1))
    P = np.zeros((t.size, 2, 2))
    P[0, :, :] = np.array([
        [error[0] ** 2, 0.],
        [0., error[0] ** 2]
    ])

    xhat = np.zeros((t.size, 2, 1))
    z = np.zeros((t.size, 2, 1))

    F = kalman_values[0]
    B = kalman_values[1]
    H = kalman_values[2]
    Q = kalman_values[3]
    R = kalman_values[4]
    v = kalman_values[5]
    w = kalman_values[6]

    K = np.zeros((t.size, 2, 2))

    for k in range(1, t.size):
        # predict state
        xhat[k] = F.dot(x[k - 1]) + B.dot(u[k]) + w[k]
        P[k] = F.dot(P[k - 1]).dot(F.transpose()) + Q

        # Measurement
        z[k] = H.dot(y[k]) + v[k]

        # Update and calculate gain
        K[k] = P[k].dot(H.transpose()).dot(np.linalg.inv(H.dot(P[k]).dot(H.transpose()) + R))
        x[k] = xhat[k] + K[k].dot(z[k] - H.dot(xhat[k]))

        # current state
        P[k] = (np.eye(2) - K[k].dot(H)).dot(P[k])
    return [x, K, P, xhat, z]


def plot_results(t, x, s, v, a, u, y, K, P, xhat, z):
    x_ = np.linspace(0, P.shape[1] - 1, P.shape[1])
    y_ = np.linspace(0, P.shape[1] - 1, P.shape[1])
    X_, Y_ = np.meshgrid(x_, y_)

    for i in range(0, 10, 1):
        fig = pl.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X_, Y_, K[i], linewidth=1, antialiased=True)
        pl.tight_layout()
        pl.show()

    s_err = np.zeros((t.size, 1))
    v_err = np.zeros((t.size, 1))
    for i in range(1, t.size):
        s_err[i] = 1. - (s[i] / x[i, 0])
        v_err[i] = 1. - (v[i] / x[i, 1])

    pl.figure()
    pl.subplot(311)
    pl.plot(s)
    pl.plot(xhat[:,0],'o')
    pl.plot(x[:,0])
    pl.plot(y[:,0],'.')
    pl.subplot(312)
    pl.plot(v)
    pl.plot(xhat[:,1],'o')
    pl.plot(x[:,1])
    pl.subplot(313)
    pl.plot(s_err)
    pl.plot(v_err)
    pl.tight_layout()
    pl.show()

def main():
    [t, dt, s, v, a] = build_real_values()
    y = build_measurement_values(t, s)
    u = build_control_values(t, a)
    [F, B, H, Q, R, vv, w] = init_kalman(t, dt)
    error = [0.25, 2.5]
    kalman_values = [F, B, H, Q, R, vv, w]
    x, K, P, xhat, z = kalman(t, kalman_values, u, y, error)
    plot_results(t, x, s, v, a, u, y, K, P, xhat, z)


if __name__ == '__main__':
    main()
