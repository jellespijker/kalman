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
    a_tan = np.zeros((num_of_time_steps, 2))
    a_r = np.zeros((num_of_time_steps, 2))
    a = np.zeros((num_of_time_steps, 2))
    theta = np.zeros((num_of_time_steps, 1))
    omega = np.zeros((num_of_time_steps, 1))
    alpha = np.zeros((num_of_time_steps, 1))

    # Build accelaration profile
    a_tan[0:10, 0] = np.linspace(0.1, 1.0, num=10)
    a_tan[0:10, 1] = a_tan[0:10, 0] ** 2
    a_tan[100:110, 0] = np.linspace(-0.1, -2., num=10)
    a_tan[100:110, 1] = a_tan[100:110, 0] ** 2
    a_tan[150:200, 0] = (np.linspace(-2., 8., num=50) ** 2) / 500
    a_tan[150:200, 1] = (a_tan[150:200, 0] ** 3) / 10
    a_tan[300:400, 0] = (np.linspace(8., -10., num=100) ** 3) / 1000
    a_tan[300:400, 1] = (np.linspace(8., -10., num=100) ** 3) / 1000
    a_tan[500:520, 0] = np.linspace(a_tan[400, 0], -1., num=20) ** 2
    a_tan[500:600, 1] = np.linspace(a_tan[400, 1], 1., num=100)
    a_tan[650:700, 0] = -np.linspace(a_tan[600, 0], 0., num=50)
    a_tan[650:700, 1] = -np.linspace(a_tan[600, 1], 0., num=50)

    v = np.cumsum(a_tan, axis=0)  # integrate to gain velocity
    s = np.cumsum(v, axis=0)  # integrate to gain position
    theta = np.tanh(np.divide(v[:, 1], v[:, 0]))  # obtain heading from ds or v
    omega = np.diff(theta, axis=0)  # obtain omega from differentiating theta
    alpha = np.diff(omega, axis=0)  # obtain alpha from differentiating omega

    # All array's to the same length
    t = t[2:num_of_time_steps]
    s = s[2:num_of_time_steps]
    v = v[2:num_of_time_steps]
    a_tan = a_tan[2:num_of_time_steps]
    a_r = a_r[2:num_of_time_steps]
    theta = theta[2:num_of_time_steps]
    omega = omega[1:num_of_time_steps]
    num_of_time_steps -= 2

    a = a_tan + a_r  # ignore acceleration due to rotation for now

    return [t, dt, s, v, a, theta, omega, alpha]


def build_measurement_values(t, real_values):
    # Add disturbance to accelerated forces
    a_m = np.random.normal(np.mean(real_values[0]), np.max(real_values[0]) / 100,
                           (2, len(real_values[0]))).transpose() + \
          real_values[0]

    # Add disturbance and drift to rotational speed
    omega_m = np.random.normal(np.mean(real_values[1]), np.max(real_values[1]) / 40, (len(real_values[1]), 1))[:, 0] + \
              real_values[1]
    drift_mu = 5e-2
    drift_sigma = 1e-3
    drift = np.ones((real_values[1].size,)) * np.random.normal(drift_mu, drift_sigma)
    omega_m += drift

    y = np.zeros((t.size, 9, 1))
    y[:, 4] = np.reshape(a_m[:, 0], (t.size, 1))
    y[:, 5] = np.reshape(a_m[:, 1], (t.size, 1))
    y[:, 7] = np.reshape(omega_m, (t.size, 1))
    return y


def build_control_values(t, real_values):
    u = np.zeros((t.size, 9, 1))
    u[:, 2] = np.reshape(real_values[:, 0], (t.size, 1))
    u[:, 3] = np.reshape(real_values[:, 1], (t.size, 1))
    return u


def init_kalman(t, dt):
    F = np.array([
        [1., 0., 0., 0., 0.5 * dt ** 2, 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0.5 * dt ** 2, 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0, 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0, 0., 0., 1., dt, 0.5 * dt ** 2],
        [0., 0., 0., 0, 0., 0., 0., 1., dt],
        [0., 0., 0., 0, 0., 0., 0., 0., 1]
    ])

    B = np.array([
        [0., 0., dt, 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., dt, 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0.]
    ])

    H = np.array([
        [0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0.]
    ])

    Q = np.diag([2, 2, 0, 0, 0.25, 0.25, 1, 1, 1])

    R = np.array([
        [1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0.0025, 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0.0025, 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.25, 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.25, 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0.5, 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 1.]
    ])
    v = np.random.normal(0, 25e-3, (t.size, 9, 1))
    w = np.random.normal(0, 25e-3, (t.size, 9, 1))
    return [F, B, H, Q, R, v, w]


def kalman(t, kalman_values, u, y, error):
    x = np.zeros((t.size, 9, 1))
    P = np.zeros((t.size, 9, 9))
    P[0, :, :] = np.array([
        [error[0] ** 2, 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., error[0] ** 2, 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., error[1] ** 2, 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., error[1] ** 2, 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., error[2] ** 2, 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., error[2] ** 2, 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., error[3] ** 2, 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., error[4] ** 2, 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., error[5] ** 2]
    ])

    xhat = np.zeros((t.size, 9, 1))
    z = np.zeros((t.size, 9, 1))

    F = kalman_values[0]
    B = kalman_values[1]
    H = kalman_values[2]
    Q = kalman_values[3]
    R = kalman_values[4]
    v = kalman_values[5]
    w = kalman_values[6]

    K = np.zeros((t.size, 9, 9))

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
        P[k] = (np.eye(9) - K[k].dot(H)).dot(P[k])
    return [x, K, P, xhat, z]


def plot_results(t, x, s, v, a, theta, omega, alpha, y, K, P, xhat, z):
    x_ = np.linspace(0, P.shape[1] - 1, P.shape[1])
    y_ = np.linspace(0, P.shape[1] - 1, P.shape[1])
    X_, Y_ = np.meshgrid(x_, y_)

    for i in range(0, 700, 50):
        fig = pl.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X_, Y_, P[i], rstride=1, cstride=1, cmap='hot', linewidth=0, antialiased=False)
        pl.tight_layout()
        pl.show()

    s_err = np.zeros((t.size, 1))
    v_err = np.zeros((t.size, 1))
    a_err = np.zeros((t.size, 1))
    for i in range(1, t.size):
        s_err[i] = 1. - (s[i, 0] / x[i, 0])
        v_err[i] = 1. - (v[i, 0] / x[i, 2])
        a_err[i] = 1. - (a[i, 0] / x[i, 4])
    pl.figure(1)
    pl.subplot(411)
    pl.plot(t, a[:, 0], label='a')
    pl.plot(t, x[:, 4], 'x', label='x')
    pl.plot(t, xhat[:, 4], '.', label='xhat')
    pl.legend()
    pl.subplot(412)
    pl.plot(t, v[:, 0], label='v')
    pl.plot(t, x[:, 2], 'x', label='x')
    pl.plot(t, xhat[:, 2], '.', label='xhat')
    pl.legend()
    pl.subplot(413)
    pl.plot(t, s[:, 0], label='s')
    pl.plot(t, x[:, 0], 'x', label='x')
    pl.plot(t, xhat[:, 2], '.', label='xhat')
    pl.legend()
    pl.subplot(414)
    pl.plot(t, s_err, label='s_err')
    pl.plot(t, v_err, label='v_err')
    pl.plot(t, a_err, label='a_err')
    pl.legend()
    pl.tight_layout()
    pl.show()


def main():
    [t, dt, s, v, a, theta, omega, alpha] = build_real_values()
    y = build_measurement_values(t, [a, omega])
    u = build_control_values(t, v)
    [F, B, H, Q, R, vv, w] = init_kalman(t, dt)
    error = [2, 0.5, 0.5, 0.5, 0.2, 0.4]
    kalman_values = [F, B, H, Q, R, vv, w]
    x, K, P, xhat, z = kalman(t, kalman_values, u, y, error)
    plot_results(t, x, s, v, a, theta, omega, alpha, y, K, P, xhat, z)


if __name__ == '__main__':
    main()
