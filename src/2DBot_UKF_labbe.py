from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise
from numpy import random, dot
from numpy.random import randn
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv


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

    a_tan /= 10

    for i in range(1, num_of_time_steps):
        v[i] = v[i - 1] + a_tan[i] * dt
        s[i] = s[i - 1] + v[i] * dt + 0.5 * a[i] * dt ** 2

    # v = np.cumsum(a_tan, axis=0)  # integrate to gain velocity
    # s = np.cumsum(v, axis=0)  # integrate to gain position
    v[0] = 0.1
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
    a_m = np.random.normal(0, 0.05, (2, len(real_values[0]))).transpose() + real_values[0]

    # Add disturbance and drift to rotational speed
    omega_m = np.random.normal(0, 0.05, (len(real_values[1]), 1))[:, 0] + real_values[1]
    drift_mu = 5e-2
    drift_sigma = 1e-3
    drift = np.ones((real_values[1].size,)) * np.random.normal(drift_mu, drift_sigma)
    omega_m += drift

    z = np.zeros((t.size, 3, ))
    z[:, 0] = np.reshape(a_m[:, 0], (t.size, ))
    z[:, 1] = np.reshape(a_m[:, 1], (t.size, ))
    z[:, 2] = np.reshape(omega_m, (t.size, ))
    return z


def build_control_values(t, real_values):
    u = np.zeros((t.size, 2, ))
    u[:, 0] = np.reshape(real_values[:, 0], (t.size, ))
    u[:, 1] = np.reshape(real_values[:, 1], (t.size, ))
    return u


def init_kalman(t, dt):
    phi_s = 2

    F = np.array([
        [1., 0., 0.5 * dt ** 2, 0., 0., 0., 0., 0., 0.],
        [0., 0., dt, 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0.5 * dt ** 2, 0., 0., 0.],
        [0., 0., 0., 0., 0., dt, 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., ],
        [0., 0., 0., 0., 0., 0., 1., dt, 0.5 * dt ** 2],
        [0., 0., 0., 0., 0., 0., 0., 1., dt],
        [0., 0., 0., 0., 0., 0., 0., 0., 1.]
    ])

    B = np.array([
        [dt, 0.],
        [1., 0.],
        [0., 0.],
        [0., dt],
        [0., 1.],
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.]
    ])

    H = np.array([
        [0., 0., 0.],
        [0., 0., 0.],
        [1., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 1., 0.],
        [0., 0., 0.],
        [0., 0., 1.],
        [0., 0., 0]
    ]).T

    Q = np.array([
        [0.05 * dt ** 5, 0., 0.1666 * dt ** 3, 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0.1666 * dt ** 3, 0., dt, 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0.05 * dt ** 5, 0., 0.1666 * dt ** 3, 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0.1666 * dt ** 3, 0., dt, 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.05 * dt ** 5, 0.125 * dt ** 4, 0.1666 * dt ** 3],
        [0., 0., 0., 0., 0., 0., 0.125 * dt ** 4, 0.333 * dt ** 3, 0.5 * dt ** 2],
        [0., 0., 0., 0., 0., 0., 0.1666 * dt ** 3, 0.5 * dt ** 2, dt]
    ]) * phi_s

    R = np.array([
        [9e-1, 0., 0.],
        [0., 9e-1, 0.],
        [0., 0., 2.e-3],
    ]) * 100
    return [F, B, H, Q, R]


def construct_xground(s, v, a, theta, omega, alpha, shape):
    xground = np.zeros(shape)
    xground[:, 0] = s[:, 0].reshape(s[:, 0].size, )
    xground[:, 3] = s[:, 1].reshape(s[:, 1].size, )
    xground[:, 1] = v[:, 0].reshape(v[:, 0].size, )
    xground[:, 4] = v[:, 1].reshape(v[:, 1].size, )
    xground[:, 2] = a[:, 0].reshape(a[:, 0].size, )
    xground[:, 5] = a[:, 1].reshape(a[:, 1].size, )
    xground[:, 6] = theta.reshape(theta.size, )
    xground[:, 7] = omega.reshape(omega.size, )
    xground[:, 8] = alpha.reshape(alpha.size, )
    return xground


def plot_results(t, x, xground, z, nees):
    plt.figure(0)
    plt.grid(True)
    plt.plot(xground[:, 0], xground[:, 3], '.', label='s')
    plt.plot(x[:, 0], x[:, 3], 'x', label='x')
    plt.ylabel('y [m]')
    plt.xlabel('x [m]')
    plt.legend()
    plt.show()

    plt.figure(1)
    plt.subplot(311)
    plt.plot(t, xground[:, 2], label='a')
    plt.plot(t, x[:, 2], 'x', label='x')
    plt.plot(t, z[:, 0], '.', label='y')
    plt.xlabel('Time [s]')
    plt.ylabel('A_x [m/s^2]')
    plt.legend()
    plt.subplot(312)
    plt.plot(t, xground[:, 1], label='v')
    plt.plot(t, x[:, 1], 'x', label='x')
    plt.xlabel('Time [s]')
    plt.ylabel('v_x [m/s]')
    plt.legend()
    plt.subplot(313)
    plt.plot(t, nees, label='NEES')
    plt.legend()
    plt.tight_layout()
    plt.show()


def NEES(xs, est_xs, ps):
    est_err = xs - est_xs
    err = np.zeros(xs[:, 0].size)
    i = 0
    for x, p in zip(est_err, ps):
        err[i] = (np.dot(x.T, inv(p)).dot(x))
        i += 1
    return err


def f_bot(x, dt, u):
    F = np.array([
        [1., 0., 0.5 * dt ** 2, 0., 0., 0., 0., 0., 0.],
        [0., 0., dt, 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0.5 * dt ** 2, 0., 0., 0.],
        [0., 0., 0., 0., 0., dt, 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., ],
        [0., 0., 0., 0., 0., 0., 1., dt, 0.5 * dt ** 2],
        [0., 0., 0., 0., 0., 0., 0., 1., dt],
        [0., 0., 0., 0., 0., 0., 0., 0., 1.]
    ])
    B = np.array([
        [dt, 0.],
        [1., 0.],
        [0., 0.],
        [0., dt],
        [0., 1.],
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.]
    ])
    return dot(F, x) + dot(B, u)


def h_bot(x):
    H = np.array([
        [0., 0., 0.],
        [0., 0., 0.],
        [1., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 1., 0.],
        [0., 0., 0.],
        [0., 0., 1.],
        [0., 0., 0]
    ]).T
    return dot(H, x)


def main():
    [t, dt, s, v, a, theta, omega, alpha] = build_real_values()
    zs = build_measurement_values(t, [a, omega])
    u = build_control_values(t, v)
    [F, B, H, Q, R] = init_kalman(t, dt)

    sigmas = MerweScaledSigmaPoints(n=9, alpha=.1, beta=2., kappa=-1)
    kf = UKF(dim_x=9, dim_z=3, fx=f_bot, hx=h_bot, dt=0.2, points=sigmas)
    kf.x = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
    kf.R = R
    kf.F = F
    kf.H = H
    kf.Q = Q

    xs, cov = [], []
    for zk, uk in zip(zs, u):
        kf.predict(fx_args=uk)
        kf.update(z=zk)
        xs.append(kf.x.copy())
        cov.append(kf.P)

    xs, cov = np.array(xs), np.array(cov)
    xground = construct_xground(s, v, a, theta, omega, alpha, xs.shape)
    nees = NEES(xground, xs, cov)
    print(np.mean(nees))
    plot_results(t, xs, xground, zs, nees)


if __name__ == '__main__':
    main()
