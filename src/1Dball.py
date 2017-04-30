import matplotlib.pyplot as pl

pl.rcParams['legend.loc'] = 'best'
import numpy as np
from numpy import dot
from scipy.linalg import inv


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
    S_m = np.random.normal(0., 50, (len(S), 1)) + S
    y[:, 0] = S_m
    return y


def build_control_values(t, a):
    u = np.ones((t.size, 2, 1)) * a[0]
    return u


def init_kalman(t, dt):
    phi_s = 5

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
        [40**2, 0.0],
        [0.0, 15**2]
    ])
    v = np.random.normal(0, 25e-2, (t.size, 2, 1))
    w = np.random.normal(0, 25e-2, (t.size, 2, 1))
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


def NEES(xs, est_xs, ps):
    est_err = xs - est_xs
    err = np.zeros(xs[:, 0].size)
    i = 0
    for x, p in zip(est_err, ps):
        err[i] = (np.dot(x.T, inv(p)).dot(x))
        i += 1
    return err


def plot_results(t, x, xground, a, u, y, K, P, xhat, z, nees):
    pl.figure()
    pl.subplot(311)
    pl.plot(xground[:, 0])
    pl.plot(x[:, 0], '+')
    pl.plot(z[:, 0], '.')
    pl.subplot(312)
    pl.plot(xground[:, 1])
    pl.plot(xhat[:, 1], 'o')
    pl.plot(x[:, 1])
    pl.subplot(313)
    pl.plot(nees)
    pl.tight_layout()
    pl.savefig('fallingBall.png')
    pl.show()


def save_results(t, xground, x, z, nees):
    # save csv
    datPos = np.zeros((t.size, 4, 1))
    datPos[:, 0] = t.reshape((t.size, 1))
    datPos[:, 1] = xground[:, 0]
    datPos[:, 2] = x[:, 0]
    datPos[:, 3] = z[:, 0]
    np.savetxt('fallingBallPos.dat', datPos, delimiter=',')

    datSpeed = np.zeros((t.size, 3, 1))
    datSpeed[:, 0] = t.reshape((t.size, 1))
    datSpeed[:, 1] = xground[:, 1]
    datSpeed[:, 2] = x[:, 1]
    np.savetxt('fallingBallSpeed.dat', datSpeed, delimiter=',')

    datNEES = np.zeros((t.size + 1, 2, 1))
    datNEES[0:t.size, 0] = t.reshape((t.size, 1))
    datNEES[0:t.size, 1] = nees.reshape(t.size, 1)
    datNEES[-1, 0] = t[-1]
    datNEES[-1, 1] = 0.0
    np.savetxt('fallingBall_NEES.dat', datNEES, delimiter=',')

    np.savetxt('fallingBall_meanNEES.dat', [np.mean(nees)], delimiter=',')


def main():
    [t, dt, s, v, a] = build_real_values()
    z = build_measurement_values(t, s)
    u = build_control_values(t, a)
    [F, B, H, Q, R, vv, w] = init_kalman(t, dt)
    error = [40, 10]
    kalman_values = [F, B, H, Q, R, vv, w]
    x, K, P, xhat, y = kalman(t, kalman_values, u, z, error)
    xground = np.zeros(x.shape)
    xground[:, 0] = s
    xground[:, 1] = v
    nees = NEES(xground, x, P)
    print(np.mean(nees))
    save_results(t, xground, x, z, nees)
    plot_results(t, x, xground, a, u, y, K, P, xhat, y, nees)


if __name__ == '__main__':
    main()
