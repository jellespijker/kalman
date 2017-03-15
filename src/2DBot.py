import math as m

import matplotlib.pyplot as pl
import numpy as np


def build_real_values():
    num_of_time_steps = 700
    dt = 0.2
    t = np.linspace(start=0., stop=num_of_time_steps * dt, num=num_of_time_steps)

    s = np.zeros((num_of_time_steps, 1))

    theta = np.zeros((num_of_time_steps, 1))  # heading
    theta[0] = m.pi / 4  # initial heading

    alpha = np.zeros((num_of_time_steps, 1))
    for i in range(100, num_of_time_steps):
        if i < 200 and i > 100:
            alpha[i] = m.pi / 40000
        elif i < 300 and i > 200:
            alpha[i] = -m.pi / 40000
        elif i < 400 and i > 300:
            alpha[i] = - m.pi / 100000
        elif i < 500 and i > 400:
            alpha[i] = m.pi / 100000
        elif i < 600 and i > 500:
            alpha[i] = m.pi / 2000

    omega = np.zeros((num_of_time_steps, 1))
    for i in range(1, num_of_time_steps):
        omega[i] = omega[i - 1] + alpha[i] * dt

    for i in range(1, num_of_time_steps):  # heading profile
        theta[i] = theta[i - 1] + omega[i] + 1 / 2 * alpha[i] * dt ** 2

    v = np.ones((num_of_time_steps, 2))  # constant velocity
    for i in range(num_of_time_steps):
        v[i] = [m.cos(theta[i]), m.sin(theta[i])]

    s = np.zeros((num_of_time_steps, 2))  # position
    for i in range(1, num_of_time_steps):
        s[i] = s[i - 1] + v[i] * theta[i] * dt

    a_t = np.zeros((num_of_time_steps, 2))
    for i in range(1, num_of_time_steps):
        a_t[i] = -(v[i] - v[i - 1]) / dt

    dy_dx = np.zeros((num_of_time_steps, 1))
    for i in range(1, num_of_time_steps):
        ds = s[i] - s[i - 1]
        dy_dx[i] = ds[0] / ds[1]

    d2y_dx2 = np.zeros((num_of_time_steps, 1))
    for i in range(2, num_of_time_steps, 1):
        d2y_dx2[i] = dy_dx[i] / dy_dx[i - 1]

    #remove first two steps because of deravatives
    num_of_time_steps -= 2
    d2y_dx2 = d2y_dx2[2:]
    dy_dx = dy_dx[2:]
    a_t = a_t[2:]
    s = s[2:]
    v = v[2:]
    theta = theta[2:]
    omega = omega[2:]
    alpha = alpha[2:]
    t = t[2:]

    rho = np.zeros((num_of_time_steps, 1))
    a_n = np.zeros((num_of_time_steps, 2))
    for i in range(num_of_time_steps):
        rho[i] = m.pow(1 + m.pow(dy_dx[i], 2), 3/2) / m.fabs(d2y_dx2[i])

    a_n = np.divide(np.power(v, 2), rho)

    a = a_t + a_n


    pl.figure()
    pl.quiver(s[:,0], s[:,1],a[:,0], a[:,1])
    pl.show()


    return [t, dt, s, v, a, theta, omega, alpha]


def build_measurement_values(real_values):
    a_m = np.random.normal(np.mean(real_values[0]), np.max(real_values[0]) / 40, (2, len(real_values[0]))).transpose() + \
          real_values[0]
    omega_m = np.random.normal(np.mean(real_values[1]), np.max(real_values[1]) / 20, (len(real_values[1]), 1)) + \
              real_values[1]
    return [a_m, omega_m]


def init_kalman(t, dt):
    F = np.array([[1., dt, 0, 0., 0., 0.],
                  [0., 1., 0, 0., 0., 0.],
                  [0., 0., 0., 0., 0., 0.],
                  [0., 0., 0., 1., dt, 0.],
                  [0., 0., 0., 0., 1., 0.],
                  [0., 0., 0., 0., 0., 0.]])

    B = np.array([[0., 0, 0.5 * dt ** 2, 0., 0., 0.],
                  [0., 0., dt, 0., 0., 0.],
                  [0., 0., 0, 1., 0., 0.],
                  [0., 0., 0., 0., 0., 0.5 * dt ** 2],
                  [0., 0., 0., 0., 0., dt],
                  [0., 0., 0., 0., 0., 1]])

    H = np.array([[0., 0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0., 0.],
                  [0., 0., 1., 0., 0., 0.],
                  [0., 0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 1., 0.],
                  [0., 0., 0., 0., 0., 0.]])

    Q = np.array([[1., 0., 0., 0., 0., 0.],
                  [0., 1., 0., 0., 0., 0.],
                  [0., 0., 1., 0., 0., 0.],
                  [0., 0., 0., 1., 0., 0.],
                  [0., 0., 0., 0., 1., 0.],
                  [0., 0., 0., 0., 0., 1.]])

    R = np.array([[1., 0., 0., 0., 0., 0.],
                  [0., 1., 0., 0., 0., 0.],
                  [0., 0., 1., 0., 0., 0.],
                  [0., 0., 0., 1., 0., 0.],
                  [0., 0., 0., 0., 1., 0.],
                  [0., 0., 0., 0., 0., 1.]])

    v = np.random.normal(0, 0.25, (t.size, 6, 1))
    w = np.random.normal(0, 0.25, (t.size, 6, 1))
    return [F, B, H, Q, R, v, w]


def kalman(t, kalman_values, measured_values, error):
    x = np.zeros((t.size, 6, 1))
    P = np.array([[error[0] ** 2, 0., 0., 0., 0., 0.],
                  [0., error[1] ** 2, 0., 0., 0., 0.],
                  [0., 0., error[2] ** 2, 0., 0., 0.],
                  [0., 0., 0., error[3] ** 2, 0., 0.],
                  [0., 0., 0., 0., error[4] ** 2, 0.],
                  [0., 0., 0., 0., 0., error[5] ** 2]])

    xhat = np.zeros((t.size, 6, 1))
    z = np.zeros((t.size, 6, 1))
    y = np.zeros((t.size, 6, 1))

    F = kalman_values[0]
    B = kalman_values[1]
    H = kalman_values[2]
    Q = kalman_values[3]
    R = kalman_values[4]
    v = kalman_values[5]
    w = kalman_values[6]

    K = np.zeros((t.size, 6, 6))

    for k in range(1, t.size):
        # predict state
        xhat[k] = F * x[k - 1] + B * u[k] + v[k]
        P[k] = F * P[k - 1] * F.transpose() + Q

        # Measurement
        z[k] = H * y[k] + w[k]

        # Update and calculate gain
        K[k] = P[k] * H.transpose() * np.linalg.inv(H * P[k] * H.transpose() + R)
        x[k] = xhat[k] + K[k] * (z[k] - H * xhat[k])

        # current state
        P[k] = (np.eye(6) - K[k] * H) * P[k]
    return x


def plot_results(x, real_values, measured_values):
    pl.figure(1)
    pl.subplot(211)
    pl.show()


def main():
    [t, dt, s, v, a, theta, omega, alpha] = build_real_values()
    [a_m, omega_m] = build_measurement_values([a, omega])
    [F, B, H, Q, R, v, w] = init_kalman(t, dt)
    x = kalman([F, B, H, Q, R, v, w], [a_m, omega_m], [0.02, 0.02, 0.04, 0.02, 0.02, 0.04])
    # plot_results(x, [location, velocity, acceleration, rotation], [acceleration_measurements, rotation_measurements])
    pl.figure()
    pl.subplot(211)
    pl.plot(t, a_m)
    pl.subplot(212)
    pl.plot(t, a_m)
    pl.show()


if __name__ == '__main__':
    main()
