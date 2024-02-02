import numpy as np


def path_generation(T, n, nbr_paths, sigma, r, s0):
    # Generation des Gaussienes
    z = np.random.normal(0.0, 1.0, [nbr_paths, n])
    z_bar = np.random.normal(0.0, 1.0, [nbr_paths, n])

    w = np.zeros([nbr_paths, n + 1])
    w_bar = np.zeros([nbr_paths, n + 1])

    x = np.zeros([nbr_paths, n + 1])
    x_bar = np.zeros([nbr_paths, n + 1])

    time = np.zeros(n+1)
    dt = T / n

    x[:, 0] = np.log(s0)
    x_bar[:, 0] = np.log(s0)


    for i in range(0, n):
        if nbr_paths > 1:
            z[:, i] = (z[:, i] - np.mean(z[:, i])) / np.std(z[:, i])
            z_bar[:, i] = (z_bar[:, i] - np.mean(z_bar[:, i])) / np.std(z_bar[:, i])

        w[:, i+1] = w[:, i] + np.power(dt, 0.5) * z[:, i]
        w_bar[:, i + 1] = w_bar[:, i] + np.power(dt, 0.5) * z_bar[:, i]

        x[:, i+1] = x[:, i] + (r - 0.5 * sigma**2) * dt + sigma * (w[:, i+1] - w[:, i])
        x_bar[:, i + 1] = x_bar[:, i] + (r - 0.5 * sigma ** 2) * dt + sigma * (w_bar[:, i + 1] - w_bar[:, i])

        time[i + 1] = time[i] + dt

    s = (np.exp(x) + np.exp(x_bar)) / 2
    paths = {"time":time, "s":s,"z":z}
    return paths

if __name__ == '__main__':

    T = 1
    s0 = 100
    K = 100
    sigma = 0.2
    r = np.log(1.1)
    n = 10
    nbr_paths = 100

    paths = path_generation(T, n, nbr_paths, sigma, r, s0)

    S = paths['s'][:, -1]
    z = paths['z']

    s_max = np.maximum(S-K, 0)

    prix = np.exp(-r * T) * np.mean(s_max)

    print(prix, np.std(s_max))