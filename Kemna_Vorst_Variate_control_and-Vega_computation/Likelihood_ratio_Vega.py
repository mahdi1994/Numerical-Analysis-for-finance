import numpy as np


#
# Generating m paths with time descritization for monte carlo simulation
def generate_paths(m, n, s0, T, r, sigma):

    z = np.random.normal(0.0, 1.0, [m, n])
    w = np.zeros([m, n + 1])
    s = np.zeros([m, n + 1])
    s[:, 0] = np.log(s0)
    time = np.zeros(n + 1)
    dt = T / n
    for i in range(0, n):
        if m > 1:
            z[:, i] = (z[:, i] - np.mean(z[:, i])) / np.std(z[:, i])
        w[:, i + 1] = w[:, i] + dt ** 0.5 * z[:, i]
        s[:, i + 1] = s[:, i] + (r - 0.5 * sigma ** 2) * dt + sigma * (w[:, i + 1] - w[:, i])
        time[i + 1] = time[i] + dt

    paths = {"z": z, "s": s}
    return paths

def compute_vega2(s, z, m, n, s0, T, r, sigma, k):

    dt = T / n
    s_ = np.exp(s)
    mean_s = np.mean(s_, axis=1)
    temp1 = np.maximum(mean_s - k, 0)
    z_v = (z**2 - 1)/sigma - z*dt
    temp2 = np.sum(z_v, axis=1)
    vega = np.exp(-r*T)*np.mean(temp1*temp2)

    return vega, z_v, temp1

T = 1
s0 = 100
k = 100
sigma = 0.2
r = np.log(1.1)
n = 100
m = 500000
paths = generate_paths(m, n, s0, T, r, sigma)
s = paths['s']
z = paths['z']
vega, z_v, price = compute_vega2(s, z, m, n, s0, T, r, sigma, k)
print(vega)