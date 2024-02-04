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

    paths = {"z": w, "s": s, "time": time}
    return paths


def compute_vega(s, z, time, k, n, m, sigma, T, r):

    s_ = np.exp(s)
    mean_row = np.mean(s_, axis=1)


    temp1 = np.where(mean_row - k > 0, 1, 0)
    t1 = np.zeros([m, n])
    #dt = T / n
    for i in range(1, n):
        t1[:, i] = (s_[:, i]) * (-sigma * time[i] + ((z[:, i])))
    #t2 = (s_) * (-sigma * time + (time**0.5 * z))
    #temp2 = np.mean(t2, axis=1)
    temp3 = np.mean(t1, axis=1)
    vega2 = np.exp(-r*T)*np.mean(temp3*temp1)
    #vega1 = np.exp(-r*T)*np.mean(temp2*temp1)
    return vega2, t1, temp1, temp3

T = 1
s0 = 100
k = 100
sigma = 0.2
r = np.log(1.1)
n = 100
m = 500000
paths = generate_paths(m, n, s0, T, r, sigma)
s = paths['s'][:, 1:]
z = paths['z']
time = paths['time']
vega, t1, temp1, temp3 = compute_vega(s, z, time, k, n, m, sigma, T, r)
