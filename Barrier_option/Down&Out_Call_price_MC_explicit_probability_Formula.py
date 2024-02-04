import numpy as np


def generate_paths(z, n, m, s0, r, sigma, T):

    dt = T/n
    w = np.zeros([m, n+1])
    s = np.zeros([m, n+1])
    s[:, 0] = np.log(s0)

    for i in range(0, n):
        if m > 1:
            z[:, i] = (z[:, i] - np.mean(z[:, i])) / np.std(z[:, i])

        w[:, i+1] = w[:, i] + (dt**0.5) * z[:, i]
        s[:, i+1] = s[:, i] + (r - 0.5 * sigma**2)*dt + sigma *(w[:, i+1] - w[:, i])

    s_ = np.exp(s)
    paths = {'s': s_, 'w': w}
    return paths


def law_of_min(s, T, n, m, L,sigma):

    dt = T/n
    y = np.zeros([m, n])
    for i in range(n):
        for j in range(m):
            if (s[j, i] > L) and (s[j, i+1] > L):
                y[j, i] = 1 - np.exp(-(2*np.log(L/s[j, i])*np.log(L/s[j, i+1]))/(sigma**2*dt))
            else:
                y[j, i] = 0
    vect_product = np.prod(y, axis=1)

    return vect_product

T = 1
s0 = 100
k = 105
sigma = 0.25
L = 90
r = 0.02
n = 100
m = 50000

z = np.random.normal(0.0, 1.0, [m, n])

# Payoff specification
Payoff = lambda S: np.maximum(S-k, 0.0)

paths = generate_paths(z, n, m, s0, r, sigma, T)
s = paths['s']
vect_proba = law_of_min(s, T, n, m, L, sigma)

price = np.exp(-r*T)*np.mean(Payoff(s[:, -1])*vect_proba)
print("price of down and out call option at t0 = {0}".format(price))
