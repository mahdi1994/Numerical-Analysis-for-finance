import numpy as np
import matplotlib.pyplot as plt


def generate_paths(z, n, m, s0, r, sigma, T):

    dt = T/n
    w = np.zeros([m, n+1])
    s = np.zeros([m, n+1])
    s[:, 0] = np.log(s0)

    for i in range(0, n):
        if m > 1:
            z[:, i] = (z[:, i] - np.mean(z[:, i])) / np.std(z[:, i])

        w[:, i+1] = w[:, i] + (dt**0.5) * z[:,i]
        s[:, i+1] = s[:, i] + (r - 0.5 * sigma**2)*dt + sigma *(w[:, i+1] - w[:, i])

    s_ = np.exp(s)
    paths = {'s': s_, 'w': w}
    return paths


def down_and_out_option(s, L, k, r, T):

    barrier = np.full(s.shape, L)
    down = s < barrier
    # if we hit the barrier once it means that our sum is different from 0
    down_per_path = np.sum(down, axis=1)
    hit_barrier = (down_per_path == 0).astype(int)

    return np.exp(-r*T)*np.mean(payoff(s[:, -1]*hit_barrier))


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
payoff = lambda S: np.maximum(S-k,0.0)

paths = generate_paths(z, n, m, s0, r, sigma, T)
s = paths['s']

price = down_and_out_option(s, L, k, r, T)

print("price of down and out call option at t0 ={0}".format(price))

n1 = np.linspace(10, 100, 10).astype(int)
price1 = np.zeros(len(n1))

for i in range(len(n1)):

    paths1 = generate_paths(z, n1[i], m, s0, r, sigma, T)
    s1 = paths1['s']
    price1[i] = down_and_out_option(s1, L, k, r, T)

plt.figure(1)
plt.grid()
plt.plot(n1, price1)
plt.xlabel('nbr of time steps')
plt.ylabel('Option price')
plt.title('Down and out price option W.R.T nbr of time steps')

# we can conclude that the more we have time steps there is a higher probability that stock price hit the barrier
