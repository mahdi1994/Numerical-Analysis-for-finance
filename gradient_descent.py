import numpy as np
import matplotlib.pyplot as plt
from variance_reduction import variance_MC


def var_MC_price(z, z_bar, T, sigma, r, s0, K, mu, mu_bar):

    #w = np.zeros(nbr_paths)
    w = np.power(T, 0.5) * (z + mu)
    w_bar = np.power(T, 0.5) * (z_bar + mu_bar)

    x = np.log(s0) + (r - 0.5*sigma**2)*T + sigma * w
    x_bar = np.log(s0) + (r - 0.5*sigma**2)*T + sigma * w_bar

    s = (np.exp(x) + np.exp(x_bar)) / 2
    s_f = np.maximum(s - K, 0)
    weight1 = np.exp(-mu*z - 0.5 * (mu**2))
    weight2 = np.exp(-mu_bar*z_bar - 0.5 * (mu_bar**2))
    #prix = np.exp(-r*T) * np.mean(s_f*weight2*weight1)
    var = np.var(s_f*weight2*weight1)

    return var

def calculate_gradient(z, z_bar, T, sigma, r, s0, K, mu, mu_bar):

    w = np.power(T, 0.5) * (z + mu)
    w_bar = np.power(T, 0.5) * (z_bar + mu_bar)

    x = np.log(s0) + (r - 0.5 * sigma ** 2) * T + sigma * w
    x_bar = np.log(s0) + (r - 0.5 * sigma ** 2) * T + sigma * w_bar

    s = (np.exp(x) + np.exp(x_bar)) / 2
    s_f = np.maximum(s - K, 0)
    weight1 = np.exp(-2*mu * z - (mu ** 2))
    weight2 = np.exp(-2*mu_bar * z_bar - (mu_bar ** 2))
    weight3 = np.exp(-(mu * z) - 0.5*mu**2)
    weight4 = np.exp(-(mu_bar * z_bar) - 0.5*mu_bar**2)

    grad_V = np.exp(-r*T)*-np.mean(z*(s_f**2)*weight1*weight2)
    grad_V_bar = np.exp(-r*T)*-np.mean(z_bar*(s_f**2)*weight2*weight1)

    return grad_V, grad_V_bar


T = 1
s0 = 100
K = 100
sigma = 0.2
r = np.log(1.1)
nbr_paths = 5000

z = np.random.normal(0.0, 1.0, nbr_paths)
z_bar = np.random.normal(0.0, 1.0, nbr_paths)

z = (z - np.mean(z)) / np.std(z)
z_bar = (z_bar - np.mean(z_bar)) / np.std(z_bar)

a = 0
b = 1

x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)

X, Y = np.meshgrid(x, y)
Z = variance_MC(x, y)


current_position = (a, b, var_MC_price(z, z_bar, T, sigma, r, s0, K, a, b))
learning_rate = 0.0001
delta = 1

ax = plt.subplot(projection="3d", computed_zorder=False)
ax.plot_surface(X, Y, Z, cmap='viridis', zorder=0)

while delta > 0.000001:
    x_derivative, y_derivative = calculate_gradient(z_bar, z, T, sigma, r, s0, K, current_position[0], current_position[1])
    x_new, y_new = current_position[0] - learning_rate * x_derivative, current_position[1] - learning_rate * y_derivative
    current_position = (x_new, y_new, var_MC_price(z, z_bar, T, sigma, r, s0, K, x_new, y_new))
    delta = np.sqrt(x_derivative**2 + y_derivative**2)
    ax.plot_surface(X, Y, Z, cmap='viridis', zorder=0)
    ax.scatter(current_position[0], current_position[1], current_position[2])
    plt.pause(0.001)
    ax.clear()

print(x_new, y_new, var_MC_price(z, z_bar, T, sigma, r, s0, K, x_new, y_new))