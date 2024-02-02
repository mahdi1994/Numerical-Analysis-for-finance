import numpy as np
import matplotlib.pyplot as plt

nbr_paths = 1000
T = 1
sigma = 0.2
r = np.log(1.1)
s0 = 100
K = 100

# generation de deux variables aléatoires pour les mvts browniens
z = np.random.normal(0.0, 1.0, nbr_paths)
z_bar = np.random.normal(0.0, 1.0, nbr_paths)

# deux varaible de réduction de variances
mu = np.arange(0, 1, 0.05)
mu_bar = np.arange(0, 1, 0.05)

def variance_MC(mu, mu_bar):

    z_ = z.reshape(-1, 1)
    z_bar_ = z_bar.reshape(-1, 1)
    w = np.power(T, 0.5) * (z_ + mu)
    w_bar = np.power(T, 0.5) * (z_bar_ + mu_bar)

    x = np.log(s0) + (r - 0.5 * sigma**2) * T + sigma * w
    x_bar = np.log(s0) + (r - 0.5 * sigma**2) * T + sigma * w_bar

    s = np.zeros([len(mu), len(mu_bar)])
    n = len(mu)

    for i in range(n):
        for j in range(len(mu_bar)):
            M = (np.exp(x[:, j]) + np.exp(x_bar[:, i])) / 2
            weight1 = np.exp(-(mu[j]*z)) - 0.5 * (mu[j] **2)
            weight2 = np.exp(-(mu[i]*z_bar)) - 0.5 * (mu[i] **2)

            s_option = np.maximum(M - K, 0)
            s[i, j] = np.var(s_option*weight1*weight2)

    return s


# main
X, Y = np.meshgrid(mu, mu_bar)
Z = variance_MC(mu, mu_bar)
fig = plt.figure()

ax = plt.subplot(projection="3d")
ax.plot_surface(X, Y, Z, cmap='viridis')

# Add labels
ax.set_xlabel('mu-axis')
ax.set_ylabel('mu_bar-axis')
ax.set_zlabel('Variance-axis')
ax.set_title('3D Plot of variance ')

valeur_min = np.min(Z)

# Trouver les indices correspondants à la valeur minimale
indices_min = np.unravel_index(np.argmin(Z), Z.shape)

# Affichage de la valeur minimale et de ses indices
print("Valeur minimale :", np.sqrt(valeur_min))
print("Indices de la valeur minimale :", indices_min)