import numpy as np
import scipy.stats as st


# Generating m paths with time discretization for monte carlo simulation
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

    paths = {'time': time, "s": s}
    return paths


# Calculate the Numerical expectation of Y = exp(-rT) * exp(1/T * integral(0, T){ log(Su du}
def compute_control_variate_z(s, r, T, k):

    mean_row = np.mean(s, axis=1)
    expected_z = np.mean(np.exp(-r*T)*np.maximum(np.exp(mean_row) - k, 0))
    return expected_z, np.exp(-r*T)*np.maximum(np.exp(mean_row) - k, 0)


def compute_asian_option_price(s, k):

    s_ = np.exp(s)
    mean_row = np.mean(s_, axis=1)
    payoff = np.exp(-r*T)*np.maximum(mean_row - k, 0)

    return payoff

#compute the explicit expectation of the control variate:
T = 1
s0 = 100
k = 100
sigma = 0.2
r = np.log(1.1)
n = 100
m = 100000

#explicit formula of E[1/T * Integrale[0,T]log(su)du]
expectation_cv = np.log(s0) + T*((r/2) - ((sigma**2)/4))
variance_cv = sigma * ((T/3)**0.5)
d = (np.log(k) - expectation_cv) / variance_cv
expectation_z = np.exp(-r*T)*(np.exp((0.5*(variance_cv**2)) +
                                     expectation_cv)*st.norm.cdf(variance_cv - d) - k*st.norm.cdf(-d))

generated_paths = generate_paths(m, n, s0, T, r, sigma)
s = generated_paths['s']

computed_z, z = compute_control_variate_z(s, r, T, k)
option_price = compute_asian_option_price(s, k)

ctrl_variate_coef = -(np.cov(z, option_price)[0, 1] / np.var(z))

print(expectation_z - computed_z)
print(ctrl_variate_coef)
asian_option_price = np.mean(option_price)

asian_option_price_CV = np.mean(asian_option_price +
                                ctrl_variate_coef*(z - expectation_z))

print(asian_option_price_CV - asian_option_price)
print(np.var(option_price))
print(np.var(option_price + ctrl_variate_coef*(z - expectation_z)))
