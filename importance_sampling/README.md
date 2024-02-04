In this example we consider two asset price <span class="tex2jax">$(S_t)_{0 \leq t \leq T}$</span> and <span class="tex2jax">$(\tilde{S}_t)_{0 \leq t \leq T}$</span> defined by :

$$
\tilde{S}_t= S_0e^{(r - \frac{\sigma^2}{2})t + \sigma \tilde{W}_t}
$$

where <span class="tex2jax">$(\tilde{W}_t)_{t\geq 0}$</span> is a standard Brownian motion independent of <span class="tex2jax">$(W_t)_{t\geq0}$</span>. In what follows, we set

<span class="tex2jax">$W_T = \sqrt{T}G$$ and $$\tilde{W}_T=\sqrt{T}\tilde{G}$</span>

- Using Monte Carlo we compute the price of the option given :

$$ 
\Pi = \mathbb{E}[\phi(G,\tilde{G})]  
$$

Where

$$
\phi(G, \bar{G}) = \left( \frac{s_0 e^{(r - \frac{\sigma^2}{2} + \sigma \sqrt{T} G)} + s_0 e^{(r - \frac{\sigma^2}{2} + \sigma \sqrt{T} \bar{G})}}{2} - k \right)_+
$$ 

- We use the the formula below to compute the price by adding <span class="tex2jax">$ \lambda$$, $$\tilde{\lambda} $</span> <span class="tex2jax">$\in \mathbb{R}$</span> to reduce the variance :
$$ \Pi = \tilde{\Pi } = e^{-rT}\mathbb{E}(\phi(G + \lambda, \bar{G}+\bar{\lambda})e^{-\lambda G-\frac{-\lambda^2}{2}}e^{-\bar{\lambda} \bar{G}-\frac{-\bar{\lambda}^2}{2}} )$$

- We plot in a three dimensional graph the evolution of the variance of the Monte Carlo method associated to the computation of <span class="tex2jax">$\bar{\Pi}$</span> as function of <span class="tex2jax">$ (\lambda,\bar{\lambda}) $</span> and find numerically the optimal <span class="tex2jax">$(\lambda^*,\bar{\lambda}^*)$</span>

- We implement a Gradient Descent algorithm to find automatically the couple <span class="tex2jax">$(\lambda^*,\bar{\lambda}^*)$</span> that minimize the empirical variance.

- <span class="tex2jax">$(W_t)_{t \geq 0 }$</span>
