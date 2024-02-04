<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
</script>


In this example we consider two asset price $$(S_t)_{0 \leq t \leq T}$$ and $$(\tilde{S}_t)_{0 \leq t \leq T}$$ defined by :

$$\tilde{S}_t= S_0e^{(r - \frac{\sigma^2}{2})t + \sigma \tilde{W}_t}$$

where $$ (\tilde{W}_t)_{t\geq 0} $$ is a standard Brownian motion independent of $$ (W_t)_{t\geq0} $$. In what follows, we set

$$W_T = \sqrt{T}G$$ and $$\tilde{W}_T=\sqrt{T}\tilde{G}$$

- Using Monte Carlo we compute the price of the option given :

$$ \Pi = \mathbb{E}[\phi(G,\tilde{G})]  $$

Where

$$
\phi(G, \bar{G}) = \left( \frac{s_0 e^{(r - \frac{\sigma^2}{2} + \sigma \sqrt{T} G)} + s_0 e^{(r - \frac{\sigma^2}{2} + \sigma \sqrt{T} \bar{G})}}{2} - k \right)_+
$$ 

- We use the the formula below to compute the price by adding $$ \lambda$$, $$\tilde{\lambda} $$ $$ \in \mathbb{R}  $$ to reduce the variance :
$$ \Pi = \tilde{\Pi } = e^{-rT}\mathbb{E}(\phi(G + \lambda, \bar{G}+\bar{\lambda})e^{-\lambda G-\frac{-\lambda^2}{2}}e^{-\bar{\lambda} \bar{G}-\frac{-\bar{\lambda}^2}{2}} )$$

- We plot in a three dimensional graph the evolution of the variance of the Monte Carlo method associated to the computation of $$ \bar{\Pi} $$ as function of $$ (\lambda,\bar{\lambda}) $$ and find numerically the optimal $$ (\lambda^*,\bar{\lambda}^*) $$

- We implement a Gradient Descent algorithm to find automatically the couple $$ (\lambda^*,\bar{\lambda}^*) $$ that minimize the empirical variance.
