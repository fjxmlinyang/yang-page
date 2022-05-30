---
title: 'Modulus of continuity&approximated convolution identity'
date: 2020-04-02
permalink: /posts/2020/04/Modulus of continuity&approximated convolution identity/
tags:
  - optimization
  - concepts
---

This is one concept for the smooth in optimziation

* a vector function $g:\mathbb{R}^m \rightarrow \mathbb{R}^s$ admits a modulus of continuity $w_g: \mathbb{R}_{+} \rightarrow \mathbb{R}$ if

$$
\begin{split}
&\lim_{t \rightarrow 0}w_g(t)=0\\
& \|g_i(x+z)-g_i(x)\| \leq w_g(\|z\|)
\end{split}
$$

* Modulus of continuity and nondecreasing										

​							$w_g(\cdot)$ is non-decreasing

* $\mu_N$-adapted:

$$
\lim_{N \rightarrow \infty} \int_{\mathbb{R}^m}w_g(\|z\|)d \mu_N(z)=0
$$

* $\textbf{a proper approximated convolution identity}$ $\{\mu_N\}_{N=1}^{\infty}$: 

  a sequence of measure  which converges weakly to point mass $\delta_0$, and for every $a>0$, $\lim_{N} \mu_N({\mathbb{R}}^{M}\setminus{[-a,a]}^{M})=0$

* We call a proper approximate convolution identyty $\{\mu_N\}$ a $\textbf{strong approximate identity}$, if it satisfies the condition

  $$
  \lim_{N\to\infty} \sqrt{N} \int_{\mathbb{R}^m} \|y\| d \mu_{N}(y)= 0
  $$

