---
title: 'Two Concepts of convergence'
date: 2019-12-01
permalink: /posts/2019/12/Two Concepts of convergence/
tags:
  - RL&DL&SO
  - Optimziation
---

Here we discuss two concepts in proving consistency.

#### Uniform Normaized(UN) convergence of random variables

Let $(\Omega,\Sigma_{\Omega},P)$ be a probability space 

$X$ be a seperable metric space with distance function  $\rho(\cdot,\cdot)$

A sequence of random variables $\theta_n: \Omega \rightarrow X$, $n=1,2, \cdots,$ is UN convergent to a random variable $\theta: \Omega \rightarrow X$ , with rate(of a numerical sequence) $\frac{1}{v_n}$ and distribution $\Phi$, 

if there is a sequence of numbers $v_n \rightarrow \infty$, $0 \leq v_n \leq \infty$, and (left continuous) distribution function $\Psi: \mathbb{R}^1 \rightarrow [0,1]$ such that

$$
Pr \{v_n \rho(\theta_n, \theta)<t\} \geq \Psi(t), \forall t>0, n=1,2,\cdots
$$

## 





#### UN-Convergence

If in the previous definition, inequality $Pr \{v_n \rho(\theta_n, \theta)<t\} \geq \Psi(t), \forall t>0, n=1,2,\cdots$ is fulfilled uniformly in $t$, i.e., for any $\epsilon>0$ and all $n \geq n(\epsilon)$
$$
{\lim\inf}_{n \rightarrow \infty} Pr\{v_n\rho(\theta_n,\theta)<t\}\geq \Phi(t)-\epsilon, \ , \forall t \in \mathbb{R}^1
$$

take place, and then we say that $\{\theta_n\}$ is uniformly normalized convergent to $\theta$ and denote $\theta_n \longrightarrow^{UN} \theta$



####Some Notation

Let $\xi^n=\{\xi_1(\omega),\cdots, \xi_n(\omega)\}$ be a set of i.i.d. random variables with the same distribution as $\xi(\omega)$, and denote $$\begin{equation} aaaaa\end{equation}$$

