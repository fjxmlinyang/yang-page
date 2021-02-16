---
title: 'On Policy/Off Policy&Morte Carlo/ Temporal Difference'
date: 2020-02-29
permalink: /posts/2020/02/concepts in reinforcement learning/
tags:
  - RL&DL&SO
  - Reinforcement Learning
  - Optimization
---

We discuss some concepts in reinforcement learning

## On Policy/Off Policy&Morte Carlo/ Temporal Difference

In reinforcment learning, there is a very important problem. Under a given policy $\pi$,  how to evaluate the state value function $V^{\pi}(s)$, or the state- action value function  $Q^{\pi}(s,a)$ by the $\textbf{experience}$. This process is policy evaluation

There are two terminology:

* on policy: we evaluate the policy $\pi$ using the experience sampled from polucy $\pi$
* Off policy: evaluate the policy $\pi$ using the experience sampled from a different policy $\pi$
  * The off policy is similar to the important sampling



At the same time, in the situation of model-free, there are two methods to evaluate the policy: 

* $\textbf{Monte Carlo method}$: the key is to have the whole trajectory of the experience $\gamma=<s_1,a_1,r_1,s_2,a_2,r_2, \cdots,s_T, a_T,r_T>$

* $\textbf{Temporal Differennce Method}$: only need is the transition $<s, a, r, s^{’}>$



#### Morte Carlo Policy Evaluation

##### 1.1 Monte Carlo on policy evaluation 

$$
V^{\pi}(s)=E_{\gamma \sim \pi}(R(\gamma)|s_0=s)=\frac{1}{n}\sum_{i=1}^{n}R(\gamma_i)
$$

Where $R(\gamma)=r_1+\gamma r_2+\cdots \gamma^{T-1}r_{T}$

###### Pseudocode:

* Initialize $N(s)=0, G(s)=0, \forall s \in S$
* Loop
* using policy $\pi$ sample trajectory $\gamma_i=<s_{i,1},a_{i,1},r_{i,1},s_{i,2},a_{i,2},r_{i,2}, \cdots,s_T, a_{i,T},r_T{i,T}>$
* $R(\gamma_i)=r_{i,t}+\gamma r_{i,t+1}+\gamma^2 r_{i,t+2}+\cdots+\gamma^{T_i-1}r_{i, T_i}$
* for each state $s$ visted in trajectory $i$:
* if state $s$ is first visited in trajectory $i$:
* Increment counter of total first visits: $N(s)=N(s)+1$
* Increment total return $G(s)+G(s)+G_{i,t}$
* Update estimate: $V^{\pi}=G(s)/N(s)$

##### 1.2 Monte Carlo off policy evaluation 

$$
V^{\pi}(s)=E_{\gamma \sim \pi}(R(\gamma|s_0=s))=\int P(\gamma|\pi)R(\gamma)=\int P(\gamma|\mu)\frac{P(\gamma|\pi)}{\gamma|\mu}R(\gamma)=E_{\gamma \sim \mu}(\frac{P(\gamma|\pi)}{P(\gamma|\mu)}R(\gamma))
$$

Therefore, we sample it from another policy $\mu$
$$
\begin{split}
V^{\pi}(s)&=\frac{1}{n}\sum_{i=1}^{n}(\frac{P(\gamma_i|\pi)}{P(\gamma_i|\mu)}R(\gamma_i))=\frac{1}{n}\sum_{i=1}^{n}\frac{P_{s_{i,1}}\prod_{t=1}^{T_i}\pi(a_{i,t}|s_{i,t})P(s_{i,t+1}|s_{i,t}a_{i,t})}{P_{s_{i,1}}\prod_{t=1}^{T_i}\mu(a_{i,t}|s_{i,t})P(s_{i,t+1}|s_{i,t}a_{i,t})}R(\gamma_i)\\
&=\frac{1}{n}\sum_{i=1}^{n}\frac{P_{s_{i,1}}\prod_{t=1}^{T_i}\pi(a_{i,t}|s_{i,t})}{P_{s_{i,1}}\prod_{t=1}^{T_i}\mu(a_{i,t}|s_{i,t})}R(\gamma_i)\\
\end{split}
$$


It is easy to see the algorithm use the idea of  important sampling



#### Temporal Difference Policy Evaluation

##### 2.1 Temporal Difference on Policy Evaluation

* Set $t=0$, initial state $s_t=s_0$

* Take $a_t \sim \pi(s_t)$

* Observe $(r_t,s_{t+1})$

* Loop

* Sample $\textbf{action}$ $a_{t+1} \sim \pi(s_{t+1})$  % sample action from the $\pi$ itself

* Observe $(r_{t+1},s_{t+2})$

* Update $Q$ given $(s_t, a_t, r_t, s_{t+1},a_{t+1})$  % use $a_{t+1}$
  $$
  Q(s_t,a_t)=Q(s_t,a_t)+\alpha[r_t+\gamma Q(s_{t+1}, a_{t+1})-Q(s_t,a_t)]
  $$

* 

##### 2.2 Temporal Difference off Policy Evaluation

* Set $t=0$, initial state $s_t=s_0$

* Loop

* Sample **action** $a_{t} \sim \pi(s_{t})$  

* Observe $(r_{t},s_{t+1})$

* Update $Q$ given $(s_t, a_t, r_t, s_{t+1})$  % without $a_{t+1}$  **%compare this with the previous one**
  $$
  Q(s_t,a_t)=Q(s_t,a_t)+\alpha_t[r_t+\gamma \max_{a^{'}} Q(s_{t+1}, a^{'})-Q(s_t,a_t)]
  $$

* t=t+1



2.1 is the  SARSA algorithm policy evalution, and it is on-policy

2.2 is the Q-learning algorithm policy evaluation, and it is off-policy

Because SARSA pick the particular action next, while Q-learning pick the max action next(action from a different policy)

#### Conclusion

Under the transition $<s_t,a_t, r_t, s_{t+1},a_{t+1}>$ to evaluate the valuation of the policy $\pi$. 

If we **have $a_t$ and $a_{t+1}$ depends on policy $\pi$**,  we say this policy evaluation is on-policy, otherwise it is off-policy

