---
title: 'notes-Model Performance'
date: 2021-01-25
permalink: /posts/2021/03/notes-Reinforcement Learning(ADP)/
tags:
  - Models
  - Machine Learning
  - Optimization
---

If we would like to talk about reinforcement learning, we cannot leave without dynamic programming.

Since this project is for energy system. I would like to use the terminology in energy area.(yes, I am still junior in this area.) 

There are three pigments in dynamic programming, one is the state variable. The second is action, sometimes we can use decision(this is usually used in the stochastic programming area). The last is the transition function.



## Background

Our problem can be considered into the economic dispatch problem. If in a more general way, this is a resource allocation problem. We are trying to distribute energy resources( such as electric power grid and other forms of energy allocation wind/gas/heat) to serve different types of demand. [[1]](#1). 





## Model

The problem can be formulated as following:

### State

$S_t=(R_t,W_t)$, 

where $R_t$ is a variable describing the storage amount at time $t$, 

and $W_t$ is the current level of exogenous information, i.e. $W_t=(P_t, D_t)$, here $P_t$ is the price given by our time series model, and $D_t=(Gen_t, Pump_t)$ is the demand for electricity.

### Control/Decision/Action

Consider the decision , we denote $x_t$ as the control/action/decision, where $x_t=(pump_t,gen_t) \in \mathcal{X_t}(R_t,W_t)$, describing the pump and gen amount.

#### $x_t$ 的constraint

At the same time, we have the following feasible region for $x_t$:

1. $Ax_t=b_t(R_t, W_t)$ 当作本身的约束，和storage还有model内部的不确定性

$$
R_{t+1}=R_t+A^{s}x_t, A^{s}=(1,-1),x_t=({gen_t, pump_t})
$$

2. $0 \leq x_t \leq u_t(W_t)$ 自己本身的约束，和model外部的关系(和价格没有关系对把？)

$$
0 \leq gen_t  \leq Gen_t,  0 \leq pump_t \leq Pump_t
$$



#### $S_{t}$的constraint

##### transition model:

we describe the transitions by the state transition model, written by
$$
S_{t+1}=S^{M}(S_t,x_t, \xi_{t+1})
$$
where $\xi=(\xi_1,\cdots,\xi_T)$ and $F_t=\sigma(\xi_1,\cdots,\xi_t)$.

Then we define the probability space $(\Omega,\mathcal{F},\mathcal{P})$ and any random variable indexed by $t$ is $\mathcal{F}_t$-measurable

For $R_{t+1}=R_t+A^{s}x_t+\hat{R}_{t+1}$, we have
$$
R_{t+1}=R_t+gen_t-pump_t
$$
Also,
$$
\underline{R}\leq R_t \leq \bar{R}
$$
For $W_{t+1}=W_t+\hat{W}_{t+1}$, and since $W_t=(P_t, D_t)$, we have
$$
P_{t+1}=P_t+\hat{P}_{t+1}, D_{t+1}=D_t+\hat{D}_{t+1}
$$


##### 特别的要求

we use the concept of the post_decision state denoted $S_{t}^x$, which is the state of the system at time $t$, right after we have choose a decision $x$.

Then $S_t^{x}=(R_t^x,W_t)$, and the corresponding post-decision resource state:
$$
R_t^x=f^{x}(R_t,x_t)=R_t+A^{S}x_t
$$


### Cost function

We have the linear cost function
$$
C(S_t,x_t)=C(W_t)x_t=p_t(gen_t-pump_t)
$$

### Our Goal

We denote $X_t^{\pi}(S_t)$ be a policy that returns a feasible decision vector $x_t$ given the information in $S_t$.
$$
\max_{\pi \in \Pi}F^{\pi}(S_0)=\mathbb{E}\sum_{t=0}^T \gamma^tC(S_t,X_t^{\pi}(S_t))
$$
where the discount factor $\gamma$ may be equal to 1



### Bellman equation

$$
V_t^{\ast}(R_t,W_t)= \max_{x_t \in \mathcal{X}_t(R_t,W_t)}(C_t(R_t,W_t,x_t)+\gamma V_t^{x}(R_t^x, W_t))
$$

and where 
$$
V_t^{x}(R_t^x,W_t)=\mathbb{E}[V_{t+1}^{\ast}(R_{t+1},W_{t+1})|R_t^{x},W_t]
$$






### Value function approximation by piecewise linear value function

#### Optimal value function

The mode can turn to
$$
F_t^{\ast}(v_t(W_t),R_t,W_t)=\max_{x_t,y_t}C_t(R_t,W_t,x_t) +\gamma \sum_{r=1}^{B^R}v_t(r,W_t)y_{tr}
$$
which can equal to
$$
\max_{x_t,y_t} p_t(gen_t-{pump}_t)+\gamma \sum_{r=1}^{B^R}斜率_t y_{tr}
$$
where the breakpoints $R=1,\cdots,B^{R}$ and $斜率=v_t(W_t)=(v_t(1,W_t),\cdots,v_t(B^R,W_t))$

s.t.
$$
\begin{align}
& 0 \leq gen_t  \leq Gen_t, \\ 
& 0 \leq pump_t \leq Pump_t, \\
& R_{t+1}=R_t+gen_t-pump_t \\
& \underline{R}\leq R_t \leq \bar{R} \\
& \sum_{r=1}^{B^R}y_{tr}\rho =f^{x}(R_t,x_t) = R_t^x？ (这个将来可以sample)
\end{align}
$$




Reference:

```
## References
<a id="1">[1]</a> 
Juliana Nascimento,Warren B Powell. (2013). 
An optimal approximate dynamic programming algorithm for the economic dispatch problem with grid-level storage, IEEE Transactions on Automatic Control, to appear print.
```

