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

and $W_t$ is the current level of exogenous information, i.e. $W_t=(P_t, D_t)$, here $P_t$ is the price given by our time series model, and $D_t$ is the demand for electricity.

### Control/Decision/Action

Consider the decision , we denote $x_t$ as the control/action/decision, where $x_t=(pump_t,gen_t) \in \mathcal{X_t}(R_t,W_t)$.

At the same time, we have the following feasible region:













Reference:

```
## References
<a id="1">[1]</a> 
Juliana Nascimento,Warren B Powell. (2013). 
An optimal approximate dynamic programming algorithm for the economic dispatch problem with grid-level storage, IEEE Transactions on Automatic Control, to appear print.
```

