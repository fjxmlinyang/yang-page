---

title: 'notes-Reinforcement Learning(ADP)'
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

2. $0 \leq x_t \leq u_t(W_t)$ 自己本身的约束，和model外部的关系(和价格没有关系对把？)（换成一个）

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

For $R_{t+1}=R_t+A^{s}x_t+\hat{R}_{t+1}$, we have（相当于你之前flow流到这个time $t$的总量）（加上effieiecny）
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

Then $S_t^{x}=(R_t^x,W_t)$, and the corresponding post-decision resource state:（swith max &E）
$$
R_t^x=f^{x}(R_t,x_t)=R_t+A^{S}x_t
$$

in our case this is ( 相当于你在time $t$有的量，给将来$t+1$使用)
$$
R_{t}^x=R_t?
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

注意这里，你来自未来的potential value是不固定的所以可以由下面这个式子决定。那我们需要的就是，
$$
\bar{V}_t(R_t^x, W_t) \rightarrow V_t^x(R_t^x,W)
$$
where $\bar{V}_t(R_t^x,W_t)$ is a piecewise linear approximation function.(就是后面的$\sum_{r=1}^{B^R}v_t(r,W_t)y_{tr}$, 注意这里是一个和x没有任何关系的一个式子（换成是$y_{t}$））

### Value function approximation by piecewise linear value function

#### Optimal value function

The mode can turn to
$$
F_t^{\ast}(v_t(W_t),R_t,W_t)=\max_{x_t,y_t}C_t(R_t,W_t,x_t) +\gamma \sum_{r=1}^{B^R}v_t(r,W_t)y_{tr}
$$
which can equal to
$$
\max_{x_t,y_t}\  p_t(gen_t/\eta^g-{pump}_t \ast \eta^p)+\gamma \sum_{r=1}^{B^R}斜率_t y_{tr}
$$
where the breakpoints $R=1,\cdots,B^{R}$ and $斜率=v_t(W_t)=(v_t(1,W_t),\cdots,v_t(B^R,W_t))$


$$
\begin{align}
& 0 \leq gen_t  \leq Gen_t, \\ 
& 0 \leq pump_t \leq Pump_t, \\
& R_{t}=R_{t-1}+gen_t/\eta^g-{pump}_t \ast \eta^p \\
& \underline{R}\leq R_t \leq \bar{R} \\
<<<<<<< HEAD
& \sum_{r=1}^{B^R}y_{tr}\rho =f^{x}(R_t,x_t)=R_t? (这个将来可以sample\&用来catch将来的soc的？)
=======
& \sum_{r=1}^{B^R}y_{tr}\rho =f^{x}(R_t,x_t) =R_t^x? (这个将来可以sample\&用来catch将来的soc的？)
>>>>>>> b04a124642748cd0ee399bdabc0f8d8870f31c03
\end{align}
$$

notes：如果用了$\rho$，相当于每一步最长是$\rho$,$y_{tr} \leq 1$；如果不用$\rho$，相当于你的$y_{tr} \leq \rho$



## How to learn? SPAR-Storage Algorithm



#### Step 0: Initialization

1. Initialize $\bar{v}_t^0(W_t)$ for $t=0,\cdots, T-1$, and $W_t \in \mathcal{W}_t$ monotone decreasing
2. Set $R_{-1}^{x,n}=\bar{r}=k \rho$ for some $k \geq 0$ for all $n \geq 0$
3. Set $n = 1$

notes：这里的$\bar{v}_t^0(W_t)=(\bar{v}_t^0(1,W_t),\cdots, \bar{v}_t^0(B^R,W_t))$,（固定在$W_t$下），同时是monotone decreasing，文章中说可以设计成0，zhengmao自己选了一个decreasing



#### Step 1: Sample/ Observe the information sequence $W_0^n,\cdots W_t^n$

##### from step 2 to step 5, we do for $t=0,\cdots, T$

#### Step 2-4:前面的linear approximation的模型I/O

##### Step 2: Compute the pre-decision asset level : 

$$
R_t^n =R_{t-1}^{x,n} +\hat{R}_t(W_t^n)
$$

notes: 我们这里就直接update？相当于说借用上一个阶段的？还是

##### Step 3: Find the optimal solution $x_t^n$ 

$$
\max_{x_t \in \mathcal{X}_t(R_t,W_t^n)}C_t(R_t^n,W_t^n,x_t) +\gamma \bar{V}_t^{n-1}(f^x(R_t^n,x_t))
$$



notes:这里要小心，第二个部分是上一个scenario$n-1$的?？why？？？

##### Step 4: Compute the post-decision asset level


$$
R_t^{x,n}=f^x(R_t^n,x_t^n)
$$
这个就相当于你因着将来，而让现在需要storage多少（和SOC-shadow price curve得出来类似）(我们没有这这一步)

#### Step 5: update the slope:

##### If $t < T$:

##### Step 5-1: Observe $\hat{v}_{t+1}^n(R_t^{x,n})$ and $\hat{v}_{t+1}^n(R_t^{x,n}+\rho)$

notes:这里相当于你找出 关于$R-v$ curve 上的两个点，我们用来update斜率，很自然你选择的是两个比较近的点，那公式就用
$$
\begin{align}
\hat{v}_{t+1}^n(R) &=F_t^{\ast}(\bar{v}_{t+1}^{n-1}(W_{t+1}^n), R+\hat{R}_{t+1}(W_{t+1}^n),W_{t+1}^n)-F_t^{\ast}(\bar{v}_{t+1}^{n-1}(W_{t+1}^n), R-1+\hat{R}_{t+1}(W_{t+1}^n),W_{t+1}^n)\\
&=?
\end{align}
$$

##### Step 5-2: for $W_t\in\mathcal{W_t}$, and $R=1,\cdots,B^R$, update $z_t^n(R,W_t)$:

$$
z_t^n(R,W_t)= (1-\bar{\alpha}_t^n(R,W_t)) \bar{v}_t^{n-1}(R,W_t)+\bar{\alpha}_t^n(R,W_t)\hat{v}_{t+1}^n(R)
$$

notes:注意这里的scenario,一个是 $n-1$，另一个是 $n$.

##### Step 5-3: perform the projection operation $\bar{v}_t^n= \Pi_{\mathcal{C}}(z_t^n)$

$$
\Pi_{\mathcal{C}}(z_t^n)(R,W_t)=
\begin{cases}
z_t^n(R_t^{x,n},W_t^n), & if \  W_t= W_t^n , R< R_t^{x,n}, z_t^n(R, W_t)\leq z_t^n(R_t^{x,n},W_t^n) \\
z_t^n(R_t^{x,n}+\rho, W_t^n), & if \ W_t=W_t^n, R>R_t^{x,n}+\rho, z_t^n(R,W_t) \geq z_t^{n}(R_t^{x,n}+\rho, W_t^n) \\
z_t^n(R,W_t),& otherwise. \\
\end{cases}
$$

where 
$$
\bar{\alpha}_t^n(R,W_t)= \alpha_t^n 1_{W_t=W_t^n}(1_{R=R_t^{x,n}}+1_{R=R_t^{x,n}+\rho})
$$

#### Step 6: Increase $n$ by one and go to step 1









nonanticipity你的决定和将来的情况无关,相当于你种田，不管你将来的天气





Reference:

1.  Juliana Nascimento,Warren B Powell. (2013). An optimal approximate dynamic programming algorithm for the economic dispatch problem with grid-level storage, IEEE Transactions on Automatic Control, to appear print.
2. https://zhuanlan.zhihu.com/p/25319023