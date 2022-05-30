---
title: 'Machine Learning-risk minimization problem'
date: 2020-04-01
permalink: /posts/2020/04/risk minimization problem/
tags:
  - Learning Theory
  - Optimization
---

We discuss what is risk minimization problem

In machine learning area, there is a very classical and vital question, which is to describe the machine's performance.
It is the problem of risk minimization problem. (Vapnik(1992,1999\cite{Vapnik1992}\cite{Vapnik1999}))

Consider the following situation as the supervisor leaning problems. There are two spaces $X$ and $Y$. Let $X$ be the input space(feature space), and the $Y$ as the output space. Consider a random vector $x \in X \subset \mathbb{R}^d$ as input, and the corresponding output $y \in Y \subset \mathbb{R}^1$. Meanwhile, the join space $\Omega= X \times Y$ defined on the probability space $(\Omega,\mathcal{F}, \mathbb{P})$

Also, the output $m(x,u), u \in U$ is considered as difference types of learning machine.

In order to choose the machine's best performance, define the object function:
$$
L(X,u;y)\xlongequal{\bigtriangleup} l\left(m(X,u);y \right)
$$


where function $l$ is the loss function. The loss function depends on the question. For regression problem, usually people choose the quadratic function as the loss function. Different classification problem also has different loss function. The most usual loss function is 0-1 loss function.  %The Adaboost method in ensemble learning, we can choose exponential function(???)





Therefore, the expected of the loss function is called the risk functional
$$
    R(u)=\int L(X,u;y) dP(x,y)
$$
The best performance of the machine(over the different types of learning machine in $U$) can be described as the minimization problem of the function $R(u)$, i.e.
$$
    \min_{u \in U} R(u)=\min_{u\in U}E L(X,u;y)
$$


This is exactly the stochastic optimization problem when we consider the $Z=(X,y)$. If the machine learning problem is an unsupervised problem, we consider the model without the output $Y$

##### Linear Regression

For example,
consider basic Linear regression model  $m(X;u)=(A)^{\top}X+\alpha$ as the classical linear model, which $u$ is the parameter $u \xlongequal{\bigtriangleup}\left\{ (A,\alpha) \right\}$.

For the loss function $l$, since the $m(X;u) \in L_2$, we can choose the following loss function 
$$
    L(\cdot;y)=(y-\cdot)^2
$$


In this case we have $L(m(X;u);y)=(y-m(X;u))^2$


\textcolor{red}{change to the problem professor gave?}

##### SVM for Classification


The only different of the classification and the regression is the loss function, for example, the general loss function is 
$$
     L(\cdot;y)= \begin{cases}
     0
    & \text{if $ y=m(X;u) $}
    \\
     1
      & \text{if $ y \neq m(X;u) $}
    \end{cases}
$$


For the most famous classification method, SVM,
the hinge loss function 
$$
    L(m(X;u);y)=\max(0,1-y \cdot m(X;u))
$$



also we can use the huber risk function or risk measure for the robust binanry classification problem from (Dentcheva, Xiong, Vitt\cite{Dentcheva2019} 2019).

##### Ensemble Method for Classfication

adaboost, Bootstrap, Bayes optimal classifier

##### Deep Learning

Consider a two layer neural network with the ReLU activation function($m_1(\cdot)=\max(0,\cdot)$), the form of function $m$ is
$$
   m(X;u) \xlongequal{\bigtriangleup} \max\left(b^{\top} \max(AX+\alpha,0)+\beta,0 \right)
$$


The data $X \in \mathbb{R}^d$. In the first layer, the parameter $A\in \mathbb{R}^{k \times d}$ and $\alpha \in \mathbb{R}^d$. In the second layer, the parameter $b \in \mathbb{R}^k$ and $\beta \in \mathbb{R}$.

The loss function can be the square loss, the cross entropy function or the huber loss. All these loss functions leads to a nonconvex and nonsmooth structure (Pang(2018,2019 \cite{Pang2018}\cite{Pang2019}).

##### Unsupervise

kmeans





#### Bellman Error Problem in Reinforcement Learning

Consider a Markov Chain $\{X_1,X_2,\cdots\} \subset \mathcal{X}$ with an unknown transition operation $P$, a reward function $r: \mathcal{X} \rightarrow \mathbb{R}$, and a discount factor $\gamma$. We would like to evaluate the value function $V: \mathcal{X} \rightarrow \mathbb{R}$ by $V(x)=\mathbb{E}[\sum_{t=0}^{\infty}\gamma^{t}r(X_t)|X_0=x]$.

If $\mathcal{X}$ is finite space, the functions $r$ and $V$ may be viewed as vectors. The following policy equation is satisified:
$$
V=r+\gamma P V
$$
As $P$ is not known and $|\mathcal{X}|$ may be large, this system cannot be solved directly. To reduce the dimension of this problem, we employ a sketching matrix $S \in \mathbb{R}^{d \times |\mathcal{X}|}$ and a linear model forthe  value funciton $V(x) \approx \sum_{i=1}^k w_i \phi_i(x)$, where $\phi_1(\cdot),\cdots, \phi_k(\cdot)$ are given basis functions. Then we can formulate the minimization problem for least square policy evaluation:
$$
\min_{w \in \mathbb{R}^d}\|S(\Phi w-r-\gamma \mathbb{E}[\hat{P}]\Phi w)\|^2=\min_{w \in \mathbb{R}^d}\|S\left( (I-\gamma E[\hat{P}])\Phi w-\gamma \right)\|^2
$$
where $\Phi$ is the matrix with columns being the basis functions, and $\hat{P}$ is sample transition matrix. In this case, we may define the outer function $f$ as the squared norm, and the inner function $g$ as the linear mapping inside the norm. Neither of the functions has an easily available value or derivative, but their samples can be generated by simulation.}
As a result, solving the Bellman equation becomes a special case of the stochastic composition optimization problem:
$$
    \min_{x\in X} \|\mathbb{E}[A]x-\mathbb{E}[b]\|^2
$$
where A,B,b are random matrices and random vectors such that $\mathbb{E}[A]=I-\gamma P^{\pi}$ and $\mathbb{E}[b]=r^{\pi}$. It can be viewed as the composition of the sqaure norm function $f(\cdot)=f_v(\cdot)=\|\cdot\|$ and the expected linear function $g(x)=E[A]x-E[b]$



We consider the Bellman's equation assuming linear model.
However, we are not yet ready to introduce the dimension of optimizing over policies, so we are still simply trying to approximate the value of being in a state.

This presentation can be viewed as another method for handling infinite horizon models, while using a linear architecture to approximate the value function.

First recall that Bellman's equation(for a fixed policy) is written
$$
V^{\pi}(s)=c(s,A^{\pi}(s))+\gamma \sum_{s^{'}\in S}p(s^{'}|s,A^{\pi}(s))V^{\pi}(s^{'})
$$



In vector-matrix form, we let $V^{\pi}$ be a vector with element $V^{\pi}(s)$, we let $c^{\pi}$ be a vector with element $C(s,A^{\pi}(s))$, and we let $P^{\pi}$ be the one-step transition matrix with element $p(s^{'}|s,A^{\pi}(s))$ at row $s$, column $s^{'}$. By this notation, Bellman's equation becomes:
$$
V^{\pi}=c^{\pi}+\gamma P^{\pi}V^{\pi}
$$
Now assume that we replace $V^{\pi}$ with an approximation ${\bar{V}}^{\pi}=\Phi \theta$, where $\Phi$ is a $|\mathcal{S}|\times |\mathcal{S}|$ diagonal matrix where the stat probabilities $d_1^{\pi}, \cdots, d_{|S|}^{\pi}$ make up the diagonal. We to choose $\theta$ to minimize the weight sums of errors squared, where the error for the state $s$ is given by 
$$
\varepsilon^{n}(s)=\sum_{f}\theta_f\phi_f(s)-\left(c^{\pi}(s)+\gamma \sum_{s^{'}\in S}p^{\pi}(s^{'}|s, A^{\pi})\sum_f \theta_f^{n}\phi_f(s^{'}) \right)
$$
The first term on the right-hand side of this equation is the predicted value of being in each state given $\theta$, while the second term on the right-hand side is the "predicted" value computed using the one-period contribution plus the expected value of the future, which is computed using $\theta^{n}$. The expected sum of errors squared is then given by 

$$
\min_{\theta} \sum_{s \in S}d_s^{\pi}{\left(
    \sum_{f}\theta_f\phi_f(s)-\left(c^{\pi}(s)+\gamma \sum_{s^{'}\in S}p^{\pi}(s^{'}|s, A^{\pi})\sum_f \theta_f^{n}\phi_f(s^{'}) \right)
    \right)}^2
$$
In matrix form this can be written 
$$
   \min_{\theta}{\bigg(\phi\theta-(c^{\pi}+\gamma P^{\pi}\Phi \theta^{n})\bigg)}^{T}D^{\pi}\bigg(\phi\theta-(c^{\pi}+\gamma P^{\pi}\Phi \theta^{n})\bigg)
$$





Notice:


#### 1)Policy&Decision Rule&Control

Policy: a sequence of decision rule

Decision Rule: $\pi_{t}(X_t): \rightarrow U$: state control histories $\pi_{t}(x_1, \cdots, u_{t-1},x_t) \in U_t(x_6)$

Control: u： control



Reference:

1. p367 powell's book \cite{Powellbook2007}
2. p268 BartoSutton's reinforcement learning \cite{BartoSuttonlbook2018}

