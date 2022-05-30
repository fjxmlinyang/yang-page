## 2019/12/01

#### Uniform Normaized(UN) convergence of random variables

Let $(\Omega,\Sigma_{\Omega},P)$ be a probability space 

$X$ be a seperable metric space with distance function  $\rho(\cdot,\cdot)$

A sequence of random variables $\theta_n: \Omega \rightarrow X$, $n=1,2, \cdots,$ is UN convergent to a random variable $\theta: \Omega \rightarrow X$ , with rate(of a numerical sequence) $\frac{1}{v_n}$ and distribution $\Phi$, 

if there is a sequence of numbers $v_n \rightarrow \infin$, $0 \leq v_n \leq \infin$, and (left continuous) distribution function $\Psi: \mathbb{R}^1 \rightarrow [0,1]$ such that

$$
Pr \{v_n \rho(\theta_n, \theta)<t\} \geq \Psi(t), \forall t>0, n=1,2,\cdots
$$


## 2019/12/03

#### UN-Convergence

If in the previous definition, inequality $Pr \{v_n \rho(\theta_n, \theta)<t\} \geq \Psi(t), \forall t>0, n=1,2,\cdots$ is fulfilled uniformly in $t$, i.e., for any $\epsilon>0$ and all $n \geq n(\epsilon)$
$$
{\lim\inf}_{n \rightarrow \infin} Pr\{v_n\rho(\theta_n,\theta)<t\}\geq \Phi(t)-\epsilon, \ , \forall t \in \mathbb{R}^1
$$

take place, and then we say that $\{\theta_n\}$ is uniformly normalized convergent to $\theta$ and denote $\theta_n \longrightarrow^{UN} \theta$



####Some Notation

Let $\xi^n=\{\xi_1(\omega),\cdots, \xi_n(\omega)\}$ be a set of i.i.d. random variables with the same distribution as $\xi(\omega)$, and denote $$\begin{equation} aaaaa\end{equation}$$











## 2020/02/29 

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

#####1.2 Monte Carlo off policy evaluation 

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



### 2020/03/17

The best way to showcase your Data Science skills is with these 5 types of projects:

1. **Data Cleaning**
2. **Exploratory Data Analysis**
3. **Interactive Data Visualizations**
4. **Machine Learning**
5. **Communication**







#### 2020/04/01 Machine Learning-risk minimization problem

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



$\textcolor{red}{p367 powell \cite{Powellbook2007}!!!
p268 book  reinforcement learning \cite{BartoSuttonlbook2018}}$

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




#### Policy&Decision Rule&Control

Policy: a sequence of decision rule

Decision Rule: $\pi_{t}(X_t): \rightarrow U$: state control histories $\pi_{t}(x_1, \cdots, u_{t-1},x_t) \in U_t(x_6)$

Control: u： control







#### 2020/04/22

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
  



#### 2020/07/02: Topic: Cone

We discuss convex cone and optimal conditions here

####  (Reference :当我们谈凸锥时，我们在谈些什么？https://zhuanlan.zhihu.com/p/50156433)



We start talking about cone from a geometric propsect. Then we move to its relationship with optimaltiy condition.

There are three different cone today: Tangent cone, Linearized cone, Feasible cone.

Let us start from the definition of the tangent cone:



##### Tangent Cone

Let $C \subset \mathbb{R}^n$ be a closed nonempty set and let $\bar{x} \in C$. The vector $d \in \mathbb{R}^n$ is a $\textcolor{red}{tangent \  direction}$ to $C$ at $\bar{x}$ if there exists a sequence $\{x^{k}\}$ of points $x^{k} \in C$ converging to $\bar{x}$ and a sequence of positive numbers $\tau^{k} \rightarrow 0$ such that the vectors $d^{k}:=(x^k-x)/\tau$ converge to $d$.

The set of all tangent directions $d$ to $C$ at $\bar{x}$ is called tangent cone at $\bar{x}$ denoted by $T_C(\bar{x})$:
$$
T_C(\bar{x})=\{d \in \mathbb{R}^n: \exists \{x^k\}  \textit{ with } x^k \in C \textit{, and } \{\tau^k\} \rightarrow 0 \textit{ such that } d=\lim_{k \rightarrow \infty} \frac{x^k-\bar{x}}{\tau^{k}} \}
$$
这个定义告诉我们什么？假如有一个闭集，然后有一点x在这个闭集内，与此同时闭集内有一组逐渐收敛于x的序列。在点x附近（neighborhood）所有可以收敛的方向都是这个点的tangent direction。点x在该闭集上的tangent cone一般来说是从该点做切线，并包含闭集内部的向外延伸的锥。![Screen Shot 2020-07-02 at 3.26.00 PM](/Users/yanglin/Documents/GitHub/fjxmlinyang.github.io/_posts/Screen Shot 2020-07-02 at 3.26.00 PM.png)

##### Linearized cone

对于多面体，linearized cone 如下：
$$
C=\{x \in \mathbb{R}^n: a_i^{\top}x \leq b_i, i=1,\cdots,m \qquad, c_j^{\top}=f_j, j=1,\cdots,p\}
$$
and $\bar{x} \in C$. Define:
$$
I(\bar{x}):=\{i=1,\cdots,m: a_i^{\top}x=b_i\}=\textit{set of active inequalities at }\bar{x}
$$

$$
L_C(\bar{x}):=\{d \in \mathbb{R}^n: a_i^{\top}d \leq 0, i=I(\bar{x}) \qquad \textit{and} \qquad c_j^{\top}d=0, \ j=1,\cdots,p\}
$$



在这种情况下，linearized cone 恰好与tangent cone 相同。如果引申到非线性不等式，等式的情况，类似地，我们有如下定义：

Given the set
$$
C=\{x \in \mathbb{R}^n: g_i(x) \leq 0, i=1,\cdots,m, h_j(x)=0, j=1,\cdots,p\}
$$
The linearized cone($\textcolor{red}{linearized \  direction?}$) $L_C(\bar{x})$ at $\bar{x} \in C$ is 
$$
L_C(\bar{x})=\{d \in \mathbb{R}^n: \nabla g_i(\bar{x})^{\top}d \leq 0, i\in \mathcal{I}(\bar{x}), \nabla h_j(\bar{x})^{\top}d=0, j=1,\cdots, p\}
$$


##### Feasible direction

定义是这样的，

Definition：

Given a closed set $C \subset \mathbb{R}^n$ and a point $\bar{x} \in C$, a vector $d \in \mathbb{R}^n$ is called $\textcolor{red}{feasible \ direction}$ of $C$ at $\bar{x}$ if there exists a $\bar{\lambda}>0$ such that 
$$
\bar{x}+\lambda d \in C, \qquad \forall \lambda \in [0, \bar{\lambda}]
$$
The set of all feasible directions of $C$ at $x$ is denoted by $\mathcal{F}_C(x)$

1)直观来说就是在某个点出发”可以移动的方向“，就是沿着这个方向走的步长打鱼0，但仍然在这个set中。

2）如果定义是凸集，这两个结果一样，如果不是凸集就不一样了。对于非图集合，需要加一些附加条件才行![Screen Shot 2020-07-02 at 2.29.58 PM](/Users/yanglin/Documents/GitHub/fjxmlinyang.github.io/_posts/Screen Shot 2020-07-02 at 2.29.58 PM.png)

图2-1中，根据定义可以得到，tangent cone 和 feasible direction是一样的。而图2-2，feasible direction为空集，因为向任意一个方向移动严格大于零的一步，都有可能不在定义域内。所以我们初步得到，对于凸集，两者相等；对于非凸集合，需要一些附加条件才行。



#### 那么这三种cone又与optimal condition 有什么联系呢？



最优解的定义：此时从最优解 $x$ 点出出发，不存在feasible descent direction，可以使得目标函数值下降。这句话，其实是优化问题的最优解判断条件的本质。
$$
\min(C,f): \qquad \min_{x{\in}C} f(x)
$$

#### feasible direction——》tangent cone

##### descent direction：

假设向量$P$ 是目标函数在当前点$x$ 处的 descent direction，那么沿着$P$移动，目标函数值下降。

##### feasible direction：

（注意这个和你所处的点还有关系，当提到方向，就意味着说到一个起点）: 在有约束的优化问题中，不仅要考虑descent direction，更是要考虑feasible set里的移动方向，两者结合在一起，就是feasible descent direction。

###### 但是出现一些问题&解决办法：

1）feasible direction 的约束作用在一些情况下略显严格，会导致空集得到产生（例如非凸集合），因此我们需要适当放宽这个约束

2）因此我们考到用tangent cone来替代：2-1）在约束条件是凸集的情况下，tangent cone 等同于feasible direction；2-2）在非凸情况下，tangent cone 避免空集的产生。

3）下面的式子告诉我们最优解处的tangent cone必须属于由梯度决定的half space中，有次我们可以得到再下面的式子

Given $Min(C,f)$, let $f$ be $C^{1}$ on the closed set $C$. If $\bar{x} \in C$ is local minimum, then
$$
\nabla f(\bar{x})^{T} d \geq 0 \qquad d \in T_C(\bar{x})
$$
If, in addition, $C$ os convex, then this reduces to 
$$
\nabla f(\bar{x})^{T} (y-\bar{x}) \geq 0 \qquad \forall y \in C
$$




#### tangent cone——》Linearized cone

1）Linearized cone相比于其他两个cone更容易通过计算求得

2）Linearized cone和tangent cone只有在集合是多面体的时候才完全相等

3）If是非线性约束或者等式约束，就需要引入其他条件，从而将tangent cone换成linearized cone

所以问题来了，哪些条件满足时可以证明两者相等，下面是几个条件

##### Constraint qualification：

1) [linear constraints]: all the $g_i$ and $h_j$ are affine.

2) [linear independence constraint qualification(LICQ)]: the set of active constraint gradients
$$
\{\nabla g_i(\bar{x}), i \in \mathcal{I}(\bar{x}), \nabla h_j(\bar{x}), j=1,\cdots,p\}
$$
is linearly independent;

3) [Mangasarian-Fromovitz CQ(MFCQ)]:

​	3-1) the set $\{\nabla h_j(\bar{x}) , j=1\cdots,p\}$  is linear independent;

​	3-2) there exists a $d \neq 0$ such taht $\nabla g_i(\bar{x})^{T}d \leq 0$,  $i \in \mathcal{I}(\bar{x})$ and $\nabla h_j(\bar{x})^{T}d=0$, $j=1,\cdots,p$ ;

4)[Slater's CQ]:

​	4-1)All $g_i$ are convex and ahh $h_j$ are affine

​	4-2) There exists a feasible $\hat{x}$ such that $g_i(\hat{x})<0$ for all $i=1,\cdots,n$

满足以上条件之一，我们就可以说：
$$
T_C(\bar{x})=L_C(\bar{x})
$$



Reference:

1）知乎回答https://zhuanlan.zhihu.com/p/50156433





# What is mathematical programming?

Mathematical programming is the study or use of the mathematical program.

It includes any of all of the following:

1. **Theorems about the form** of a solution, including whether one exists;
2. **Algorithms** to seek a solution or ascertain that none exists;
3. **Formulation of problems into mathematical programs**, including understanding the quality of one formulation in comparison with another;
4. **Analysis of results**, including debugging situations, such as infeasible or anomalous values;
5. **Theorems about the model structure,** including properties pertaining to feasibility, redundancy and/or implied relations (such theorems could be to support analysis of results or design of algorithms);
6. **Theorems about approximation** arising from imperfections of model forms, levels of aggregation, computational error, and other deviations;
7. **Developments** in connection with other disciplines, such as a computing environment.

#Under my view:

**Theorem and Model**:

1)***theorem about the form***(==subfield==, and in subfield such as convex set& convex function......==constrained==)

2)***theorem about the model structure***(==optimal condition==(neccesarry and sufficient)/ ==sensitivity analysis==)

**Approximation and Algorithm**:

1)***Theorem about approximation***(==Dual theorem==&other approach,like SAA)

2)***Algorithm***(==convergence==&==stop time==&==speed==)

**Application**:

1)***Formulation of problems into mathematical programming***(==formulation==& ==explanation==& ==comparison==)

2)***Analysis of the result***(==the meaning of lagrange multiplier?== Like  shadow price)



Reference:

1)https://glossary.informs.org/ver2/mpgwiki/index.php?title=Main_Page







##2020/08/25

What is Big Data？

• Poseaproblem
 • Collectdata
 • Pre-processandcleandata
 • Formulateamathematicalmodel • Findasolution
 • Evaluateandinterprettheresults



What is Optimization

Find the optimal solution that minimize/maximize an objective function subject to constraints









## 2020/09/17

统计的思想（以parameter estimation为基准）

 Population distribution $F(x,\theta)$, we consider use  $\theta$ to be its parameter

Sample Distribution $F(X_1,X_2,\cdots,X_n;\theta)$, we consider use statistics(estimator)$\hat{\theta}=\hat{\theta}(X_1,X_2,\cdots,X_n)$ to approach $\theta$

### How to find the estimator?

1) method: moment estimation: 直接基于 $\hat{\theta}=\hat{\theta}(X_1,X_2,\cdots,X_n)=(\hat{\theta}_1,\cdots,\hat{\theta}_n)$ 来逼近，找每个moment相当直接逼近distribution的重要parameter

2)method：likelihood function：用distribution本身来逼近， $L(\hat{\theta};x_1,\cdots,x_n)=\max_{\theta}L(\theta;x_1,\cdots,x_n)=\max_{\theta} \Pi_{i=1}^{n}p(x_i;\theta)$  will approximate

(相当于如果每个$X_i$最逼近$X$,那么一定到的是最大值，也就是$L(\hat{\theta};x_1,\cdots,x_n) \rightarrow L(\theta;x,\cdots,x)$)

### How good it is?

1)基于这个parameter，$\hat{\theta}=\hat{\theta}(X_1,X_2,\cdots,X_n)$ 相当于也是有个分布，

2） 如果这个分布 $E(\hat{\theta})=\theta$, then it is called unbiassed;

 如果这个分布 $Var(\hat{\theta})\rightarrow 0$, then it is called effectiveness;

3) 如果这个分布 从依概率收敛；则是consistency

$\hat{\theta}=\hat{\theta}(X_1,X_2,\cdots,X_n) \rightarrow \theta$



4) 如果我可以把这个问题换到数轴上，同时减小performance误差，找上下两个interval，或者说一个带出来的上下bound，那我就这就是confidence interval, 

$P(\hat{\theta}(X_1,X_2,\cdots,X_n)\leq   \theta    \leq \hat{\theta}(X_1,X_2,\cdots,X_n)) \geq 1-\alpha$

i.e. The CI is a random interval with probability – confidence level $1-\alpha$ quantifying the chance

of capturing the true population parameter.



what is p-value &what is 











