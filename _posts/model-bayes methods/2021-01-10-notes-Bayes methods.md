---
title: 'notes-Bayes methods'
date: 2021-01-10
permalink: /posts/2021/01/Bayes methods/
tags:
  - Models
  - Machine Learning
  - Optimization
---

We are talking about Bayes models here.



# Bayes Method







### EM（Expectation-Maximization）&MLE

1）EM 算法本来用来找MLE；相当于你多加点，调整parameter

这个写的好 https://xxwywzy.github.io/2019/02/12/cs229-9/

2）**em算法，第一步，选择归于哪个类（比如二项分布的哪个类型）；第二步调整中心**

### naive bayes

1）MLE

2)  是bayes决策理论的框架下的attribute condtional independence assumption

2-1)假设有$N$种可能的类别标记，即$Y=\{c_1,\cdots,c_N\}$, $\lambda_{ij}$是将一个真实的标记为$c_j$的样本误分类为$c_i$所产生的损失。基于后验概率$P(c_i|\textbf{x})$ 可获得将样本 $\textbf{x}$分类为 $c_i$所产生的期望损失（expected loss），即在样本$\textbf{x}$ 上的条件风险（conditional risk）
$$
R(c_i\vert \textbf{x})=\sum_{j=1}^N \lambda_{ij}P(c_j\vert \textbf{x})
$$
我们的任务是寻找一个判定准则 $h: \mathcal{X}\rightarrow \mathcal{Y}$ 以最小化总体风险
$$
R(h)=E_x[R(h(x))\vert x]
$$
2-2)显然，对每个样本$\textbf{x}$,若 $h$能最小化条件风险 $R(h(x)\vert x) $,则总体风险 $R(h)$也将被最小化，这就产生了贝叶斯判定准则（bayes decision rule）：为最小化总体风险，只需要在每个样本上选择哪个能使得**条件风险$R(c\vert\textbf{x})$**最小化的标记，即
$$
h^{\ast}(x)=\arg \min_{c \in \mathcal{Y}}R(c\ \vert \textbf{x})
$$
此时 $h^{\ast}$称为贝叶斯最优分类器（bayes optimal classifier）；与之对应的总体风险$R(h^{\ast})$称为贝叶斯风险（bayes risk）；$1-R(h^{\ast})$反应了分类器所能达到的最好性能，则通过机器学习所能产生的模型精度的理论上限。

3）这里就是naive bayes的情况：（bayes说白了就是你找损失最小化的方式，和之前的mean square error对应？）

$P(c\vert \textbf{x})=\frac{P(c)P(\textbf{x}\vert c)}{P(\textbf{x})}=\frac{P(c)}{P(\textbf{x})}\Pi_{i=1}^dP(x_i\vert c)$

$i$是$\{x_i\}$的个数

$h_{nb}(\textbf{x})=\arg \max_{c \in Y}P(c)\Pi_{i=1}^dP(x_i\vert c)$



### naive bayes classifier

condition: attribute conditional independence assumption:every feature independently influences the outcome



We have 

$$P(c\vert x)=\frac{P(c)P(x\vert c)}{P(x)}=\frac{P(c)}{P(x)}\Pi_{i=1}^{d}P(x_i\vert c)$$

where $d$ is the number of the features, and $x_i$ is the $i$th outcome in feature $i$.

The principle of bayes classification:

$$h_{nb}(x)=\arg \max_{c \in \mathcal{Y}}P(c)\Pi_{i=1}^d P(x_i\vert c)$$

显然，naive bayes classification的训练过程就是基于训练集$D$ 来估计 先验概率 $P(c)$, 并为每个属性估计条件概率 $P(x_i \vert c)$

1. 令$D_c$ 表示训练集 $D$中第$c$类样本组成的集合，若有充足的独立同分布样本，则可$P(c)=\frac{\left|D_c\right|}{D}$

2. 对离散属性，另$D_{c,x_i}$ 表示 $D_c$中的第$i$个属性上取值为$x_i$的样本组成集合，则$P(x_i\vert c)=\frac{\vert D_{c,x_i}\vert }{\vert D_c\vert }$

   对连续型，可以考虑密度函数 $p(x_i\bigm|c)=\frac{1}{\sqrt{2\pi}\sigma_{c,i}}\exp(-\frac{(x_i-\mu_{c,i})^2}{2\sigma_{c,i}^2})$ 





#### 一个关于条件概率的例子

1) 如何理解调节因子



#### 实际例子：fraud activity detection

社交网络公司的判断异常用户，对于用户的行为进行一个detection。

(比如说不看广告的账号,很正常这个对整个运营是干扰。ex如果你要做推广。。)

Prior information : $C_0=0.6$ , $C_1=0.1$ （两种fraud，僵尸号，水军）

Features:(blogs量，朋友量，头像)

$F_1$:

$F_2$:

$F_3$:



1)

What we shall use? 条件概率$P(C\vert A)=\frac{P(C)P(A \vert C)}{P(A)}$

if we know F_1=small, F_2=medium, F_3=real ,can we tell which type of fraud?

we need to find $P(C\vert  F_1, \cdots, F_n)$

$P(C\vert  F_1, \cdots, F_n)=\frac{P(C)P(F_1,\cdots,F_n \vert C)}{P(F_1,\cdots,F_n)}$

 where $P(F_1,\cdots,F_n\bigm|C)=P(F_1\bigm|C)P(F_2\bigm|C,F_1)\cdots P(F_n\bigm|C, F_1,\cdots,F_{n-1})$

Then under the assumption of conditional independent

we have the $P(F_1,\cdots, F_n \vert C)=P(F_1\vert C)P(F_2\vert C)\cdots F(F_n\vert C)$

2) how to calculate? 

Put in.

Notes：

1）我们只关心相对之间的概率，而不是绝对的概率

2）所以选择一个threshold的时候要小心，否则可能哪一类都不属于









# Basic Probability

#### randomness?

1. the probabiltiy distribution ss a description of a random phenomeon

#### random variable?

1. r.v.
2. Parameter  vs random variable:
   1. the paraemters are numbers which **helps** uniquely define the proability distribution or model
   2. also, it can be determined by the historical data after the parameter estimation methods

#### type of random variable?

1. Discrete：bernoulli？et.c

   Bernoullli？$P(Y=y)=p^{y}(1-p)^{1-y}$, $0<p<1$, $y=0,1$

2. Continuous： normal， weibull？







#### cdf/pdf

pdf是cdf的求导，cdf是pdf的积分，这个相当于是密度和面积（体积）的概念。





#### CLT/SLLN

从无到有













#### bayes： adjusted coeffiency

解释方式1

​	 $P(B \vert A) = P(B) \frac{P(A|B)}{P(A)}$

​	新信息出现后B的概率=B原先的概率*信息的调整（$>1, \ , =1 \ , <1$）

​	$\frac{P(A|B)}{P(A)}=1$说明是独立

​	$\frac{P(A|B)}{P(A)}>1$

​	$\frac{P(A|B)}{P(A)}<1$

 解释方式2

​	 $P(B \vert A) = P(A\vert B) \frac{P(B)}{P(A)}$

​	对概率的偏见，





























reference:

1) code part: link Dataquest





### Bayesian Network





Reference：

1）西瓜书

2)https://sylvanassun.github.io/2017/12/20/2017-12-20-naive_bayes/

3)https://github.com/lixianmin/cloud/blob/master/writer/R/bayes.md

4) https://www.zhihu.com/question/19725590/answer/32177811#





**不要当作是概率，当作是另一个事情对这个事情的提升**

​	作者：知乎用户
链接：https://www.zhihu.com/question/19725590/answer/32177811
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

[futher reading](./2021-01-10-notes-Bayes methods-supplement1.md)





