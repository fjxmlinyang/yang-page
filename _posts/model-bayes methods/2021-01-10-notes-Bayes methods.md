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
R(c_i|\textbf{x})=\sum_{j=1}^N \lambda_{ij}P(c_j|\textbf{x})
$$
我们的任务是寻找一个判定准则 $h: \mathcal{X}\rightarrow \mathcal{Y}$ 以最小化总体风险
$$
R(h)=E_x[R(h(x))|x]
$$
2-2)显然，对每个样本$\textbf{x}$,若 $h$能最小化条件风险 $R(h(x)|x)$,则总体风险 $R(h)$也将被最小化，这就产生了贝叶斯判定准则（bayes decision rule）：为最小化总体风险，只需要在每个样本上选择哪个能使得**条件风险$R(c|\textbf{x})$**最小化的标记，即
$$
h^{\ast}(x)=\arg \min_{c \in \mathcal{Y}}R(c|\textbf{x})
$$
此时 $h^{\ast}$称为贝叶斯最优分类器（bayes optimal classifier）；与之对应的总体风险$R(h^{\ast})$称为贝叶斯风险（bayes risk）；$1-R(h^{\ast})$反应了分类器所能达到的最好性能，则通过机器学习所能产生的模型精度的理论上限。

3）这里就是naive bayes的情况：（bayes说白了就是你找损失最小化的方式，和之前的mean square error对应？）

$P(c|\textbf{x})=\frac{P(c)P(\textbf{x}|c)}{P(\textbf{x})}=\frac{P(c)}{P(\textbf{x})}\Pi_{i=1}^dP(x_i|c)$

$i$是$\{x_i\}$的个数

$h_{nb}(\textbf{x})=\arg \max_{c \in Y}P(c)\Pi_{i=1}^dP(x_i|c)$



### naive bayes classifier

condition: attribute conditional independence assumption:every feature independently influences the outcome



We have 

$$P(c|x)=\frac{P(c)P(x|c)}{P(x)}=\frac{P(c)}{P(x)}\Pi_{i=1}^{d}P(x_i|c)$$

where $d$ is the number of the features, and $x_i$ is the $i$th outcome in feature $i$.

The principle of bayes classification:

$$h_{nb}(x)=\arg \max_{c \in \mathcal{Y}}P(c)\Pi_{i=1}^d P(x_i|c)$$

显然，naive bayes classification的训练过程就是基于训练集$D$ 来估计 先验概率 $P(c)$, 并为每个属性估计条件概率 $P(x_i|c)$

1. 令$D_c$ 表示训练集 $D$中第$c$类样本组成的集合，若有充足的独立同分布样本，则可$P(c)=\frac{|D_c|}{D}$

2. 对离散属性，另$D_{c,x_i}$ 表示 $D_c$中的第$i$个属性上取值为$x_i$的样本组成集合，则$P(x_i|c)=\frac{|D_{c,x_i}|}{|D_c|}$

   对连续型，可以考虑密度函数 $p(x_i|c)=\frac{1}{\sqrt{2\pi}\sigma_{c,i}}\exp(-\frac{(x_i-\mu_{c,i})^2}{2\sigma_{c,i}^2})$ 







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





