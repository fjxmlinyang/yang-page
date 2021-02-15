---
title: 'Statistcs&Decision Theory'
date: 2012-08-14
permalink: /posts/2020/09/statistcs&decision theory/
tags:
  - concepts
  - statistics
---

Here we talk some basic concepts in statistics



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



next question: based on the previous description, what is p-value &what is ?



what is difference between Z test and T test?(with one sample or two sample)?

