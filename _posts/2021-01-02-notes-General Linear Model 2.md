---
title: 'notes-General Linear Model 2'
date: 2021-01-01
permalink: /posts/2021/01/notes-General Linear Model 2/
tags:
  - Models
  - Machine Learning
  - Optimization
---

Continue previous linear regression. Here we would like to introduce Logistic regression/

### Logistic Regression



























1. Linear discriminant analysis 线性判别分析 (LDA)是对费舍尔的线性鉴别方法的归纳，这种方法使用统计学，模式识别和机器学习方法，试图找到两类物体或事件的特征的一个线性组合，以能够特征化或区分它们。所得的组合可用来作为一个线性分类器，或者，更常见的是，为后续的分类做降维处理。
2. LDA与方差分析（ANOVA）和回归分析紧密相关，这两种分析方法也试图通过一些特征或测量值的线性组合来表示一个因变量。然而，方差分析使用类别自变量和连续数因变量，而判别分析连续自变量和类别因变量（即类标签）。逻辑回归和概率回归比方差分析更类似于LDA，因为他们也是用连续自变量来解释类别因变量的。LDA的基本假设是自变量是正态分布的，当这一假设无法满足时，在实际应用中更倾向于用上述的其他方法。





### Regularization——Ridge regression/Lasso regression



1. **model interpretability**：by removing irrelevant features- that is, by setting the corresponding coefficient estimates to zero-- we can obtain a model that is more easily interpreted. We will present some approaches for automatically performing **feature selection.**
2. **predictive performance**：especially when $p>n$ to control the variance.
3. Three methods:

**Subset selection:** we identify a subset of the $p$ predictors that we believe to be related to the response. We then fit a model using least squares on the reduced set of variables:(相当于你选一个度量单位，然后以此来选择)

1. 一个个去选，你到底要几个feature，基于smallest RSS/$C_p$/AIC/BIC/adjusted $R^2$
2. Overfitting&stepwise methods, 
3. Forward/backward stepwise selection
4. estimating test error: two approach: 
   1. smallest **RSS/$C_p$/AIC/BIC/adjusted $R^2$**
   2. Validation/cross varidation??

**Shrinkage:** we fit a model involving all $p$ predictors, but the estimated coefficients are shrunken towards zero relative to the least squares estimates. This shrinkage (regularization) has the effect of reducing variance and can also perform variable selection.(shrinkage，相当于渐进，直到消失)

1. Ridge regression:$\sum_{i=1}^n(y_i-\beta_0-\sum_{j=1}^p\beta_j x_{ij})^2+\lambda\sum_{j=1}^p \beta_j^2$
2. Ridge regression之前最好先stadardizing the predictors；因为substantially实质上，不同的scale会导致不同的coefficient
3. Why does ridge regression improve over least squares？
4. Lasso regression:$\sum_{i=1}^n(y_i-\beta_0-\sum_{j=1}^p\beta_j x_{ij})^2+\lambda\sum_{j=1}^p \vert  \vert \beta_j \vert \vert$
5. Lasso regression：overcome the disadvantage（包含所有的input/predictors 在最后的模型里面）；这个用的$l_1$ penalty
6. Lasso regression: yields sparse models, models that involve only a subset of the variables
7. Lasso regression: performs variable selction///select a good value of $\lambda$ for the lasso is critical///cross-validation is again the method of choice//MSE smallest
8. tuning parameter:对于一个sample，用cross validation

**Dimension reduction**: we project the $p$ predictors into $M$-dimensional subspace, where $M<p$. This is achieve by computing $M$ different linear combinations or projections, of the variables. Then these $M$ projections are used as predictors to fit a linear regression model by least squares.(把$p$压缩$M$)

1. PCA；2. transform； 3。partial least square



