---
title: 'notes-General Linear Regression'
date: 2021-01-01
permalink: /posts/2021/01/notes-General Linear Regression/
tags:
  - Models
  - Machine Learning
  - Optimization
---



Here are the notes for general linear regression. 

We shall also put the notes in goodnotes here

# General Linear Regression



### simple linear regression

1.what is regression for?

关系，自身的，和prediction的；

2. RSS（residual sum of sqaures）

3. assessing the accuracy of the coefficient estimation： $\hat{\beta}_1+/-2SE(\hat{\beta}_1)$ (confidence interval)

4. hypothesis test: null hypothesis $\hat{\beta}_1=0$

   $t=\frac{\hat{\beta}_1-0}{SE(\hat{\beta}_1)}$

$t$ distribution with $n-2$ degrees of freedom assuming $\beta_1=0$

$p$-value

5. assessing the overall accuracy

 $R^2=\frac{TSS-RRR}{TSS}$, where  $TSS$ is total sum of squares, $RSS$ is the residual sum of squares

 当是simple regression时，他相同于correlation

### multiple linear regression

1. interpreint regression coefficients:希望input 时uncorrelated；correlation 会影响；可以单独和output比较

2. RSS来判定好坏

3. **Is at least one of the predictors $X_1,\cdots, X_p$ useful in predictiing the response?** $F$ Statistic: $F=\frac{(TSS-RSS)/p}{RSS/(n-p-1)}$

4. **Do all the predictors help to explain $Y$, or is only a subset of the predictors useful?**

   ​			(不可能经过所有的input；所以基于最小化$RSS$选择一个$X_i$，然后你基于最小化$RSS$选择第二个$X_j$，直到选出来的$p$-value合格)（或者你可以采用全部放进去，基于$p$-value,然后一个个删掉）

5. **How well does the model fit the data?**

​	systematic criteria for choosing an 'optimal' member in the path of models produced by forward or backward stewise selection;

其他度量方式 Mallow's C_p, Akaike informantion criterion(AIC), Bayesian information criterion(BIC), adjusted $R^2$, Cross-validation(CV)

6. **Given a set of predictor values, what response value should we predict, and how accurate is our prediction**

7. 小心qualitative data；可以换成binary $x_1=0\&1$ 在不同情况下，当然还可以有 $x_2$
8. Removing the additive assumption: **interactions and nonlinearity**

Interaction:市场造成的相互的影响，比如说你增加$x_1$会影响$x_2$;这时候刚增加一个 $x_1x_2$项

hierarchy：hierarchy principle：if we include an interaction in a model, we should also include the main effects, even if the $p$-value associated with their coefficients are not significant.

9. outliers&non-constant variance of error terms& high leverage points& collinearity section3.3



### Logistic Regression

1. 
2. Linear discriminant analysis 线性判别分析 (LDA)是对费舍尔的线性鉴别方法的归纳，这种方法使用统计学，模式识别和机器学习方法，试图找到两类物体或事件的特征的一个线性组合，以能够特征化或区分它们。所得的组合可用来作为一个线性分类器，或者，更常见的是，为后续的分类做降维处理。
3. LDA与方差分析（ANOVA）和回归分析紧密相关，这两种分析方法也试图通过一些特征或测量值的线性组合来表示一个因变量。然而，方差分析使用类别自变量和连续数因变量，而判别分析连续自变量和类别因变量（即类标签）。逻辑回归和概率回归比方差分析更类似于LDA，因为他们也是用连续自变量来解释类别因变量的。LDA的基本假设是自变量是正态分布的，当这一假设无法满足时，在实际应用中更倾向于用上述的其他方法。



### Ridge regression/Lasso regression



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
4. Lasso regression:$\sum_{i=1}^n(y_i-\beta_0-\sum_{j=1}^p\beta_j x_{ij})^2+\lambda\sum_{j=1}^p \|\beta_j\|$
5. Lasso regression：overcome the disadvantage（包含所有的input/predictors 在最后的模型里面）；这个用的$l_1$ penalty
6. Lasso regression: yields sparse models, models that involve only a subset of the variables
7. Lasso regression: performs variable selction///select a good value of $\lambda$ for the lasso is critical///cross-validation is again the method of choice//MSE smallest
8. tuning parameter:对于一个sample，用cross validation

**Dimension reduction**: we project the $p$ predictors into $M$-dimensional subspace, where $M<p$. This is achieve by computing $M$ different linear combinations or projections, of the variables. Then these $M$ projections are used as predictors to fit a linear regression model by least squares.(把$p$压缩$M$)

1. PCA；2. transform； 3。partial least square





### Gradient descent简单的解释

Gradient descent is a commonly used optimization technique for other models as well, like neural networks, which we'll explore later in this track. Here's an overview of the gradient descent algorithm for a single parameter linear regression model:

- select initial values for the parameter: $a_1$
- repeat until convergence (usually implemented with a max number of iterations):
  - calculate the error (MSE) of model that uses current parameter value: $MSE(a_1)=\frac{1}{n}\sum_{i=1}^n({\hat{y}}^{(i)}-y^{(i)})^2$
  - calculate the derivative of the error (MSE) at the current parameter value: $\frac{d}{da_1}MSE(a_1)$
  - update the parameter value by subtracting the derivative times a constant ($\alpha$, called the learning rate): $a_1=a_1-\alpha \frac{d}{da_1}MSE(a_1)$













# SVM

1.见自己的文章，

$E((w^Tx+b,0)_{\max})$?

2. Maximum margin classifier

   $\max \frac{1}{\|w\|},\qquad s.t. y_i(w^Tx_i+b)\geq 0,\qquad i=1,\cdots,n$

3. $\min \frac{1}{2}\|w\|^2,\qquad s.t. y_i(w^Tx_i+b)\geq 1,\qquad i=1,\cdots,n$

   Dual  $\mathcal{L}(w,b,a)=\frac{1}{2}\|w\|^2-\sum_{i=1}^n \alpha_i(y_i(w^Tx_i+b)-1)$ 变形后

   $\mathcal{L}(w,b,a)=\sum_{i=1}^n\alpha_i-\frac{1}{2}\alpha_i\alpha_jy_i y_jx_i^T x_j$ And $w=\sum_{i=1}^n\alpha_iy_ix_i$

   dual problem 

4. 换成核估计

   $f(x)=\sum_{i=1}^Nw_i\phi_i(x)+b$ 转换成为 $f(x)=\sum_{i=1}^l\alpha_i y_i \langle \phi(x_i), \phi(x) \rangle+b$

$\alpha$ 可以由dual 来求

$\max_{\alpha}\sum_{i=1}^n \alpha_i-\frac{1}{2}\sum_{i,j=1}^n\alpha_i\alpha_jy_iy_j\langle \phi(x_i)\phi(x_j)\rangle \qquad s.t. \alpha_i \geq 0,i=1,\cdots, n; \sum_{i=1}^n\alpha_iy_i=0$ 

$\max_{\alpha}\sum_{i=1}^n \alpha_i-\frac{1}{2}\sum_{i,j=1}^n\alpha_i\alpha_jy_iy_jx_ix_j \qquad s.t. \alpha_i \geq 0,i=1,\cdots, n; \sum_{i=1}^n\alpha_iy_i=0$ 





