---
title: 'notes-General Linear Model 2'
date: 2021-01-01
permalink: /posts/2021/01/notes-General Linear Model 2/
tags:
  - Models
  - Machine Learning
  - Optimization
---

Continue previous linear regression. Here we would like to introduce Logistic regression.

Review:

1) what is linear regression?

2) how to get loss function? two method? Least Square（残差满足正态上的最大似然估计）/MLE?



Quote: when you are learning a new model, it is not for model, but it is for the idea behind

notes：

1. 任何机器学习都是为了找 $X$ 和 $Y$ 的关系
2. logistic regression是不能用least square 推导(non-linear的least square是可以的)

# Logistic Regression

## 1. background

当我们有categorical的variable的时候，比如说肿瘤的问题，得还是不得;或者是信用卡是否批下来贷款



<img src="./Screen Shot 2021-03-12 at 11.06.34 AM.png" alt="Screen Shot 2021-03-12 at 11.06.34 AM" style="zoom:25%;" />

这类问题我们怎么处理？

## 2.Model

### 2.1 logistic function

#### 2.1.1 图形角度理解（和linear regression比较）



<img src="./Screen Shot 2021-03-12 at 11.12.02 AM.png" alt="Screen Shot 2021-03-12 at 11.12.02 AM" style="zoom:25%;" />

所以我们需要一个新的一个函数来逼近，比如上面的图（注意这个图形是关于$x$ 和在$y$ 下的概率$p$的情况),

所以这里有个问题，$x \in (0, \infty)$,   但是$y= \pm 1$, 所以怎么联系 $x \rightarrow y$呢？（从连续映射到离散）

**采用这个方法**：$x \rightarrow p(x) \rightarrow y$, i.e.

1. 首先用$x$ 去拟合 概率$p$，
2. 然后用$p$再去拟合 $y$（采用threshold）



很自然我们要选用一个好的$p$去模拟，那在这里我们用sigmoid function中的logistic function（其实还有其他的平滑曲线：https://en.wikipedia.org/wiki/Sigmoid_function）
$$
p(x)=\frac{e^{f(x)}}{1+e^{f(x)}}=\frac{1}{1+e^{-f(x)}}
$$
Where $f(x)$ can be $x$ or other function(其他扭动方式) $ax +b$

#### 2.1.2 值域的理解（和linear regression比较）

1. Y是得病不得病，$p(x)$是相当于肿瘤的指数，$X$ 是关于肿瘤的input（size，位置，etc）
2. 很自然想到伯努利分布$p(Y=y_i)=p^{y_i}(1-p)^{1-y_i},\ 0<p<1,\ y_i=0,1$




$$
log\frac{p(X)}{1-p(X)} = \beta_0+\beta_1X
$$

3. 如何推导， 用博弈论中的$odds=\frac{p}{1-p}$,值域 $[0,\infty)$, 然后 $log(odds)=log\frac{p}{1-p}$,值域就到了$[0,\infty]$
   $$
   \begin{align}
   log\frac{p}{1-p}=f(x)=\beta_0+\beta_1x \\
   p=\frac{1}{1+e^{\beta_0+\beta_1x}}
   \end{align}
   $$




### 2.2 How is the performance of this model?(估计回归系数)



#### 2.2.1 Using MLE to get the loss function（面试必考题）

Step 1: Choose model

In (binary) Logistic Regression, $Y$ follows Bernoulli distribution**($y$和$p$的bernoulli关系)** 
$$
P(Y=y_i\vert X) = p^{y_i}(1-p)^{1-y_i},  \ 0<p<1,\  y_i=0,1
$$
**$p$ 和 $x$ 是logic function的关系**
$$
p=h_{\beta}(x_i)=\frac{1}{1+e^{-\beta_0+\beta_1x}}
$$
Step 2: Calculate loss function by MLE:
$$
L(\hat\beta_0,\hat\beta_1)=P(Y_1,\cdots,Y_n \vert X_1,\cdots,X_n)
$$
Since $(x_1,y_1),\cdots(x_n,y_n)$ are linear independent.(obersvation is independent & $x_2$和$y_1$没有关系)

​	a.  $P(Y_1, \cdots, Y_n\vert X_1,\cdots, X_n)=\Pi_{i=1}^{n}P(Y_i\vert X_1,\cdots,X_n)$

​	b. $P(Y_i \vert X_1,\cdots,X_n)=P(Y_i \vert X_i)$

Therefore 
$$
L(\hat\beta_0,\hat \beta_1)=\Pi_{i=1}^{n}P(Y_i \vert X_i)= \Pi_{i=0}^n p^{y_i}(1-p)^{1-y_i}=\Pi_{i=0}^n {h_{\beta}(x_i)}^{y_i}(1-h_{\beta}(x_i))^{1-y_i}
$$
Which is 
$$
\log L(\hat\beta_0,\hat \beta_1)= \sum_{i=1}^{n}[y_i \log(h_{\beta}(x_i)+(1-y_i)\log(1-h_{\beta}(x_i))]
$$
Based on MLE, it means we are finding the $\beta$, which is the solution for the following optimziation problem
$$
\begin{align}
\arg \max_{\beta}\log L(\hat\beta_0,\hat \beta_1)= \sum_{i=1}^{n}[y_i \log(h_{\beta}(x_i)+(1-y_i)\log(1-h_{\beta}(x_i))] \\
\arg \min_{\beta}\log L(\hat\beta_0,\hat \beta_1)= \sum_{i=1}^{n}[-y_i \log(h_{\beta}(x_i)-(1-y_i)\log(1-h_{\beta}(x_i))] \\
\end{align}
$$



#### 2.2.2 linear regression VS logistics regression

当自变量$x$ 给定时，我们对$y$ 的distribution的假设**（这块需要rethink）**

1. Linear regression的assumption 是： $p(y \vert x)$是$mean=f(x)=ax+b$,$variance= \sigma^2$(一个与$x$无关的值)的normal distribution
2. logistic regression的assumption 是：$y$服从 $p=\frac{1}{1+e^{-ax+b}}$ Bernoulli distribution



#### 2.2.3. 机器学习模型的‘能力范围’：

1. Step1： ‘**Given $X$，得$Y$的distribution**’——这是机器学习模型利用数据可以解决的问题
2. Step2:   ‘**根据$Y$的distribution，得到$Y$的取值**’——判断怎么用是你的问题



example：

1）predict model： y=2x+5000,（从新司机预测老司机的值）， 如果一个新司机拿x=1000，得出y是7000，但是不代表说这个y就一定是7000，因为你得出来的是y的Gaussian分布的mean=7000, 这表示的是这个老司机最有可能性的是7000，但是不一定是7000

2）所以你通常用均值(就是你期待的，也就是expected)**来代表**模型预测值

Example2:

1）比如说你预测下雨否，狗叫不叫预测下雨，其实也是两步，比如说狗叫5声下雨的概率是$0.6$，所以明天下不下雨？

2）这个问题到会与不会是因为你多加了一步，threshold定了个$0.5$



Example3:

1)一个网络公司，想知道提高病毒判断率高了——thereshold的问题，这个玩意数据有时候没法告诉你，所以可能需要有trade off



#### 2.2.4. Logistic regresion的拓展

$$
p(x)=h_{\beta}(x_i)=\frac{1}{1+e^{g(x)}}, \ \textit{where }\ g(x)=\beta_0+\beta_{1,1}x_1+\beta_{1,2}x_1^2+\beta_2x_2
$$

里面的polinomial可以是多个未知数的多项式

但是依然你会面对一个（overfit/underfit）的问题：

1. 模型复杂度对分类效果的影响

2. 模型复杂提到，可以更好的对应training data，但是对testing data不一定好
3. 过度拟合就是overfitting，但是过少可能就underfitting



notes：

1. **（threshold相当于调整你图上的hyperplane的后面那个constant $\log \frac{p(X)}{1-p(X)}=C=\beta_0+\cdots\beta_1X_1$）<img src="/Users/yanglin/Documents/GitHub/fjxmlinyang.github.io/_posts/Screen Shot 2021-03-12 at 4.17.02 PM.png" alt="Screen Shot 2021-03-12 at 4.17.02 PM" style="zoom:25%;" />**
2. 很多时候，多分类问题可以比较成为多个两分类问题，两两二分类来做
3. logistics regression并不适用linear的来考虑sysem error $\varepsilon$





Reference:

1) this is the notes for laioffer+ my understanding

2) the bool ISL



### 2.3 Multiple Logistic Regression

$$
\begin{align}
log\frac{p(X)}{1-p(X)} = \beta_0+\beta_1X_1+\beta_2X_2+\cdots+\beta_p X_p \\
p(X) =\frac{1}{1-e^{-(\beta_0+\beta_1X_1+\beta_2X_2+\cdots+\beta_p X_p)}}
\end{align}
$$

（这里可以用前面的值域的博弈论来推导，当然这个就是概率函数$p(x)$和$x$的关系）



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



