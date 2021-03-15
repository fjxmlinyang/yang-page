---

title: 'notes-General Linear Model 3'
date: 2021-01-01
permalink: /posts/2021/01/notes-General Linear Model 3/
tags:
  - Models
  - Machine Learning
  - Optimization
---



先有点废话：

1. 当你有了模型以后，你该如何调整模型，这就是regularization，
2. 最后就是模型的cross validation



那如何保证不overfitting呢，因为这是最需要小心的，那很自然我们就会用到regularization

1. 理解model error 中bias 与variance的关系
2. 知道regularization方法试图解决的问题是什么
3. 了解hyper-parameter的概念，以及优化计算hyperparameter的方法——cross validation



# Model Evaluation

## 1. Overfitting

当你太关心一些细节，那反而在未知的情况下表现的不好。就像你考gre花大量时间在一道极难的问题上,会出现什么情况呢？

<img src="./Screen Shot 2021-03-13 at 9.11.36 PM.png" alt="Screen Shot 2021-03-13 at 9.11.36 PM" style="zoom:50%;" />

- **机器学习的目的是为了构建一个模型，让这个模型尽可能的在未知数据上表现的好。**

- 表现好或者差的定义=$\vert$预测值-真实值$\vert^2$（差值的平方和），所以就是square error或者是mean squre error

- **overfitting的定义**：如果模型在训练数据上表现很好，反而在未知的数据表现的不太好，这就出于overfitting（这里是连接training和）

- 什么情况出现overfitting？

1. Q1: Is $F(x)$ the more complicated the better?

   When your model is too complicated, it may get overfitting.

2. Q2: In clssification problems. are our features the more the better?（features多余你的data）

   1. 比如说你只有两个人的data，但是你用了一堆的feature，这就像你解方程的时候，未知数多余方程数

3. Fundamental causes of overfiitng:

   Complicated Model; Limited learning data.





Loss 里的error：（手段）

- 针对**训练数据**：观察training data里面的$y$和模型预测的值之间的差别

预测error（**Model Error**）

- 针对未来的**未知数据**：未来的$y$和模型之间的差别

感想：

1. 同为一样，但是目的不同，前面是为了找parameter，后面是为了看对未知数据的效果；
2. 换句话说，这就是理想和现实的差距



## 2. Model Error

##### Model error 的定义

定义：是指未来未知的数据（test数据）上的error
$$
MSE=\frac{1}{n}\sum_{i=1}^{n}(y_i-f(x_i))^2
$$
一点比较：model error 很像loss，但是

1. loss function 是用来构造模型；
2. model error 是用来检查模型效果的



##### Model的部分（这个是关于你的training data）

Expected error of regression models(bias+variance的推导)
$$
E(Error)=bias^2+variance+Irreducible
$$
Error的期望组成：（是model 的error）

1. **Bias**： measures how far off **in general** the models predictions are from the correct value
2. **Variance**: is how much the predictions for  **a given point** vary between <u>different realization of the model</u>
3. (不重要)Irreducible error：the part that can not be reduced by optmizing the model

一个例子：

1. 当你数据不同的时候，loss function的parameter就会有些不同，所以bias是平均你的parameter



##### 一个类比（这个是training data）

- 学生的考试能力 = 你的平均水平（与这一次没有关系）+这一次的发挥来说对你多正常
- 模型的精准性bias：
  - 第一层次的理解：模型输出结果与真实值之间的差距。错误！！！
  - 第二层次的理解：这个model在训练数据有变化下的平均输出结果与真实值相比，得到的平均准确性
- 模型的稳定性variance：
  - 第一层次的理解：模型输出结果的稳定性
  - 第二层次的理解：’**某一次**model的数据结果与这次model的平均水平的差距‘的平方的期望



<img src="./Screen Shot 2021-03-13 at 9.47.02 PM.png" alt="Screen Shot 2021-03-13 at 9.47.02 PM" style="zoom:50%;" />





Error due to Variance:

- The error due to variance is taken as the variability of a model prediction for **a given data point**
- Again, imagine you can repeat the entire model building process multiple times. The variance is how much the predictions for a given point vary between different realizations of the model.



Q：那你怎么从一个traning的bias 和variance去看model本身的bias和variance呢？

Ans：这里的模型空间，会随着模型复杂度变大，很自然，你的variance增大，你的bias减小

重要事情：

<img src="./Screen Shot 2021-03-13 at 9.55.08 PM.png" alt="Screen Shot 2021-03-13 at 9.55.08 PM" style="zoom:50%;" />

要回答overfitting和underfitting可以用上面这个图

	1. 第一部分，是指在testing data上的表现？就是上面的图？对的，注意纵坐标是testing error；
 	2. 第二部分，是symptoms是提到

要回答bias和variance可以用下面这个图（**这个是对training data的**， 就是上面的表格？对的因为在看你的model的参数，就是参数a和b）

例子：有个点（X，Y）它对应的下面四张图代表了四个模型，请判断他们的bias 和variance什么状态

​	1. 这是你不同的model，不同的loss的parameter不一样，因为training data 不一样

<img src="/Users/yanglin/Documents/GitHub/fjxmlinyang.github.io/_posts/Screen Shot 2021-03-13 at 9.58.41 PM.png" alt="Screen Shot 2021-03-13 at 9.58.41 PM" style="zoom: 50%;" />



## 3. How to solve the overfitting problem？

1. Increase traning data size

2. avoid over-traning your dataset:

   1. Filter out features, e.g. feature reduction

      Principal component analysis(PCA)

   2. **Regularization**

      **Ridge regression, Least absolute shrinkage and selction operation(LASSO)**

      **Logistic Regression-L2, Logistic Regression-L1**

   3. Ensemble Learning







# Regularization

情况：在你的testing data 上出现了overfitting的情况（直接在你的loss function上来治疗你的问题，所以这是在testing上增加的？）

所以我们要对overfitting做点事情

当$f(x_i)=ax_i+b$时：
$$
Loss= \sum_{i=1}^{n}(y_i-ax_i-b)^2
$$
Loss特别小的时候，model error 会很大，因为什么？ var 大(相当于之前只和training有关系，但是我们需要和testing建立连接)
$$
Loss= \sum_{i=1}^{n}(y_i-ax_i-b)^2+testing \ variance
$$
可以吗？不行。testing data 不知道，testing var 不知道

Model training process
$$
Loss= \sum_{i=1}^{n}(y_i-ax_i-b)^2+间接描述\ testing \ \  variance
$$
例子：y=ax+b,a 和b已知 a=0.5, b= 85，相当于能连接training 和test的，间接描述testing的，只有这个parameter的函数了
$$
Loss= \sum_{i=1}^{n}[(y_i-ax_i-b)^2+w(a,b)]
$$


## Ridge Regression: L2 penalty in loss function



**Ridge Regression** improves the loss function definition of Linear Regression by introducting  variance into formula

**Regularization** is nothing but adding a penalty term to the objective function and controlling the model complexity using that penalty term

**Training  stage:**
$$
Loss=\sum_{i=1}^{n}(y_i-\beta_0-\sum_{j=1}^n \beta_j x_{ij})^2+\lambda\sum_{j=1}^p\beta_j^2
$$
$MSE=\frac{1}{m}\sum_{i=1}^{n}(y_i-\beta_0-\sum_{j=1}^n \beta_j x_{ij})^2$



Ridge regression 就是linear regression加上平方项的penalty term



1. 画出来像等高线
2. 后来可以推广到logistics, 所以就是一个是L1，另外一个就是L2



## LASSO: L1 penalty in loss function

Lasso(Least absolute shrinkage and selection operator)
$$
Loss=\sum_{i=1}^{n}(y_i-\beta_0-\sum_{j=1}^n \beta_j x_{ij})^2+\lambda\sum_{j=1}^p \vert \beta_j \vert
$$


## Hyperparameter Optimization

这样的penalty公式中的$\lambda$ 就是hyperparameter：

1. 　$\lambda$可以让方差和偏差达到平衡：$\lambda$增大，模型方差（variance）减少，偏差增大（bias）
2.  相当于你在用$\lambda$来控制你的参数a和b





总结：

1. L1:：feature selection regulariation

L2：是correlation的情况处理的比较好

2. model error 不存在training上，mse是用来测试机器好不好，所以不是不用Regularization
3. L1和L2一般就用在linear 和logistics上现在，不过regulizaration这个概念在其他地方都有



建模分两个部分，

实验室

1. 骨架：选择你的大的框架
2. 肉：就是你里面的parameter，min loss function就是填肉

面对世界

1. 做出来这个机器人到底行不行，
2. 这时候就要regularization（要解决overfitting（low bias/high variance），虽然已经在testing data上了，但是还是要回到training data 里面），所以是骨架，还是肉呢？





1. 用regularization的时候只是在training data里，目的是为了调整parameter
2. 从而当用于testing data 时候，我们就取消这个regularization的这个项对么？



Q：对应logistics

bias has a negative first-order derivative in response to model complexity while variance has a positive slope这个图的注解是这样的



http://scott.fortmann-roe.com/docs/BiasVariance.html

https://www.zhihu.com/question/27068705

pattern recognition？



# SVM

1.refer to your publications

$E((w^Tx+b,0)_{\max})$?

2. Maximum margin classifier

   $\max \frac{1}{\vert \vert w \vert \vert},\qquad s.t. y_i(w^Tx_i+b)\geq 0,\qquad i=1,\cdots,n$

3. $\min \frac{1}{2}\vert\vert w \vert\vert^2,\qquad s.t. y_i(w^Tx_i+b)\geq 1,\qquad i=1,\cdots,n$

   Dual  $\mathcal{L}(w,b,a)=\frac{1}{2}\vert \vert w\vert \vert^2-\sum_{i=1}^n \alpha_i(y_i(w^Tx_i+b)-1)$ 变形后

   $\mathcal{L}(w,b,a)=\sum_{i=1}^n\alpha_i-\frac{1}{2}\alpha_i\alpha_jy_i y_jx_i^T x_j$ And $w=\sum_{i=1}^n\alpha_iy_ix_i$

   dual problem 

4. 换成核估计

   $f(x)=\sum_{i=1}^Nw_i\phi_i(x)+b$ 转换成为 $f(x)=\sum_{i=1}^l\alpha_i y_i \langle \phi(x_i), \phi(x) \rangle+b$

$\alpha$ 可以由dual 来求

$\max_{\alpha}\sum_{i=1}^n \alpha_i-\frac{1}{2}\sum_{i,j=1}^n\alpha_i\alpha_jy_iy_j\langle \phi(x_i)\phi(x_j)\rangle \qquad s.t. \alpha_i \geq 0,i=1,\cdots, n; \sum_{i=1}^n\alpha_iy_i=0$ 

$\max_{\alpha}\sum_{i=1}^n \alpha_i-\frac{1}{2}\sum_{i,j=1}^n\alpha_i\alpha_jy_iy_jx_ix_j \qquad s.t. \alpha_i \geq 0,i=1,\cdots, n; \sum_{i=1}^n\alpha_iy_i=0$ 





