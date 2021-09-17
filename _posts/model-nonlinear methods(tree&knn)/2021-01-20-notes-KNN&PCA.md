---
title: 'notes-KNN&PCA'
date: 2021-01-20
permalink: /posts/2021/01/notes-KNN&PCA/
tags:
  - Models
  - Machine Learning
  - Optimization
---

Here are notes about KNN&PCA



# KNN

## Introduction

- K denotes the k labels that are 'closest' to your target sample

  the higher K is, the smoother(most robust) your model can be. 

- Multiple definition choice for ‘distance’
  
  - CityBlock Distance, Euclidean Distance, Gaussian Distance



#### Implement KNN algorithm

- Find the top K closest distance from your returned unsorted distance array(heap sort会提到)
- 面试：关键其实是coding，如何sorting的问题



1)近朱者赤近墨者黑：

给定测试样本，基于某种距离度量找出训练机中与其最靠近的k个训练样本，然后基于这k个“邻居”的信息来进行预测；
通常，在分类任务中使用“投票法”，即选择这k个样本中出现最多的类别标记作为预测结果。
在回归中使用‘平均法’，即将这k个样本的实际值输出的标记的平均值作为预测结果

3）距离问题：欧式距离等

4）为了保证每个特征同等重要性，我们这里对每个特征进行**归一化**。

5）k值的选取，既不能太大，也不能太小，何值为最好，需要实验调整参数确定！

## Implement KNN algorithm

Find the top K shortest distance from your returned unsorted distance array(coding heap sort)



## FLANN annoy faiss

复杂度太高该怎么办？

- locality-sensitive hashing（https://github.com/spotify/annoy）
  N is total number of training data;
  m is average data size of split areas
  $O(N) \rightarrow O(m) \ \ (m << N)$
- 相当于你从找点，现在开始找区域

- 如何构造映射？geo-hashing









# PCA（**Principal component analysis**）

别忘了它也是个model，它在feature selection上的使用，请见后面的details

2-1）Eigenvalue, eigenvector 

2-2）The Principal Component Analysis (PCA) procedure is a dimension reduction technique that projects the data on k dimensions by maximizing the variance of the data as follows:（方差最大化很自然类别就分开了）

$\min -tr(W^{T}XX^{T}W)$ s.t. $W^TW=I$



# ICA（**Independent component analysis**）

- ICA （独立成分分析）与 PCA 类似，也会找到一个新基底来表示数据，但两者的目标完全不同
- ICA 的一个典型案例是“鸡尾酒宴会问题“：
  - 在一个宴会上有 n 个人同时说话，并且房间里的麦克风只接收到了这 n 个声音的叠加
  - 假定该房间有 n 个麦克风，则每个麦克风记录了说话者声音的不同叠加（由于距离不同）
  - 我们希望基于这些麦克风记录的声音，来还原原始的 n 名说话者的声音信号







# Feature Selection

##### 1Feature selection的常用方法

1. Manual selection with Pearson Correlation Value
2. Regularized Models
3. Principal Component Analysis(如果是从多的feature到低的feature)

##### 2.减少 feature multicollinearity对model testing error 影响的常用方法

- Manual selection with Pearson Correlation Value;
- Regularized Models(L1);
- Regularized Models(L2);可以提高精度/但是不能用feature selection
- Ensemble learning model, e.g. RF；这个是feature importance
- Principal Component Analysis



## 1.Why feature selection?

why feature selection? $f(X_1,\cdots,x_{100}) \rightarrow Y$

- example:有些feature强相关我们就不太需要
- feature selection examines the **strength of the relationship** of the feature with the response variable.(想知道和你的response variable)
- It can improve our modeling work form the following perspectives:
  - better understanding of your model(想知道关系呀)
  - improve model stability(i.e. reduce model variance). multicollinearity（多重共线性强相关性，如果又这样的情况，会非常不稳定）
    - 常有的状况，可能是feature太多，data太少
- Feature Selection的效果——reduce model variance



一个例子，

- 当$X_1$和$X_2$有关系的时候，很容易造成不stable，你的coeffieient 不稳定,因为你只知道共同的贡献
- 比如说你可以用 $Y=2X_1+3X_2+2X_3+3X_4$或者$Y=-2X_1+7X_2+2X_3+3X_4$，你看你的强相关就会导致model不stable
- 这个可以在置信区间里面看到



那问题来了，怎么解决共线性呢？问题如下



## 2 Feature Selection Methods

### 1. Manual selection with Pearson Correlation Value

- 注意，这个说白了就是$X$之间的关系，就是内卷的关系，这都不需要模型（后面两种都是用着建模来解决）

- 这个初衷，就是找到他们之间的关系，虽然数据很重要，那去掉的数据就该找找的很相似的（比如说话费和通话时常）

- use pearson correlation coefficient to measure linear dependency between features
  $$
  \rho_{X_1,X_2}= \frac{cov(X_1,X_2)}{\sigma_{X_1}\sigma_{X_2}}
  $$
  

  - $cov(X_1,X_2)$ means covariance between $X_1$, $X_2$(联合变化程度), $\sigma_X$ means standard deviation
  - $cov(X_1,X_2)=E(XY)-E(X)E(Y)$





### 2. Regularized Models

Use regularization methods to do feature selection

- regularization review: regularization is a method for adding constraints or penalty term to a model, with the goal or preventing overfitting and improving generalization(相当于在training data上放大一点点，让其在testing上效果比较好)

- $$
  \sum_{i}^n(y_i-h_{\theta}(x_i))^2 +\lambda \vert\vert\theta \vert\vert
  $$

  

- L1 regularization

  - $$
    \lambda \sum_{i}^n \vert \theta_i \vert
    $$

  - L1 regularization tends to provide sparse solutions

  - 缺点：计算结果不稳定（因为你是not differentiable的，可能会有多个最小值）（但是他在运行中会让这个不稳定的系数相对小一些）（可能是原始数据有问题）

- L2 regularization

  - $$
    \lambda\sum_{i=1}^n \theta_i^2
    $$

  - L2 regularization tends to spread out more equally

  - L2 regularization is more stable(你只个只能improve但是却没法选)（不是feature selection的方式）





**（面试题）问题1：features selection 有几个方法？**

- pearson corr
- L1，把想象的feature的coefficient变成0
- Ensemble learning 中的feature importance（RF，不能用decision tree）
- PCA，can helps reduce number但是和前面三个不太一样哪个

**（面试题）问题2：在系统中有high corr 的features，怎么来提高模型精度？（你提高精度，说白了就是feature selection+L2）**

- pearson corr
- L1
- Ensemble learning
- PCA
- L2

**（面试题）问题3：如果有high corr的features，model testing的error很高，怎么办？**

- 扔feature
- l2 或者random forest， reduce var的方法
- 前两个选谁？ 你得看模型，feature可能不多，当然能不扔就不扔。
  - random forest好处就是没有扔feature，但是自己过滤掉，而不是人为的。总是feature会有不同侧重点嘛





### 3. Other Feature Selection Methods

可以利用嗯某些non-linear model自带的feature importance 功能， eg random forest-feature importance



Multicollinearity(feature的关联性多) -> high model variance（变成overfitting） -> high testing error



### 4. Principal Component Analysis(如果是从多的feature到低的feature)

#### Why PCA?

- Dimensionality reduction
  - remove feature correlation
  - feature selection- reduce model overfitting(比如说你拍照，景色的feature已经改变了)

#### Objective of PCA

- Given original data in d-dimensional space, we want to find a low-dimensional projection(k-dimensional space, where k<d) such that it captures the most of the 'variability' in the original data and **minimizes the reconstruction error.**



##### PCA的证明: 忽略



#### Intuition of PCA



##### 1. What  do we want to do?

就尽可能多扔掉一些数据，select only one or a few 'important' 'features' from the original feature space

##### 2. Problems we need to solve:

- how to quantitatively describe the correlation between axis - Covariance matrix
- how to reduce correlation between axes by performing linear transformation-----eigendecomposition
- how to describe 'importance' among axes.----Eigenvalue

##### 3. Covariance matrix

$$
A= \sum = \begin{matrix}
 E[(X-\mu)^2] &  E[(X-\mu)(Y-v)]  \\
 E[(X-\mu)(Y-v)] &   E[(Y-v)^2]
\end{matrix}
$$



##### 4. Eigen decomposition for covariance Matrix

- Convect covariance matrix to diagonal matrix
- $A= Q \Sigma Q^{-1}$, $Q^{-1}A= \Sigma Q^{-1}$, $Q^{-1}AQ =\Sigma$

##### 5. Select Dimensions



##### example:

Suppose the data is D=1000行*10列

step0:对data D做feature normalization，ie 数据归一化处理

step1:计算协方差矩阵： A=10*10

step2:对协方差矩阵做eigen-decomposition，即 $AQ= Q\Sigma$

假设我们要取5个features则$\Sigma= 5*5$,即特征值top 5, where Q= 10*5，  

step3: 用投影矩阵Q对data做降维

D* Q = new_D （这个D相当于linear regression里的系数呀，pca其实就是个模型）





#### Details of PCA（面试题）

注意它的feature已经改变了

- Step1：对样本矩阵做feature normalization，ie数据归一化处理
- Step2：计算样本矩阵的协方差矩阵（covariance matrix）
- Step3：对协方差矩阵做特征值分解（Eigendecomposition），选取最大的前d个特征值所对应的特征向量，构成投影举证（projection matrix）
- Step4：利用投影矩阵对样本矩阵进行projection变换，得到降维后的新矩阵



其实它出来的就是压缩模型





#### Importance notes of PCA

- PCA is sensitive to the relative scaling of the original variables
- We can also use PCA to do feature selection

注意，PCA之后feature变化了，feature的数量也变化了。所以这里的feature selection= feature 变化之后再做selection

example: 你用\rho告诉老板，老板很开心；但是PCA呵呵，你的feature没法解释呀



列全部的大纲和bullet point





### 5.补充linear algebra

直线加工出一个新的直线

5.1 matrix: can be considered as linear transform&change main direction

​	这个就是运动咯

5.2 eigenvalue and eigenvector

​	一个是运动方向，另一个是运动速度

5.3 Eigen-decomposition



如果是不对称的，记得用**奇异值分解**（singular value decomposition）





Reference

https://www.zhihu.com/question/20534668





列全部的大纲和bullet point



数据来源&数据前期&原处理的认知；