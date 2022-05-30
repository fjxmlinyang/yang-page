---
cutitle: 'notes-What is ML for'
date: 2021-01-25
permalink: /posts/2021/01/notes-Model Performance/
tags:
  - Business sense
  - Machine Learning
---



say something for general ml?



# What is ML for?

Business Situation Framework

Customer/ Product/ Company/Competition

Customer: customer value, customer product, company retention.

Product: product reviews/quality

Company: cost structure

Competition: market share competitor,etc







总的来说：数据——方法——结果

面试：数据多大？你怎么evaluation？

#### What is Artificial Intellgence?

图灵测试



deep learning：

1. 数据不够，数据bias，不知道如何使用数据
2. 它可以把这feature和model混在一起



**常用的方向：**

1. recommendation systems
2. search engine
3. 银行/金融，social media
4. 生物特征识别的公司



## **types of machine learning**

1. supervised learning(直到明确标识)
   1. linear/logistic regression
   2. decision Tree
   3. random forest
   4. K-NN
   5. SVM

2. unsupervised learning
   1. K-means
   2. PCA
   3. LDA
3. reinforcement learning
   1. Goal/State/Actions/Reward









# Basic Probability

## randomness?

1. the probabiltiy distribution ss a description of a random phenomeon

## random variable?

1. r.v.
2. Parameter  vs random variable:
   1. the paraemters are numbers which **helps** uniquely define the probability distribution or model
   2. also, it can be determined by the historical data after the parameter estimation methods

## type of random variable?

1. Discrete：bernoulli？et.c

   Bernoullli？$P(Y=y)=p^{y}(1-p)^{1-y}$, $0<p<1$, $y=0,1$

2. Continuous： normal， weibull？







## cdf/pdf

pdf是cdf的求导，cdf是pdf的积分，这个相当于是密度和面积（体积）的概念。





## CLT/SLLN

从无到有







## Independent/Dependence





## bayes： adjusted coeffiency

解释方式1

​	 $P(B \vert A) = P(B) \frac{P(A|B)}{P(A)}$

​	新信息出现后B的概率=B原先的概率*信息的调整（$>1, \ , =1 \ , <1$）

​	$\frac{P(A|B)}{P(A)}=1$说明是独立

​	$\frac{P(A|B)}{P(A)}>1$

​	$\frac{P(A|B)}{P(A)}<1$

 解释方式2

​	 $P(B \vert A) = P(A\vert B) \frac{P(B)}{P(A)}$

​	对概率的偏见，







Actually, see my course slides on probability and statistics







































# Probability vs Statistics

- in probability theory we consider some underlying process which has some randomness or uncertainty modeled by random variables, and we figure out what happens. It is a numerical description of the likelihood of an event.
- in statistics we observe something that has happened, and try to figure out what underlying process would explain those observations. It concerns the collection, organization, displaying analysis, interpretation and presentation of data.
- 概率是根据你知道的背后的uncertainty来计算轨迹（演绎）。统计是根据观察到的很多的信息，来进行解释/推断（推断）。



Reference:

https://www.caseinterview.com/case_interview_frameworks.pdf







