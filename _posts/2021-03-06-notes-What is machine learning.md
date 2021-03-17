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





bayes： adjusted coeffiency

解释方式1

​	 $P(B \vert A) = P(B) \frac{P(A|B)}{P(A)}$

​	新信息出现后B的概率=B原先的概率*信息的调整（$>1, \ , =1 \ , <1$）

​	$\frac{P(A|B)}{P(A)}=1$说明是独立

​	$\frac{P(A|B)}{P(A)}>1$

​	$\frac{P(A|B)}{P(A)}<1$

 解释方式2

​	 $P(B \vert A) = P(A\vert B) \frac{P(B)}{P(A)}$

​	对概率的偏见，



Reference:

https://www.caseinterview.com/case_interview_frameworks.pdf