---
title: 'notes-Model Performance'
date: 2021-01-25
permalink: /posts/2021/01/notes-Model Performance/
tags:
  - Models
  - Machine Learning
  - Optimization
---

Here are notes about model performance







## 实验评估方法（Model Selection）：

1）hold out：stratofied sampling，直接分开两个部分；

2）cross-validation:交叉验证，将数据分成十个部分，9个training，1个test，交替交换，把training里的集合换成test集合

3）bootstrapping，从本来的样本中，产生新的样本，用empirical distribution来形成新的数据集合，有个不被抽到的概率为$(1-\frac{1}{m})^m$， 尤其对集成算法效果不错

4）parameter tuning



## 性能度量（performance measure）

### Classification

In a context of a binary classification, here are the main metrics that are important to track to assess the performance of the model.

##### Confusion Matrix

TP:true positive（真正例）; FN:false negative（假反例）(type II)

FP:false positive（假正例type I）; TN:true negative（真反例）

##### Precision&Recall&F1&Accuracy

Precision ：$P=\frac{TP}{TP+FP}$(查准率precision你说的准确到底有多少可以相信的)； 

Recall Sensitivity：$R=\frac{TP}{TP+FN}$（查全率recall：真的是不是都预测对了）

F1： $F1=\frac{2TP}{2TP+FP+FN}$

Accuracy: $\frac{TP+TN}{TP+TN+FP+FN}$

##### ROC and AUC（ROC下的面积）

3-1)The receiver operating curve, also noted ROC, is the plot of TPR versus FPR by varying the threshold.

3-2)纵轴TPR（recall）$ROC=\frac{TP}{TP+FN}$ ； 横轴FPR $\frac{FP}{TN+FP}$

3-3）补充应用上的内容：

​	1)AUC;		2)Recall and Precision(分别侧重点不同，不能一起很好);	3)用F1可以来统一

##### 代价敏感错误率&代价曲线

1）为权衡不同错误造成的效果；2）重新weight前面的AUC？3）FNR和FPR





### Regression

##### 1)ERM，empirical risk measure：

mean sqaure error: $E(f;D)=\frac{1}{m}\sum_{i=1}^m(f(x_i)-y_i)^2$

1-1）link：在classification上

1-2）这个理论上应该归到statistcal leraning theorey里面

##### 2) coefficient of determination:

The coefficient of determination, often noted $R^2$ or $r^2$ ,provides a measure of how well the observed outcomes are replicated by the model and is defined
$$
R^2=1-\frac{SS_{res}}{SS_{tot}}
$$
Total sum of squares: $SS_{tot}=\sum_{i=1}^{m}(y_i-\bar{y})^2$

Explained sum of squares: $SS_{reg}=\sum_{i=1}^m(f(x_i)-\bar{y})^2$

Residual sum of quares: $SS_{res}=\sum_{i=1}^m(y_i-f(x_i))^2$

##### 3) main metrics

3-1)mallows Cp, AIC, BIC(贝叶斯)，Adujusted $R^2$ link cluster算法

30-2)https://en.wikipedia.org/wiki/Model_selection



## 比较检验

1）有了实验评估方法&性能度量，看起来就能对学习器的性能进行评估比较：先用某种实验评估方法测得学习器的某个性能度量结果，然后对其进行比较，那怎么作比较呢？

2）需要关心的是范化性能&测试集选择上不同&算法本身自带的随机性

3）常用的方法

​	3-1）假设检验；二分法

​	3-2）交叉验证t检验：

​	3-3）McNemar检验：

​	3-4）Friedman检验和Nemenyi后续检验



## Diagnostics

Bias and Variance tradeoff

$E(f;D)=E_D[(f(x;D)-y_D)^2]=E_D[(f(x;D)-\bar{f}(x))^2]$+$(\bar{f}(x)-y)^2+E_D[(y_D-y)^2]==bias^2+var+error^2$

where $\bar{f}(x)=E_D[f(x;D)]$; 

1）$Bias=\bar{f}-f$; 度量了算法的期望预测和真实结果的偏离程度，就是刻画了学习算法本身的拟合能力

2）$Variance=E_D(y_D-y)^2$；同样大小的训练集的变动导致的学习性能的变化，即刻画了数据扰动的影响

3）$Error=E_D(y_D-y)^2$



4)The simpler the model, the higher the bias, and the more complex the model, the higher the variance.

4-1)**Underfitting**: High training error:Training error close to test error&High bias

4-2)**Just right**: Training error:slightly lower than test error

4-3)**Overfitting**:Low training error:Training error much lower than test error&High variance



### Reference:

1)西瓜书

2）vip cheatsheet:machine learning tips

3)how about statistical learning thoery with loss function?

