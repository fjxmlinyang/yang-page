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

# Model Evaluation/Model Performance

# 1. Why do model evaluation?

- 这个部分相当于是你做完clean/feature engineering/ model，就到model evaluation了，说白了就是说，你模型到底如何

<img src="./Screenshot from 2021-03-22 20-54-26.png" alt="Screenshot from 2021-03-22 20-54-26" style="zoom:70%;" />











# 2. How to evaluate a model?

**（面试题目）**Evaluation needs data + method(metrics)

- What data to evaluate: select and split data
- What method to use: cross validation
- What metrics to compare: confusion matrix, ROC, AUC



notes：你在evaluate a model，其实也在model selection

## 2.1 What method to use（实验评估方法/Model Selection）——Cross Validation

### What is Cross Validation?

- Assess how your model result will generalize to another independent data set.
- Predict and test on the same data is a methodological mistake

- There are several cross validation techniques, popular is k-fold cross validation

### K-fold Cross Validation

一些简介：

- 假设data固定位置是不能动的，第一次相当于选80%，
- 每一次training 和validation不能overlap
- 任何两次validation 都不overlap

- Performance  based on the validation data

$$
Performance = \frac{1}{N}\sum_{i=1}^n performance_i
$$

- Holdout 用于test， holdout data只用在最后一次



经典教材过程：

- 建模前，留一部分data（holdout data）
- 1）一开始用cross validation，挑选一个好的模型，做骨架；
- 2）接着用所有的training+validation 再training 一遍，找到parameter，填肉；
- 3）最后，用holdout data 得到最后一个客观的evaluation结果。holdout data（test data）只用一次



如图：

<img src="./Screenshot from 2021-03-22 21-05-10.png" alt="Screenshot from 2021-03-22 21-05-10" style="zoom:80%;" />

### Model Selection with Cross Validation

- Use the cross validation method to do hyperparameter tuning
  - e.g. k value in knn, lambda in regularization
- Cross validation can only validate your model selection

翻译成中文

- Cross validation ->找骨架（model selection/model infrastructure）（相当于确认框架）
- mixed validation and training data into training ->找肉，例如y= ax+b里面的a和b （相当于确认细节）

注意上面的图，**这两部用的数据不是完全一样**，第一个是绿vs蓝；第二个相当与多了holdout，绿vs红





### 一些补充，和统计的关系

- hold out：stratified sampling，直接分开两个部分；

- cross-validation:交叉验证，将数据分成十个部分，9个training，1个test，交替交换，把training里的集合换成test集合

- bootstrapping，从本来的样本中，产生新的样本，用empirical distribution来形成新的数据集合，有个不被抽到的概率为$(1-\frac{1}{m})^m$， 尤其对集成算法效果不错

- parameter tuning



## 2.2 What metrics to compare（性能度量performance measure）

- 这里相当于就是在前面的section里面，你的performance选的是什么呢？

## 2.2.1 Classification

In a context of a binary classification, here are the main metrics that are important to track to assess the performance of the model.

### Confusion Matrix

<img src="./Screenshot from 2021-03-22 21-33-02.png" alt="Screenshot from 2021-03-22 21-33-02" style="zoom:67%;" />



TP: true positive（真正例）; FN: false negative（假反例）(type II)

FP: false positive（假正例type I）; TN: true negative（真反例）



- **True/False** means if you made a correct/wrong prediction
- **Positive/Negative** means what your prediction is/is not









**(面试)**例子：positive 生病了

- true positive：你说对了（true），你说是他生病了-——你认为他有病，他病了
- false positive：你说错了（false），你说他生病了——你认为他生病，他没病
- true negative：你说对了（true），你说他没生病——你认为他没病，他没病
- false negative：你说错了（false），你说他没生病——你认为他没病，他生病了



#### Difference metrics（Accuracy/Precision/Recall/F1）

**Accuracy: $\frac{TP+TN}{TP+TN+FP+FN}$**

- 预测中你对的多少个

**Precision ：$P=\frac{TP}{TP+FP}$**

- 查准率precision：你说的准确到底有多少可以相信的
- 在**所有你认为**positive的数据中，有多少真的是positive？
- spam email：要求precision高（杀的必须准）

**Recall Sensitivity：$R=\frac{TP}{TP+FN}$**

- （查全率recall：真的是不是都预测对了）查全率recall：真的是不是都预测对了
- 在所有positive的数据中，有多少**被你正确地识别出**来（是positive）
- disease/cybersecuirity：要求recall高( 宁可错杀)
- 如果negative很重要，你看recall，反之你看precision更关注positive

**F1： $F1=\frac{2TP}{2TP+FP+FN}=\frac{2}{\frac{1}{recall}+\frac{1}{precision}}$(可以统一recall&precision)**

- 越高越好





### Example

<img src="./Screenshot from 2021-03-22 21-46-22.png" alt="Screenshot from 2021-03-22 21-46-22" style="zoom: 70%;" />







### ROC and AUC（ROC下的面积）

Precision ，recall=> classifier; threshold(分割positive or negative 的临界概率，默认0.5)，**如何独立于threshold来评价一个classifier的分类能力？**这就是ROC curve的使用

- Precision—Recall也是一种方法

### ROC

- Receiver operation characteristic curve（根据你的threshold来制定的，threshold是用来就判断是positive or negative，像logistic regression里，我高于0.8,判定为positive，当然你可以选0.6）
- Define False Positive Rate as X axis, True Positive Rate as Y axis
  - 纵轴TPR（recall）$ROC=\frac{TP}{TP+FN}$ ； 横轴FPR $\frac{FP}{TN+FP}$
  - False positive **rate**= Number of False positive  / Number of real negative
  - True positive **rate**= Number of True positive / Number of real positive = recall

- The receiver operating curve, also noted ROC, is the plot of TPR versus FPR by varying the threshold.
- Special Points in ROC space
  - best case(0,1); worst case:(1,0)
  - 对角线上的点：
    - 当threshold设定为最高时，所有样本都被预测为negative，此时得到的点在$(0,0)$.
    - 当threshold设定为最低时，所有样本都被预测为positive，此时得到的点在$(1,1)$.
  - why does Equal Erro rate mean FPR = FNR?
    - 已知固定关系： FNR=1-TP/Number of real positive = 1- TPR
    - 根据图中焦点性质可知： FPR （x）= 1-TPR（y）
    - FPR=FNR

### AUC

你想完整的表示前面的auc么？

- Area under the curve of ROC(AUC)
- AUC value: [0,1]
- The larger the value is, the better classification performance your classifier has.
- AUC value is a probability value.

**面试题：机器学习里0-1的值，都希望有一个概率。怎么用概率来解释AUC？**

ROC AUC is the probability that  a randomly-chosen positive example is ranked more highly than a randomly-chosen negative example.

- 就是在random排序里面，positive example >negative example的概率(如果都是positive大于negative，这是完美排列,你的true positive rate都是对的且一直是1,那就是AUC=1咯)

- for example

- $$
  \begin{align}
   0.8 \ \   0.7 \ \   0.6 \ \ 0.5 \ \ 0.4 \ \ 0.3\\
   1 \ \ 1\ \  1 \ \ 1\ \ 1\ \  0
  \end{align}
  $$

- 如果你放任何一个threshold，true是不动的，唯一有区别的是negative，比如你设threshold在0.45, TP/(TP+FN)=4/4. FP/(TN+FP)=0/2。即使你继续提高threshold，你也只会在x=0的这条上移动



注意看清楚你的x轴&y轴，有时候会让x轴变成1-false positive rate

![Screenshot from 2021-03-26 08-50-11](./Screenshot from 2021-03-26 08-50-11.png)













一点高阶补充：

#### 代价敏感错误率&代价曲线

1）为权衡不同错误造成的效果；2）重新weight前面的AUC？3）FNR和FPR



## 2.2.2 Regression

##### 1)ERM，empirical risk measure：

mean square error: $E(f;D)=\frac{1}{m}\sum_{i=1}^m(f(x_i)-y_i)^2$

1-1）link：在classification上

1-2）这个理论上应该归到statistcal leraning theory里面

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



# 3 Failures Analysis

- 出现failures，就是各种false的结果怎么办？我们就用到failures analysis
- 方法：要分类，分成不同的failures，分别讨论怎么办
- 不同的策略来解决减少failures（把你的failures的分类）
  - 可以通过模型解决
  - 可以通过调参解决
  - 可以通过data解决(可能你少一部分data)
  - 不能解决
- failures analysis的目的就是进一步提高模型的性能，进行迭代开发，retraining



总结：

- cross validation——找骨架（model selection/model infrastructure）
- mixed validation and training data into training——找肉，例如y=ax+b里面的a和b
- model evaluation——ROC，AUC，Precision，Recall
- failure analysis——确定问题在哪里，然后更新step1-3







# 4 Machine Learning End-to-End Pipeline

The end to end ML/DL pipelines play an important role in building production ready AL/ML systems. Itneeds the collaboration together of data scientists, data engineers, and full-stack developers, frond-end developers, IT operations and even hardware and system engineers. There are many common steps in ML/DL pipelines that should be automated and tracked.

End to end ML/DL in production usually contains many steps:

- Data acquisition
  - collect data from outside system
  - pre-process into the format that needed
- Data preparation
  - Clean, transform, validate and select data
  - annotation
  - feature extraction
  - prepare data into: training, validation and evaluation datasets
- Training & Validation
  - Select the model
  - train the model
  -  tune the model
  - validate the training process
- Testing Evaluation
  - evaluate the performance metrics on evaluation dataset
  - analyze failures
- Deployment& Inference
  - Integrated into production apps
  - config models based on system needs
  - predict on real-life data
  - log failures/errors

<img src="./Screenshot from 2021-03-25 11-35-41.png" alt="Screenshot from 2021-03-25 11-35-41" style="zoom:80%;" />

别傻乎乎的就觉得training&validation很重要，其他也很重要



DE：数据怎么来，数据去哪里，数据简单清理(bigdata/spark/sql)

BA/DA：数据预处理

DS：有些数据clean，training/testing/deployment（放在软件系统里面）

MLE/SDE：放在系统里面如何使用deployment/inference













# 5高阶补充：2.2.3

### 比较检验

1）有了实验评估方法&性能度量，看起来就能对学习器的性能进行评估比较：先用某种实验评估方法测得学习器的某个性能度量结果，然后对其进行比较，那怎么作比较呢？

2）需要关心的是范化性能&测试集选择上不同&算法本身自带的随机性

3）常用的方法

​	3-1）假设检验；二分法

​	3-2）交叉验证t检验：

​	3-3）McNemar检验：

​	3-4）Friedman检验和Nemenyi后续检验



### Diagnostics

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



# Reference:

1)西瓜书

2）vip cheatsheet:machine learning tips

3)how about statistical learning thoery with loss function?





from林轩田 8topics in ML

1. when can machines learn ?
2. why can machines learn?
3. how can machines learn?
4. how can machines learn better?
5. how can machines learning by embedding numerous features
6. how can machines learning by combining predictive features?
7. how can machines learning by distilling hidden features?