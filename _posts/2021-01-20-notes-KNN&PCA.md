---
title: 'notes-KNN&PCA'
date: 2021-01-20
permalink: /posts/2021/01/notes-KNN&PCA/
tags:
  - Models
  - Machine Learning
  - Optimization
---

Here is the notes about KNN&PCA



# 降维&KNN

### 2）PCA（**Principal component analysis**）

2-1）Eigenvalue, eigenvector 

2-2）The Principal Component Analysis (PCA) procedure is a dimension reduction technique that projects the data on k dimensions by maximizing the variance of the data as follows:（方差最大化很自然类别就分开了）

$\min -tr(W^{T}XX^{T}W)$ s.t. $W^TW=I$

### 3） ICA（**Independent component analysis**）

- ICA （独立成分分析）与 PCA 类似，也会找到一个新基底来表示数据，但两者的目标完全不同
- ICA 的一个典型案例是“鸡尾酒宴会问题“：
  - 在一个宴会上有 n 个人同时说话，并且房间里的麦克风只接收到了这 n 个声音的叠加
  - 假定该房间有 n 个麦克风，则每个麦克风记录了说话者声音的不同叠加（由于距离不同）
  - 我们希望基于这些麦克风记录的声音，来还原原始的 n 名说话者的声音信号

### 1）knn（**k-nearest neighbors**）

1)近朱者赤近墨者黑：

给定测试样本，基于某种距离度量找出训练机中与其最靠近的k个训练样本，然后基于这k个“邻居”的信息来进行预测；
通常，在分类任务中使用“投票法”，即选择这k个样本中出现最多的类别标记作为预测结果。
在回归中使用‘平均法’，即将这k个样本的实际值输出的标记的平均值作为预测结果

3）距离问题：欧式距离等

4）为了保证每个特征同等重要性，我们这里对每个特征进行**归一化**。

5）k值的选取，既不能太大，也不能太小，何值为最好，需要实验调整参数确定！

