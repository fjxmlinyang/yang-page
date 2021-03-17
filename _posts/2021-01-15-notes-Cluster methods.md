---
title: 'notes-Cluster methods'
date: 2021-01-15
permalink: /posts/2021/01/notes-Cluster methods/
tags:
  - Models
  - Machine Learning
  - Optimization
---

Here are notes about cluster method. 

Question: is it related to kernel method?





# Cluster



### Protoptype-based cluster（**k-means clustering**&LVQ&Mixture of Gaussian）

##### K-means clustering

1）是个non supervise learning， 只用${x^{(1)},\cdots, x^{(m)}}$;

2) 具体流程；

2-1）对于每个$i$（训练集的大小）（c是记录它属于的cluster label）

​	
$$
c^{(i)}:=\arg \min_{j}\vert \vert x^{i}-\mu_j \vert \vert ^2
$$
**将每个训练样本$x^{(i)}$分配到距离最近的中心$\mu_j$**

2-2) 对于每个$j$( 聚类的数量) **（更新中心，平均值）**
$$
\mu^{(j)}:=\frac{\sum_{i=1}^m\{c^{(i)}=j\}x^{(i)}}{\sum_{i}^m\{c^{(i)}=j\}}
$$
将每个聚类中心移动到第一步分配到该中心的样本的均值

3）收敛情况，

3-1)定义 **distortion function**： $J(c,\mu)=\sum_{i=1}^m \vert \vert x^{(i)}-\mu_{c^{(i)}}\vert \vert^2$

3-2)不一定保证最优，因为是非凸函数

##### 和em算法&GMM的关系

1）em算法，第一步，选择归于哪个类（比如二项分布的哪个类型）；第二步调整中心

1-2）GMM可以用EM的框架来推导；Kmeans 是GMM简化

1-3）https://www.zhihu.com/question/49972233

https://papers.nips.cc/paper/1994/file/a1140a3d0df1c81e24ae954d935e8926-Paper.pdf

https://zhuanlan.zhihu.com/p/36331115

##### 一个来自金融的例子：

1）目标函数是关于correlation between the returns of the assets. $\sum_{i=1}^n\sum_{j=1}^n\rho_{ij}z_{ij}$,  2）for every asset $i$, we have to find a representative asset $j$;3) $z_{ij}$ 是来找它自己 asset $i$来找其代表asset $j$； 4） constaint： $\sum_{j=1}^nz_{ij}=1$; $\sum_{i=1}^ny_i \leq k$; $z_{ij}\leq y_j$; $z_{ij}, y_i \in \{0,1\}$

### density based clustering& **hierarchical clustering**

1）It is a clustering algorithm with an agglomerative hierarchical approach that build nested clusters in a successive manner.

2）自底向上聚合类，先将数据集中的每个样本看作是一个初始cluster，然后在算法运行的每一步中找到距离最近的两个聚类进行合并，这个过程不断重复，直到找到预设的聚类个数。

3）每个聚类是一个样本集合，计算距离可以用最小&最大&平均距离

4）Clustering assessment metrics： Silhouette coefficient&Calinski-Harabaz index





Reference：

1）西瓜书；2）cheatsheet cs229

