---
title: 'notes-Tree based methods'
date: 2021-01-05
permalink: /posts/2021/01/notes-Tree based methods/
tags:
  - Models
  - Machine Learning
  - Optimization
---

We discuss tree based methods here.(Now here include ensemble methods)



#### 简介：

1）decision tree 是属于decision tree learning这一块的，如果是二分的问题，就是基于每个feature进行分类，在树结构上进行classfication和regression（divide and conquer strategy）

2）tree结构：In keeping with the tree analogy, the regions $R_1$, $R_2$ and $R_3$ are known
as terminal nodes or leaves of the tree.

3)Tree based methods partition the feature space into a set of rectangles, and then fit a simple(like a constant) in each one.

4)  link to the GAM& linear regression

$E[Y|X_1,\cdots,X_p]=\alpha+f_1(X_1)+\cdots+f_p(X_p)$  Generalized additive model(GAM)



linear regression: $f(X)=\beta_0+\sum_{j=1}^PX_j\beta_j$

tree regression: $f(X)=\sum_{m=1}^M c_m I_{(X \in R_m)}$

#### 原理1：

如何分割，其实相当于你在平面上来分割，可以当作是GAM model

**Prediction via Stratification of the Feature Space**(相当于你在特征空间分层预测,基本上和linear的不一样的地方，linear是线，这里是竖的还有横的)

1. We divide the predictor space—that is, the set of possible values for
   $X_1,X_2,\cdots, X_p$—into $J$ distinct and non-overlapping regions $R_1, R_2, \cdots, R_J$.
2. For every observation that falls into the region $R_j$ , we make the same
   prediction, which is simply the mean of the response values for the
   training observations in $R_j$ .

For step 1, **how do we construct the regions**  $R_1, R_2, \cdots, R_J$?

1. **a top-down, greedy approach** 自上而下"说的是从树顶分割成两个新的分支 The approach is top-down because it begins at the top of the tree (at which point
   all observations belong to a single region) and then successively splits the
   predictor space; each split is indicated via two new branches further down
   on the tree.
2. that is known as ***recursive binary splitting***
3. **greedy:** It is greedy because at each step of the tree-building process,在建立树的每一个步骤中
4. **best：**he best split is made at that particular step, rather than looking ahead
   and picking a split that will lead to a better tree in some future step.分裂确定的仅限于某一部进程，而不是针对区选择能够在未来进程中构建出更好的树的分裂点t
5. 寻找继续分割数据及的最优预测变量&最优分割点

#### 原理2：

6）**如何判定好坏**，比较前后的misclass的比例；用三个不同的entropy&gini

​	tree regreesion: RSS $\sum_{j=1}^J\sum_{i \in R_i}(y_i-\hat{y}_{R_i})^2$  ,where $R_{i \in J}$ is the amount of region

​	tree classification: 

- Classification error rate $E=1-\max_{k}(\hat{p}_{mk})$
- Gini Index: $G=\sum_{k=1}^K\hat{p}_{mk}(1-\hat{p}_{mk})$
- cross-entropy: $D=-\sum_{k=1}^K\hat{p}_{mk}\log {\hat{p}}_{mk} $

#### 原理3：Tree Pruning

The process described above may produce good predictions on the training set, but is likely to overfit the data, leading to poor test set performance.

**How do we determine the best way to prune the tree? **

Intuitively, our goal is to select a subtree that leads to the lowest test error rate.

Use $K$-fold cross-validation to choose $\alpha$. That is, divide the training observations into $K$ folds. For each $k = 1, \cdots, K$:

### link to Ensembling method

##### 简介：

1. 由于decision tree 的variance会比较大，bias小，所以可以用ensemble的方法;

   Ways to ensemble(这些方法的出现，原本是为了来帮助decision tree，让一个本来variance很高，但bias比较低的进行一些转变)



decision tree——1）boosting&Adaboosting（代表）；2）bagging&random forest（bagging的变形）

1）**boosting**：每次增加sample，然后调整权重，调整背后的distribution，当然是降低bias

​	boosting里面有adaboosting&GBDT&XGboosting

​	boost主要是increasing your mistakes weight（Adaboost）



2）**bagging**：其实就是bootstrap，用已有的sample，多次重复，当然是降低variance

​	random forest：随机属性？？？？



#####  bagging的原理：

 1. 就是bootstrap， 相当于你对现有的sample 再次sample，也就是resample，记得我们学高级统计时候，证明过，这样可以降低variance，同时其预测的插值还符合正态分布

 2. 采用的方式就是 $\hat{f}_{avg}=\frac{1}{B}\sum_{b=1}^B {\hat{f}_b}(x)$ 来降低variance，但实际情况不可能所以，In this approach we generate $B$ different bootstrapped training data sets. We then train our method on the $b$-th bootstrapped training set in order to get $\hat{f}^{\ast b}(x)$, and finally average all the predictions, to obtain

    ​	 
    $$
    \hat{f}_{bag}(x)=\frac{1}{B}\sum_{b=1}^B \hat{f}^{\ast b}(x)
    $$


    主要算法**random forest**:

3. Random forests provide an improvement over bagged trees by way of a small tweak that decorrelates the trees.a random sample of
   *$m$ predictors is chosen as split candidates from the full set of p predictors.*
   *The split is allowed to use only one of those m predictors.*

4. A fresh sample of $m$ predictors is taken at each split, and typically we choose $m \approx \sqrt{p} $—thatis , the number of predictors considered at each split is approximately equal to the square root of the total number of predictors

5. **The main difference** between bagging and random forests is the choice
   of predictor subset size $m$.

6. We can think of this process as **decorrelating** the trees, thereby making the average of the resulting trees less variable and hence more reliable.

7. averaging many highly correlated quantities does not lead to as large of a reduction in variance as averaging many uncorrelated quantities.
   **感想：说白了就是你用了少的sample自然reduce variance? 相当于RF就是控制m 你重复的此书**

8. out of bag estimate?



##### Boosting：

1. bagging involves creating multiple copies of the original training data set using the bootstrap, fitting a separate decision tree to eachcopy, and then combining all of the trees in order to create a single predictive model.(bagging是用再抽样法创造多个副本，在对每个副本建立决策树，然后将这些数结合在一起来建立一个预测模型)

2. 每一棵树都是建立在一次bootstrap上，与其他树独立；boosting也是类似，但是这里的树采用**sequentially** 顺序生成：

   - each tree is grown using information from previously grown trees. 
   - Boosting does not involve bootstrap sampling; instead each tree is fit on a modified version of the original data set.

   感想1:相当于改变本身的树，而不是sample 树本身&这是弱学习转化成为强学习;

   感想2：给定一份训练数据集（各样本权重是一样的，之后会有变化），然后进行M次迭代，每次迭代后，**对分类错误的样本加大权重,对正确分类的样本减少权重**，在下一次的迭代中更加关注错分的样本。

3. Boosting has three tuning parameters:

   - The number of trees B:Unlike bagging and random forests, boosting can overfit if B is too large, although this overfitting tends to occur slowly if at all. We use cross-validation to select $B$.
   - The shrinkage parameter $\lambda$: Typical values are $0.01$ or $0.001$, and the right choice can depend on the problem. Very small $\lambda$ can require using a very large value of B in order to achieve good performance.
   - The number $d$ of splits in each tree, which controls the complexity of the boosted ensemble. Often $d = 1$ works well, in which case each tree is a stump

##### 		主要算法 adaboosting&XGBoost&Gradient Boosting

Boosting 提升法.一种将多个弱分类器通过组合提升为强分类器的思想.
它实现的关键在于：在每轮迭代训练中，通过改变样本权重的方式改变样本分布，从而在下一轮对误分或者偏差大的样本进行近似局部的拟合(类似于加权回归中的加权,这更容易理解)，最后组合起来，达到提升的目的.
这里会有几个问题：

1. 每轮训练偏差大小的标准是什么？(与损失函数有关)
2. 弱分类器怎么组合？(损失函数 对 模型权重 求偏导)
3. 样本权重怎样调整？

###### Details:

##### Adaboosting:

作者：Evan
链接：https://www.zhihu.com/question/54332085/answer/296456299
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。



1，初始化训练数据的权值分布

​                          D1= (w11, ..., w1i, ..., w1N)，其中w1i= 1/N，i=1, 2, ..., N

2，对m=1, 2,..., M

​                          a，使用具有权值分布Dm的训练数据集学习，得到基本分类器

​                                   Gm(x): X -> {-1, +1}

​                          b，计算Gm(x)在训练数据集上的分类误差率

![img](https://pic1.zhimg.com/50/v2-a206e78b49a0cee7613197c81c10acbb_hd.jpg?source=1940ef5c)

​                          c，计算Gm(x)的系数

![img](https://pic1.zhimg.com/50/v2-b7f85a0220fdd4a738e4017e83f9b9ff_hd.jpg?source=1940ef5c)

​                          这里的对数是自然对数。

​                          d，更新训练数据集的权值分布

![img](https://pic2.zhimg.com/50/v2-a29b4527144bcc3b56693c92ffecb840_hd.jpg?source=1940ef5c)                          这里，Zm是规范化因子

![img](https://pic2.zhimg.com/50/v2-66039826b15f639e3a581299674ee361_hd.jpg?source=1940ef5c)

​                          它使$D_{m+1}$成为一个概率分布。

3，构建基本分类器的线性组合

![img](https://pic4.zhimg.com/50/v2-c504b60eb2e61b28db49a3ec1d2d44e3_hd.jpg?source=1940ef5c)

​                 得到最终分类器

![img](https://pic1.zhimg.com/50/v2-80257279d4695ac4e6209eaa12e0198e_hd.jpg?source=1940ef5c)

1. ADboosting的损失函数是指数函数:

   $E=\sum_{i=1}^n e^{-y_i[f_{m-1}(x_i)+\alpha_{m,i}h_m(x_i)]}$

2. 通过对损失函数的分析我们能找到每一轮的训练目标:

   $h_m=\arg \min_{h_m}\sum_{i=1}^n\omega_i^m I(y_i\neq h_m(x_i))$

3. 损失函数对模型权重求偏导可得到模型权重的具体表达

   $\alpha_m=\frac{1}{2}\log(\frac{1-err_m}{err_m})$ 				where	 $err_m=sum_{i=1}^n\omega_i^{M}I(y_i\neq h_m(x_i))$

4. 样本权重的更新由构造过程决定:

   $\omega^m=\frac{\omega^{m-1}e^{-\alpha_m y_i h_m(x_i)}}{Z_m}$

##### Gradient Boosting

1. AdaBoosting的推广，当损失函数是平方损失的时候会怎么样

2. Friedman对Gradient Boosting的定义

看到这里大家可能会想，每一轮中样本怎么改变呢？

###### LSBoost (Least Square Boosting):

AdaBoosting的损失函数是指数损失，而当损失函数是平方损失时，会是什么样的呢？损失函数是平方损失时，有:
$E=\sum_{i=1}^n(y_i-[f_{m-1}(x_i)+\alpha_{m,i}h_m(x_i)])^2$
括号换一下：
$E=\sum_{i=1}^n([y_i-[f_{m-1}(x_i)]-\alpha_{m,i}h_m(x_i)])^2$
中括号里就是上一轮的训练残差！要使损失函数最小，就要使当轮预测尽可能接近上一轮残差。因此每一轮的训练目标就是拟合上一轮的残差！而且我们可以发现，残差恰好就是平方损失函数对于f(x)的负梯度.这直接启发了Friedman提出Gradient Boosting的总体框架

###### Gradient Boosting 的定义：

Friedman提出了直接让下一轮训练去拟合损失函数的负梯度的想法.当损失函数是平方损失时，负梯度就是残差(LSBoosting);不是平方损失函数时，负梯度是残差的近似.从而Gradient Boosting诞生了.其框架如下： 步骤5中，$\rho$可用线性搜索(line search)的方式得到，可理解为步长. 显然，LSBoosting是Gradient Boosting框架下的特例



###### L2Boosting

L2Boosting是LSBoosting的特例，它对各模型权重(步长)取的是1，样本权重也是1.这在Buhlmann P, Yu Bin的文章中有详细说明[PDF](http://www.stat.math.ethz.ch/Manuscripts/buhlmann/boosting.rev5.pdf).
这意味这只需要用新模型拟合残差，然后不经压缩地加入总体模型就好了…Friedman对其评价是”L2Boosting is thus nothing else than repeated least squares fitting of residuals”.明晃晃的不屑有没有…

###### 其他Gradient Boosting 

可以看到，在Gradient Boosting框架下，随着损失函数的改变，会有不同的Boosting Machine出现.

##### xgboost:

就是从损失函数的角度提出的，它在损失函数里加入了正则惩罚项，同时认为单单求偏导还不够.因为求偏导实际是一阶泰勒展开,属于一阶优化，收敛速度还不够快.他提出了损失函数二阶泰勒展开式的想法.



















#### **Reference**:

1. 机器学习西瓜书

2. an introduction to statstical learning

https://zhuanlan.zhihu.com/p/57324157

3. Andrew Ng: https://www.youtube.com/watch?v=wr9gUr-eWdA&list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU&index=10

4. https://liangyaorong.github.io/blog/2017/%E6%B7%B1%E5%85%A5%E7%90%86%E8%A7%A3Boosting/

   https://www.zybuluo.com/frank-shaw/note/127048

5. 很好的调参模型

   https://zhuanlan.zhihu.com/p/103136609

   所以决策树算法的核心是要解决两个问题：

   **1）如何从数据集中找出最佳分枝？**

   **2）如何在合适的时候让决策树停止生长，防止过拟合？**

   几乎所有与决策树有关的模型调整方法，都逃不开这两个核心问题。

   **关于第一个问题，一个数据集必然给出了很多的特征，先按照哪个特征来高效地划分数据集，**是决策树需要解决的最大问题。当然，我们完全可以按照特征从头到尾顺次来划分数据集，但是，这样做其实并不好，因为你如果一不小心把重要的特征排到了后边，那么你的决策树会很低效。

   于是，机器学习专家们就去信息学科中偷来了一个词语：信息熵。信息越确定（单一），信息熵越小；信息越多变（混乱），信息熵越大。

   通过比较拆分前后的信息熵之差找出更重要的特征的方法，就产生了ID3和C4.5两种决策树算法。

   之后，科学家又找到了一个更好的衡量方法——基尼系数。于是就产生了CART决策树。

   ### **四、附录：ID3、C4.5和CART决策树的比较**

   ID3决策树算是决策树的鼻祖，它采用了信息增益来作为节点划分标准，但是它有一个缺点：在相同条件下，取值比较多的特征比取值少的特征信息增益更大，导致决策树偏向于选择数量比较多的特征。所以，C4.5在ID3的基础上做了改进，采用了信息增益率来解决这个问题，而且，C4.5采用二分法来处理连续值的特征。以上两个决策树都只能处理分类问题。

   决策树的集大成者是CART决策树。

   很多文章里提到CART决策树和ID3、C4.5的区别时，都简单地归结为**两者使用的节点划分标准不同（用基尼系数代替信息熵）**。其实除此之外，CART和以上两种决策树最本质的区别在于两点：

   1、CART决策树对树的形状做了规定，只能二叉树，同时规定，内部结点特征的取值左为是，右为否。这个二叉树的规定，使得对于同样的数据集，CART决策树的深度更深。

   2、从它的英文名：classification and regression就可以看出，CART决策树不仅可以处理分类问题，而且还可以处理回归问题。在处理回归问题时，回归cart树可以用MSE等多种评价指标。

   3、值得一提的是，随机森林、GBDT、XGBOOST算法都是基于CART决策树来的。

   **三种树的对比做一个总结：**

   1. ID3：倾向于选择水平数量较多的变量，可能导致训练得到一个庞大且深度浅的树；另外输入变量必须是分类变量（连续变量必须离散化）；最后无法处理空值。
   2. C4.5在ID3的基础上选择了信息增益率替代信息增益，同时，采用二分法来处理连续值的特征，但是生成树浅的问题还是存在，且只能处理分类问题。
   3. CART以基尼系数替代熵，划分规则是最小化不纯度而不是最大化信息增益（率）。同时只允许生成二叉树，增加树的深度，而且可以处理连续特征和回归问题。
   4. CART决策树是根据ID3等决策树改进来的，scikit-learn实现的决策树更像是CRAT决策树。但是，节点的划分标准也可以选择采用熵来划分。

   ### Random forest

   为了解决这个两难困境，聪明的专家们想出了这样的思路：既然我增加单棵树的深度会适得其反，**那不如我不追求一个树有多高的精确度，而是训练多棵这样的树来一块预测**，一棵树的力量再大，也是有限的，当他们聚成一个集体，它的力量可能是难以想象的，也就是我们常说的：“三个臭皮匠赛过诸葛亮”。这便是集成学习的思想。

   这里多提一句，正是因为每棵树都能够用比较简单的方法细致地拟合样本，我们可以多用几棵树来搭建准确率更高的算法，后边要说到的一些工业级的算法，比如GBDT、XGBOOST、LGBM都是以决策树为积木搭建出来的。

   随机森林的算法实现思路非常简单，只需要记住一句口诀：**抽等量样本，选几个特征，构建多棵树。**

   调参所需：

   https://zhuanlan.zhihu.com/p/139510947

   1. max_features（最大特征数）

   2. n_estimators：随机森林生成树的个数，默认为100。
   3. bootstrap：每次构建树是不是采用有放回样本的方式(bootstrap samples)抽取数据集。可选参数：True和False，默认为True。
   4. oob_score：是否使用袋外数据来评估模型，默认为False。

   boostrap和 oob_score两个参数一般要配合使用。如果boostrap是False，那么每次训练时都用整个数据集训练，如果boostrap是True，那么就会产生袋外数据。

   

   介绍完了这些参数，接下来就要介绍随机森林的调参顺序了，随机森林的调参顺序一般遵循**先重要后次要、先粗放后精细**的原则，即先确定需要多少棵树参与建模，再对每棵树做细致的调参，精益求精。





















# 集成学习（Ensemble Method）

集成方法：



基本上常和树模型一起 link tree models



decision tree——1）boosting&Adaboosting（代表）；2）bagging&random forest（bagging的变形）

1）==boosting==：每次增加sample，然后调整权重，调整背后的distribution，当然是降低bias

​	boosting里面有adaboosting&GBDT&XGboosting



2）==bagging==：其实就是bootstrap，用已有的sample，多次重复，当然是降低variance

​	random forest：随机属性？？？？



综合方法：meta algorithm&stacking algorithm



further steps:上课所记载的关于decision tree的（andrew NG的 youtube 课）