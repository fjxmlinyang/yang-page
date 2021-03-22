---
title: 'notes-Tree based methods supplement'
date: 2021-01-05
permalink: /posts/2021/01/notes-Tree based methods Supplement/
tags:
  - Models
  - Machine Learning
  - Optimization
---

Here are the supplement of tree models









# Supplement-Boosting

## Adaboosting:

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

## Gradient Boosting

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



## L2Boosting

L2Boosting是LSBoosting的特例，它对各模型权重(步长)取的是1，样本权重也是1.这在Buhlmann P, Yu Bin的文章中有详细说明[PDF](http://www.stat.math.ethz.ch/Manuscripts/buhlmann/boosting.rev5.pdf).
这意味这只需要用新模型拟合残差，然后不经压缩地加入总体模型就好了…Friedman对其评价是”L2Boosting is thus nothing else than repeated least squares fitting of residuals”.明晃晃的不屑有没有…

###### 其他Gradient Boosting 

可以看到，在Gradient Boosting框架下，随着损失函数的改变，会有不同的Boosting Machine出现.

## xgboost:

就是从损失函数的角度提出的，它在损失函数里加入了正则惩罚项，同时认为单单求偏导还不够.因为求偏导实际是一阶泰勒展开,属于一阶优化，收敛速度还不够快.他提出了损失函数二阶泰勒展开式的想法.







# Supplement-Bagging

## Random Forest

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





#### **Reference**:

1. 机器学习西瓜书

2. an introduction to statstical learning

https://zhuanlan.zhihu.com/p/57324157

3. Andrew Ng: https://www.youtube.com/watch?v=wr9gUr-eWdA&list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU&index=10

4. https://liangyaorong.github.io/blog/2017/%E6%B7%B1%E5%85%A5%E7%90%86%E8%A7%A3Boosting/

   https://www.zybuluo.com/frank-shaw/note/127048

5. 很好的调参模型

   https://zhuanlan.zhihu.com/p/103136609



# 附录：ID3、C4.5和CART决策树的比较

- ID3决策树算是决策树的鼻祖，它采用了信息增益来作为节点划分标准，但是它有一个缺点：在相同条件下，取值比较多的特征比取值少的特征信息增益更大，导致决策树偏向于选择数量比较多的特征。所以，C4.5在ID3的基础上做了改进，采用了信息增益率来解决这个问题，而且，C4.5采用二分法来处理连续值的特征。以上两个决策树都只能处理分类问题。

- 决策树的集大成者是CART决策树。

很多文章里提到CART决策树和ID3、C4.5的区别时，都简单地归结为**两者使用的节点划分标准不同（用基尼系数代替信息熵）**。其实除此之外，CART和以上两种决策树**最本质的区别在于两点**：

1、CART决策树对树的形状做了规定，只能二叉树，同时规定，内部结点特征的取值左为是，右为否。这个二叉树的规定，使得对于同样的数据集，CART决策树的深度更深。

2、从它的英文名：classification and regression就可以看出，CART决策树不仅可以处理分类问题，而且还可以处理回归问题。在处理回归问题时，回归cart树可以用MSE等多种评价指标。

3、值得一提的是，随机森林、GBDT、XGBOOST算法都是基于CART决策树来的。

**三种树的对比做一个总结：**

1. ID3：倾向于选择水平数量较多的变量，可能导致训练得到一个庞大且深度浅的树；另外输入变量必须是分类变量（连续变量必须离散化）；最后无法处理空值。

2. C4.5在ID3的基础上选择了信息增益率替代信息增益，同时，采用二分法来处理连续值的特征，但是生成树浅的问题还是存在，且只能处理分类问题。

3. CART以基尼系数替代熵，划分规则是最小化不纯度而不是最大化信息增益（率）。同时只允许生成二叉树，增加树的深度，而且可以处理连续特征和回归问题。

4. CART决策树是根据ID3等决策树改进来的，scikit-learn实现的决策树更像是CRAT决策树。但是，节点的划分标准也可以选择采用熵来划分。

   

   


















