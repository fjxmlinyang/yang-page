---

title: 'notes-What is basic statistics'
date: 2021-03-20
permalink: /posts/2021/03/notes-What is statistics?/
tags:
  - Statistics
  - Machine Learning
---

Only some scratch

Question:

1. What is a p-value?

2. What is a confidence interval?

3. What is type I error?

4. What is the CLT? What is an outliers? 

   How can outliers be determined in a dataset?/Z-score/Standard deviations



Data challenge:

1. Procedure: how do you compute p-value?
2. How do you do a two-group t test?
3. Logic: Why do you compare p-value with 0.05?





# 1 What is statistics?

- Statistics presents a rigorous scientific method for gaining insight into data.
- For example, suppose **we measure** the weight of 100 patients in a study. With so many measurements, simply looking at the data fails to provide an informative account. 
- However, statistics can give an instant **overall picture** of data based on graphical presentation or numerical summarization irrespective to the number of data points. 
- Besides data summarization, another important task of statistics is to **make inference and predict** relations of variables.



**描述性统计descriptive statistics**：整理，描述搜集的数据特征

- 比如你找mean，median等等

**推论性统计inferential statistics**：利用小群体的数据推论大群体的数据的特征，样本和总体的关系（sample&population）

Difference with probability：见后面

- 概率是推理，统计是推论

**Difference with ML**:

- Statistical modeling is a formalization of relationships between variables in the data in the form of mathematical equations
- Machine Learning is an algorithm that **can learn from data without replying on rules-based programming.**(说白了就是蒙特卡洛。。)



# 2 Population and Sample（在意什么呢？）

- **Population:** **The entire collection of individuals or measurements about which information is desired** e.g. Average height of 5-year old children in the USA
- **Sample:** **A subset of the population selected** for study. Primary objective is to create a subset of population whose center, spread and shape are as close as that of population. 
  - There are many methods of sampling. Random sampling, stratified sampling, systematic sampling, cluster sampling, multistage sampling, area sampling, etc
  - Random Sample: A simple random sample of size n from a population is a subset of n elements from that population where the subset is chosen in such a way that every possible unit of population has the same chance of being selected.
- **Example:** Consider a population of 5 numbers(1,2,3,4,5). How many random samples (without replacement) of size 2 can we draw from this population?($C_2^5$)
- **Population mean** if the five numbers is 3. Averages of 10 samples of sizes 2 are 1.5, 2, 2.5, 3, 3.5, 4, 4.5. Mean of these 10 averages (1.5+2+2.5+3+2.5+3+3.5+3.5+4+4.5)/10 =3 which is the same as the population mean.





# 3 Parameter and Statistics（用什么描述呢？）

- **Parameter:** Any statistical characteristic of a population. Population mean, population median, population standard deviation are examples of parameters.
- **Statistic(estimator):** **Any statistical characteristic of a sample(面试题)**. Sample mean, sample median, sample standard deviation are some example s of statistics.
  - How to estimate the mean parameter in a normal distribution?
    - sample mean/median
- **Statistical Issue**: Describing population through census or making inference from sample by estimating the value of the parameter using statistics.
  - Why is hypothesis testing important?



## 3.1 Statistical Inference

**Statistical Inference:** We sample the population(in a manner to ensure that the sample correctly represents the population) and then take measurements on our sample and infer(or generalize) back to the population.

<img src="./Screenshot from 2021-03-28 21-51-22.png" alt="Screenshot from 2021-03-28 21-51-22" style="zoom:67%;" />

















# 4. A Taxonomy of statistics（怎么&用什么方法描述呢？）

<img src="./Screenshot from 2021-03-28 21-54-21.png" alt="Screenshot from 2021-03-28 21-54-21" style="zoom:67%;" />

**面试题**

- Statistics describes a numerical set of data by its Shape, Center, Variability.
- Statistics describes a categorical set of data by Frequency percentage or proportion of each category

**Variable** :any characteristic of an individual or entity. A variable can take different values for different individuals. Variables can be categorical or quantitative.

- Categorical variable:
  - Nominal-- Categorical variables with no inherent order or ranking sequence such as names or classes(e.g. gender). Value may be a numerical, but without numerical value(e.g. I, II, III). The only operation that can be applied to Nominal variables is enumeration.
  - Ordinal-- Variables with an inherent rank or order, e.g. mild, moderate, severe. Can be compared for equality, or greater or less, but not how much greater or less.
- Continuous: variables with all properties of interval plus an absolute, non-arbitrary zero point,.e.g.age, weight, temperature(Kelvin). Addition, subtraction, multiplication, and division are all meaningful operations.



**Distribution**:(这个variable到底什么情况)(of a variable) tells us what values the variable takes and how often it takes these values.

- Unimodal(having a single peak)
- Bimodal(having two distinct peaks)
- Symmetric(left and right half are mirror images)

**Numerical presentation**: 

- Methods of Center Measurement: mean & median
  - mean: 
  - median:不太连续会跳跃
- Methods of variability measurement: range, variance, standard deviation

**How to characterize distribution:**

- Probability density function: $f(x)= F'(x)$
- Cumulative distribution function: $F(X)= Pr(X<x)$







# 5. Statistical description of data

**一个变量？**

**Mean and Variance**:



<img src="./Screenshot from 2021-03-28 22-29-36.png" alt="Screenshot from 2021-03-28 22-29-36" style="zoom:67%;" />





其中统计的样本均值是个random variable

无偏估计的意义是：在多次重复下，他们的平均值接近所估计的参数真值

example:

- bernoulli: $E(X)= p, Var(X)= pq$
- uniform: $E(X)= \frac{a+b}{2}, Var(X)= \frac{(b-a)^2}{12}$
- binomial: $E(X)= np, Var(X)= npq$
- Possion Distribution: $E(X)= {\lambda}, Var(X)= {\lambda}$, where $\lambda$  表示时间t内时间发生的平均次数





**如果有两个变量呢？**

**Correlation:** 两个或多个具备相关性的变量元素进行分析，从而衡量两个变量因素的相关密切程度

**Pearson Correlation Coefficient：** 
$$
\rho_{X,Y}= \frac{cov(X,Y)}{\sigma_X \sigma_Y}
$$
where $cov(X,Y)= E[(X-\mu_X)(Y-\mu_Y)]$
$$
r_{xy} =\frac{\sum_{i=1}^n(x_i-\bar{x})(y_i-\bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i-\bar{x})^2}\sqrt{\sum_{i=1}^{n}(y_i-\bar{y})^2}}
$$


Theorem:

$E（X+Y）=E（X）+E（Y）$ 

$E(XY)= E(X)E(Y)$ when $X$ and $Y$ are independence

$Cov(X,Y)= E(XY)-E(X)E(Y)$

$Cov(X,Y)=0$ when $X$ is independent of $Y$

$\rho(X,Y)= 0$ when $X$ is independence of $Y$



Question: 

$X$: correlation=0 --> independent?

Counter-example: $Z \sim N(0,1)$, $Cor(Z, Z^2)=E(ZZ^2)-E(Z)E(Z^2)=0$, $Z$ is not independent of $Z^2$.





# 6. Normal distribution

#### 6.1 good properties:

1. $X,Y$ are normal, $X+Y$ is normal
   1. $X, Y$ i.i.d Uniform $[0,1]$, what is the distribution of $X+Y$
      1. Hint: $X,Y$ are the random variable from two dices, what is the distribution of $X+Y$
2. $Cor(X,Y)= 0$ and $X,Y$ are normal then $X, Y$ are independent
3. CLT
4. Simplicity

#### 6.2 The 68-95-99.7  Rule

​	68%/95%/99.7% observations fall within  $\sigma/2\sigma/3\sigma$ of the mean $\mu$

#### 6.3 Standardizing and z-scores

$$
z= \frac{\bar{x}-\mu}{\sigma / \sqrt{n}}
$$













Reference:

1. xiaohu's lecture
2. lai's lecture











