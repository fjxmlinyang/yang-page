---
title: 'notes-General Linear Regression'
date: 2021-01-01
permalink: /posts/2021/01/notes-General Linear Regression 2/
tags:
  - Models
  - Machine Learning
  - Optimization
---





Continue





# SVM

1.refer to your publications

$E((w^Tx+b,0)_{\max})$?

2. Maximum margin classifier

   $\max \frac{1}{\|w\|},\qquad s.t. y_i(w^Tx_i+b)\geq 0,\qquad i=1,\cdots,n$

3. $\min \frac{1}{2}\|w\|^2,\qquad s.t. y_i(w^Tx_i+b)\geq 1,\qquad i=1,\cdots,n$

   Dual  $\mathcal{L}(w,b,a)=\frac{1}{2}\|w\|^2-\sum_{i=1}^n \alpha_i(y_i(w^Tx_i+b)-1)$ 变形后

   $\mathcal{L}(w,b,a)=\sum_{i=1}^n\alpha_i-\frac{1}{2}\alpha_i\alpha_jy_i y_jx_i^T x_j$ And $w=\sum_{i=1}^n\alpha_iy_ix_i$

   dual problem 

4. 换成核估计

   $f(x)=\sum_{i=1}^Nw_i\phi_i(x)+b$ 转换成为 $f(x)=\sum_{i=1}^l\alpha_i y_i \langle \phi(x_i), \phi(x) \rangle+b$

$\alpha$ 可以由dual 来求

$\max_{\alpha}\sum_{i=1}^n \alpha_i-\frac{1}{2}\sum_{i,j=1}^n\alpha_i\alpha_jy_iy_j\langle \phi(x_i)\phi(x_j)\rangle \qquad s.t. \alpha_i \geq 0,i=1,\cdots, n; \sum_{i=1}^n\alpha_iy_i=0$ 

$\max_{\alpha}\sum_{i=1}^n \alpha_i-\frac{1}{2}\sum_{i,j=1}^n\alpha_i\alpha_jy_iy_jx_ix_j \qquad s.t. \alpha_i \geq 0,i=1,\cdots, n; \sum_{i=1}^n\alpha_iy_i=0$ 





