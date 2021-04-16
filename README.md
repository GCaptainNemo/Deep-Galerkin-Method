# Deep-Galerkin-Method
神经网络求解偏微分方程的算法，称为[Deep Galerkin算法](docs/introduction.ipynb)。主要思想就是利用神经网络良好的逼近能力，通过优化逼近偏微分方程的解，
且巧妙地将偏微分方程作为神经网络的正则化项引入，在数学、计算物理方向掀起了一轮研究热潮。

目前关于这种算法的理论研究还比较缺乏，它的逼近效果如何、能否收敛到偏微分方程的解，偏微分方程适定(well-posed)与否对其的影响都还是未解答的问题。
目前该算法在一些偏微分方程求解问题上表现出比传统数值方法更快捷、准确的结果。

## 参考资料
[1] Mr A , Pp B , Gek A . Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations[J]. Journal of Computational Physics, 2019, 378:686-707.


