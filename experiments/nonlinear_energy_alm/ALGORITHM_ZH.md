# nonlinear_energy_alm 主要算法说明

求解带边界和界面约束的非线性能量极小化问题。连续形式可以写成

$$
\min_u E(u)
= \int_a^b \left(\frac12 |u'(x)|^2 + \frac14 u(x)^4 - f(x)u(x)\right)\,dx,
$$
每个单元有两个自由度，在单元 \(K_e\) 上近似解写成

$$
u_h(x)
= u_p^{(e)}(x)
+ z_{e,1}\phi_1^{(e)}(x)
+ z_{e,2}\phi_2^{(e)}(x).
$$

## 外层：Newton-space 更新

对非线性方程

$$
-u'' + u^3 = f
$$

在上一轮近似 \(u^k\) 附近做 Newton 型线性化，得到当前轮用于构造 TFPM 试空间的线性问题

$$
-u_{k+1}'' + 3(u^k)^2 u_{k+1}
= f + 2(u^k)^3.
$$

因此每一轮外层迭代会构造

$$
c_k(x)=3(u^k(x))^2,\qquad
f_k(x)=f(x)+2(u^k(x))^3,
$$

然后用已有的 TFPM 单元构造器生成当前固定解空间和约束。

## 内层：固定解空间上的增广拉格朗日法

在当前固定解空间中，算法直接最小化原始非线性能量，同时满足线性约束。
增广拉格朗日函数为

$$
\mathcal L_\rho(z,\lambda)
= E_h(z)
+ \lambda^T(Cz-d)
+ \frac{\rho}{2}\|Cz-d\|_2^2.
$$

固定 \(\lambda\) 和 \(\rho\) 后，内层用 Newton 法近似求解

$$
\nabla_z \mathcal L_\rho(z,\lambda)=0.
$$

对应的 Newton 线性系统是

$$
\left(\nabla^2 E_h(z)+\rho C^T C\right)\Delta z
=-
\left[
\nabla E_h(z)+C^T\left(\lambda+\rho(Cz-d)\right)
\right].
$$

求得方向 \(\Delta z\) 后，代码用 Armijo 回溯线搜索选择步长 \(\alpha\)，更新

$$
z \leftarrow z+\alpha\Delta z.
$$

如果线性系统奇异，代码退回到最小二乘求解；如果方向不是下降方向，则退回到负梯度方向。

## 乘子更新与收敛判据

固定空间内每完成一次增广拉格朗日迭代，乘子按

$$
\lambda \leftarrow \lambda+\rho(Cz-d)
$$

更新。