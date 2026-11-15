## 工业 CT 数学与入口映射

来源 cg_tikhonov 映射为 cg、regularizer=tikhonov。lam_rel 换算 λ=lam_rel×估计的 ||A||²，求解 (A*WA+λI)x=A*Wy。max_iter 映射 num_iterations，tol 映射 tolerance。来源推荐 max_iter=20、lam_rel=0.001、tol=1e-6。公共停止按本项目正规残差规约，来源停止与质量记录按来源条件解释。带先验中心的深度 CG 使用右端 A*Wy+λx_prior，可微校正由 solvers/deep/cg.py 提供。

几何、来源与限制见 [工业 CT](../ct/industrial.md)，两侧调用见 [Harness](../ct/harness.md)。
