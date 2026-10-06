# 浮基状态与切空间接口

## 本轮范围

提供 Torch SO(3)、SE(3)、浮基状态几何，以及 `nx != ndx` 的 DDP/FDDP 接口。
已实现几何与合成运动学模型的数值验收。Go2 模型现已在此接口上接入 Torch
质量矩阵、关节扭矩驱动、固定足端接触与站立 MPC，见 `go2.md`；碰撞切换仍未实现。

## 坐标约定

`FloatingBaseManifold(nj)` 支持自由浮基与 `nj` 个欧氏标量关节：

```text
state = [world_position(3), quaternion_xyzw(4), joint_positions(nj), velocities(6+nj)]
delta = [body_translation(3), body_rotation(3), joint_delta(nj), velocity_delta(6+nj)]
nq = 7+nj; nv = 6+nj; nx = 13+2nj; ndx = 12+2nj
```

四元数必须为单位四元数（平方模误差 <1e-5）；`q` 与 `-q` 表示同一姿态。
`neutral(like)` 根据 `like` 的前导维度、device、dtype 返回有效单位状态。
不能将全零 tensor 当作有效浮基状态。求解器将无效状态环境标记失败，并用 neutral
状态保护其他环境的计算；异常环境返回 NaN 轨迹，控制器保留上次有效动作。

- `SO3Manifold` 使用右侧指数映射 `q_new = q * Exp(delta_rotation)`。
- `SE3Manifold` 使用 `T_new = T * Exp(delta_pose)`；平移为
  `p_new = p + R * J_left(delta_rotation) * delta_translation`，包含旋转和平移耦合。
- `diff(target, base) = Log(base^-1 * target)`；**参数顺序与 Pinocchio/Crocoddyl 的
  `difference(base,target)` / `diff(base,target)` 相反**。
- 广义速度像 Crocoddyl `StateMultibody` 一样直接加减；没有随配置增量另做速度搬运。
- 旋转对数采用主值，角度为 pi 时存在切割与不可微边界。不要跨该边界解释局部导数。
- 球关节、连续关节的非标量表示需要单独配置流形，不能直接用 `nj` 替代。

## 导数契约

非欧氏 dynamics/cost 必须显式声明 `ndx`，并与传入 `manifold.ndx` 一致。
旧欧氏模型可省略 `ndx`，继续使用 `ndx=nx`。

| 字段 | 形状 | 所在坐标 |
| --- | --- | --- |
| `x`, `result.xs` | `[B,nx]`, `[B,T+1,nx]` | 存储表示 |
| `Fx` | `[B,ndx,ndx]` | 输入 x 与输出 f(x,u) 的局部增量 |
| `Fu` | `[B,ndx,nu]` | 输出 f(x,u) 的局部增量 |
| `lx`, `lxx`, `lxu` | `[B,ndx]`, `[B,ndx,ndx]`, `[B,ndx,nu]` | 当前 x 的局部增量 |
| `K`, `gap` | `[B,T,nu,ndx]`, `[B,T,ndx]` | 反馈局部误差、下一名义状态的差分坐标 |

动力学导数定义为在零点求导：

```text
diff(f(integrate(x,dx), u+du), f(x,u))
```

`tangent_dynamics_jacobians` 使用 Torch forward-mode JVP 生成这一局部导数，
支持逐环境参数，但要求模型不耦合 batch 元素。它是正确性基线，会沿 `ndx+nu`
个方向计算，不是已经优化的机器人动力学导数内核。

`ManifoldQuadraticCost` 用 `r=diff(x,reference)` 定义状态代价 `0.5*r.T*Q*r`，
`Q/Q_terminal` 的维度为 `ndx`；`R` 的维度为 `nu`。参考可以共享或逐环境。
梯度为 `J.T*Q*r`，状态 Hessian 使用 `J.T*Q*J` 的 Gauss-Newton 近似，
没有包括 residual 曲率项，也没有 dynamics 二阶导数。

## FDDP 非零 gap 的坐标处理

动力学导数位于 `predicted=f(x[t],u[t])`，下一价值函数位于 `nominal=x[t+1]`。
先计算：

```text
g = diff(predicted, nominal)
J = d diff(integrate(predicted,delta), nominal) / d delta at 0
Fx_chart = J * Fx; Fu_chart = J * Fu
Vx_gap = Vx + Vxx * g
```

再用 `Fx_chart/Fu_chart` 做 modified Riccati sweep。只删除维度检查或直接复用不同
切空间的导数会漏掉 `J`。欧氏/SO(2) 局部坐标的 `J=I`，保留原有热路径。
SO(3) 使用右对数 Jacobian；SE(3) 使用右 Jacobian 的 24 项级数及设备上的 `inv_ex`。
浮基和乘积流形逐块组装该变换。

forward pass 延续 `integrate(predicted_candidate,(alpha-1)*g_nominal)`；本轮支持的
Lie 群使用统一右平凡化坐标，`alpha=0` 恢复名义轨迹、`alpha=1` 闭合 gap。
这不意味着不同预测状态的原始存储向量可以直接相减。自定义非 Lie 群重traction
还需要核对该 gap 修正的含义，不能仅凭 `nx != ndx` 推断 FDDP 已适用。

## 验证与运行

```bash
PYTHONPATH=src python examples/floating_base_state.py --joints 12
python -m pytest tests/test_manifolds.py tests/test_tangent_ddp.py -q
```

本机 Crocoddyl/Pinocchio 测试需要此前准备的 NumPy 路径：

```bash
PYTHONPATH=.dev-tools/croco-numpy126:src \
  /Users/zhengyuanhao/anaconda3/envs/croco_env/bin/python \
  -m pytest tests/test_manifolds.py tests/test_tangent_ddp.py -q
```

验收覆盖 identity/小角度/接近 pi、四元数正负等价、float32/float64、局部有限差分，
以及 0/2 关节浮基对 Pinocchio integrate 和 Crocoddyl StateMultibody integrate/diff/Jdiff。
SO(3) DDP/FDDP 验证状态维度 4、切空间维度 3、批量隔离和不可行状态初值。
非零姿态 gap 的首轮反馈增益由独立有限差分的 Q 函数计算核对。
浮基合成模型验证 warm start、reset、无效四元数隔离及初始化辅助函数。
CUDA 用例在无 CUDA 的本机明确跳过，尚未证明新接口的 GPU 数值和性能。
