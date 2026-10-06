# 宇树 Go2：Torch 批量固定接触与站立 MPC

## 模型来源与资产

首个机器人模型选定为宇树 Go2。原始 URDF 来自
[unitreerobotics/unitree_ros](https://github.com/unitreerobotics/unitree_ros/tree/5994d4faef0a9cadd3287f8de0199a67eeb2a259/robots/go2_description)，
固定 commit `5994d4faef0a9cadd3287f8de0199a67eeb2a259`。
URDF 原样保存于 `src/crocoddyl_batched_mpc/assets/go2/go2.urdf`，
同目录保存上游 BSD-3-Clause `LICENSE` 和包含 SHA256 的 `SOURCE.json`。
wheel/sdist 配置包含上述文件，无需运行时联网获取模型。

本轮只加载运动学与惯性模型，不包含可视化 mesh。URDF 中 ROS 的 mesh 路径未修改，
因此不能将此资产目录当作完整 ROS/Isaac Sim 可视化描述包。

## 状态、动作和站立初值

- Pinocchio 使用自由浮基：nq=19、nv=18；对应 nx=37、ndx=36。
- 状态采用上一轮的 `[world xyz, xyzw quaternion, joints, generalized velocities]`。
- 12 维控制是关节扭矩，顺序为 **FL、FR、RL、RR**，每条腿依次 **hip、thigh、calf**。
  浮基没有直接驱动，actuation matrix 的前 6 行为零。
- 足端采用 `FL_foot/FR_foot/RL_foot/RR_foot` frame。
- 位置、速度和 effort 限制来自 URDF；`robot.torque_limits` 只提供元数据，没有隐式裁剪。
- `standing_configuration()` 提供 hip=0、thigh=0.8、calf=-1.6 rad 的弯腿几何初值，
  根据运动学设置高度，使四个足端 frame 原点位于 z=0。
- 本固定 URDF 的总惯性质量为 **16.087 kg**，几何初值浮基高度为 **0.296797 m**。
  这些是本模型的计算结果，不代表所有 Go2 硬件版本的实测参数。
- 点接触使用足端 frame 原点，没有计入脚底碰撞球半径。

接入 Isaac Lab 或 SDK 时应按关节名构造映射，并核对 quaternion 顺序、浮基速度的
坐标系和力矩单位；不假设模拟器或 SDK 的 tensor 索引与上述顺序相同。

## 独立固定接触参考

`Go2FixedContactReference` 是 **CPU NumPy/Pinocchio oracle**，不是 Torch DynamicsModel，
也不执行模拟器或硬件控制。默认四足支撑，也可指定固定足端子集。
每次计算从 Pinocchio 获取惯性矩阵 M、非线性项 h、世界轴足端 Jacobian J 和 drift a0，
独立求解稠密 KKT：

```text
[M  -J.T] [a]   [S*u-h]
[J    0 ] [f] = [ -a0 ]

a0 = Jdot*v + kp*(foot_position-reference_position) + kd*foot_velocity
```

接触力为世界坐标系 `[Fx,Fy,Fz]`，按指定 feet 顺序排列。
约束是双边 3D 点接触，不包含摩擦锥、只推不拉、扭矩约束或接触切换。
不对奇异 KKT 自动增加阻尼；不可解时明确抛出线性代数异常。

`quasi_static_torques()` 通过静态平衡方程生成最小范数初值，并检查代数平衡可解。
它不是带摩擦/力矩不等式的站立控制器；仅在当前测试种子下验证了四足竖直力为正，
且所需扭矩低于 URDF effort 限制。改变姿态或接触集合后不保证满足这些不等式。

`finite_difference_derivatives` 在 Pinocchio 配置切空间与速度增量中做中心差分，返回
加速度和世界接触力的状态/控制导数，供后续解析或 Torch 实现对照。
`crocoddyl_model()` 单独建立相同足端、机身、稳定化参数的
`DifferentialActionModelContactFwdDynamics`。该对照复用 Pinocchio 刚体量，但使用
不同的接触求解路径，不能当作对 URDF 物理参数或 Pinocchio 本身的独立验证。

## 安装与运行

基础包仍只要求 Torch，加载 Go2 参考时才需要 Pinocchio/NumPy；跨库对照另需 Crocoddyl。

```bash
python -m pip install -e '.[dev,robotics,crocoddyl]'
PYTHONPATH=src python examples/go2_contact_reference.py
python -m pytest tests/test_go2.py -q
```

`crocoddyl` extra 固定安装 Crocoddyl 3.2.1 / Pinocchio 4.0.0。
3.2.1 搭配 4.1.0 的 wheel 已在本机复现接触 `calc()` 段错误；Go2 factory 会在
进入原生调用前拒绝该组合并提示重建环境。仅执行 `pip check` 无法检测这一兼容性问题。
本机原有 Crocoddyl 3.0.1 / Pinocchio 3.6.0 的独立环境仍可按下述方式运行。

本机 croco_env 继续使用项目内 NumPy 1.26.4 测试路径：

```bash
PYTHONPATH=.dev-tools/croco-numpy126:src \
  /Users/zhengyuanhao/anaconda3/envs/croco_env/bin/python examples/go2_contact_reference.py
PYTHONPATH=.dev-tools/croco-numpy126:src \
  /Users/zhengyuanhao/anaconda3/envs/croco_env/bin/python -m pytest tests/test_go2.py -q
```

Python 接口：

```python
import numpy as np
from crocoddyl_batched_mpc.models import load_go2, Go2FixedContactReference

robot = load_go2()
reference = Go2FixedContactReference(robot, stabilization=(10.0, 3.0))
q = reference.reference_configuration.copy()
u = reference.quasi_static_torques()
result = reference.calc(q, np.zeros(robot.nv), u)
```

## 当前验收与下一步

专项测试覆盖资产校验、关节/足端顺序、浮基接口对照、静态重量支撑，
四足/两足（含非默认足序）与零/非零稳定化参数的加速度、接触力和局部导数，
以及输入错误、返回结果隔离。

## Torch 批量刚体与接触动力学

`Go2TorchDynamics` 从包内 URDF 一次性解析完整刚体惯性，将固定 link（含 rotor/foot）
合并到运动父体，不依赖 Pinocchio 做初始化或在线计算。模型支持 float32/float64，
所有在线运算留在指定 Torch device；没有 `.cpu()` / NumPy callback 或逐环境 Python 循环。
固定机器人拓扑的 12 关节递推与时间维循环仍由 Python 调度。

- 质量矩阵：逐体局部运动 Jacobian 与完整空间惯性，`M=sum(J_body.T I_body J_body)`。
- 偏置力：Newton–Euler 的速度叉乘、运动加速度和重力，保持 Pinocchio body 速度约定。
- 足端：世界坐标位置、点 Jacobian、classical `Jdot*v`，与前述 KKT 稳定化完全一致。
- 接触解：`torch.linalg.solve_ex`，奇异解标成该环境 NaN，由求解器逐环境失败机制处理。
- 离散积分：先 `v_next=v+dt*a`，再 `q_next=integrate(q,dt*v_next)`；不做事后足端投影。
- 导数：48 个方向的 `vmap(jvp)`，连续加速度/力和离散局部状态导数均可调用。
  自动微分基线没有有限差分近似；它尚不是优化的刚体导数内核。

```python
import torch
from crocoddyl_batched_mpc.models import Go2TorchDynamics

dynamics = Go2TorchDynamics(dt=0.02, dtype=torch.float64)
state = dynamics.standing_state.expand(4, -1).clone()  # [4,37]
torque = dynamics.quasi_static_torques().expand(4, -1)  # [4,12], Nm
contact = dynamics.contact_dynamics(state, torque)
next_state = dynamics.calc(state, torque)
Fx, Fu = dynamics.calc_diff(state, torque)  # [4,36,36], [4,36,12]
da_dx, da_du, df_dx, df_du = dynamics.contact_derivatives(state, torque)
```

默认四足固定接触，稳定化 `(kp,kd)=(100,20)`，默认 `dt=0.01` 秒；状态/动作不能
在调用时自动改变 dtype/device。`feet` 支持足序与支撑子集，整个 batch 共享接触模式。
`reference_configuration=[19]` 或 `[B,19]` 决定世界足端锚点，后者支持每个环境的独立
世界站立位置。环境重置或更换锚点需重新建立模型/problem 并清除 controller 缓存。
模型惯性固定为官方 URDF，本轮没有逐环境质量/摩擦参数随机化。

## 站立 MPC 闭环

`Go2StandingCost` 使用浮基切空间的姿态/位置、关节与速度 residual，以及
`u-u_equilibrium` 控制代价。初始化轨迹使用平衡扭矩，之后由 MPCController 移动
控制序列作为 warm start。`go2_standing_problem` 提供可接入 Torch DDP/FDDP 的问题。

```python
from crocoddyl_batched_mpc import BatchedMPC, MPCController
from crocoddyl_batched_mpc.models import go2_standing_problem

problem = go2_standing_problem(batch_size=2, horizon=8, dt=0.02, max_iterations=1)
controller = MPCController(BatchedMPC(problem, backend="torch"))
state = problem.dynamics.standing_state.expand(2, -1).clone()
action, result = controller.compute(state)
# 更新为下一周期测量状态。无约束可用候选并不自动满足机器人接触不等式。
```

```bash
python examples/go2_standing_mpc.py --batch-size 2 --steps 65
python examples/go2_standing_mpc.py --plant pinocchio
python examples/go2_standing_mpc.py --backend fddp --steps 8
python examples/go2_standing_mpc.py --device cuda
python -m pytest tests/test_go2_torch.py tests/test_go2_standing.py -q -s
```

示例中两个环境的初始姿态/关节/速度扰动不同，第 32 步再注入角速度扰动。
输出恢复后的状态误差、足端位置误差、最大扭矩占 effort 比例、最小竖直力、切向/法向
比值、接触加速度残差与实测墙钟耗时。`--plant pinocchio` 用独立 NumPy KKT 和
Pinocchio integrate 推进 CPU 状态；该模式允许 CPU 数据转换，仅用于验证。
长时闭环验收使用 Torch DDP；FDDP 覆盖平衡、warm start/reset 与异常环境隔离。

这是双边固定点接触下的离线正确性基线，不能作为实际机器人控制器：尚无单边力、
摩擦锥、扭矩 box、关节限位约束求解、接触切换或碰撞检测。测试轨迹中正法向力和
扭矩余量是观测验收条件，不代表优化器对任意扰动保证这些约束。
CUDA 测试已编写，但本机没有 CUDA，仍需实机验收；当前 CPU 延迟不满足实时控制。
下一步优化导数/递推与内存调度，再加入约束和 Isaac Lab plant 联调。
