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
固定机器人拓扑按三层同时递推四腿，时间维循环仍由 Python 调度。

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
以下新增局部控制器已推进 CPU 性能与硬约束；上述完整 DDP/FDDP 仍保留为无约束基线。

## 局部硬约束 MPC 与 50 Hz 目标

`Go2ConstrainedMPC` 是另一个控制路径：启动时在固定站立工作点计算 A/B、力的局部导数，
凝聚状态变量并缓存 QP Hessian/约束矩阵。在线只更新测量状态对应的梯度和约束边界。
Newton predictor/corrector QP 使用原变量/对偶 warm start；CPU 可提前停止，CUDA 使用
固定最大迭代预算，不检查 GPU host 标量。刚体递推按三层同时计算四腿，在线轨迹检查
使用预热的 TorchScript 图；完整自动导数和 CPU oracle 仍保留作为正确性基线。

默认 B=1、T=2、dt=0.02、float64、4 足平地固定支撑；2 节点对应 40 ms 预测时域。
这是为 20 ms 周期选择的局部站立模式，不是把完整非线性 DDP 替换成等价算法。
工作点不在线刷新；当前测量在参考的局部配置增量每分量 <=0.35、速度每分量 <=2
时才尝试控制。该范围是使用域检查，不是收敛域/稳定性证明。

### 硬约束与返回语义

全部约束以 `G du <= b` 进入 QP，没有惩罚 slack、最终动作裁剪或力截断：

- 关节扭矩：`-limit <= u <= limit`，默认官方 URDF effort，可配置更小的 12 维 limit。
- 法向力：默认 `1 <= fz <= 200` N；minimum 可设为 0，禁止负法向力。
- 内接摩擦棱锥：`|fx|+|fy| <= mu*fz`，默认 mu=0.6，严格位于圆形 Coulomb 锥内。
  法向为世界 +z，仅适用于平地；这是保守内近似，会拒绝部分圆锥内的力。

局部力模型存在误差，QP 默认将力边界再收紧 0.2 N、扭矩边界收紧 1e-5 Nm。
求得候选后，从当前测量开始用完整非线性模型重新推进 **所有受控节点**，逐节点检查
真实 KKT 接触力和扭矩，float64 容差为 1e-7（float32 为 2e-4）。收紧边界不是最终保证，
只有通过这一检查的轨迹才标为 `result.feasible/usable=True`。
CPU 先检查优化候选，必要时重新检查已缓存的控制序列；CUDA 并行检查两个候选。
没有通过检查的轨迹不返回为可用，也不无条件沿用上一动作。

```python
import torch
from crocoddyl_batched_mpc.models import Go2ConstrainedMPC

# 在进入控制循环前构建/编译/预热。应用自行配置线程策略。
controller = Go2ConstrainedMPC(batch_size=2, horizon=2, dt=0.02, friction=0.6)
state = controller.reference.expand(2, -1).clone()
action, result = controller.compute(state)
# 只有 result.usable 的环境可执行动作；其他环境的 action 为 NaN，须停止/重新规划。
controller.reset(torch.tensor([False, True]))
```

`SUCCESS` 表示收敛的 QP 候选通过非线性检查；`MAX_ITERATIONS` 表示有效有限预算候选
或经过检查的旧序列。`NO_FEASIBLE_CANDIDATE` 表示本次没有找到通过检查的候选，
不等价于数学不可行证书；`OUTSIDE_LOCAL_MODEL` 与异常观测分别明确报告。
失败环境的轨迹/动作均为 NaN；不要使用通用 controller 的 hold-last 策略绕开检查。
模型、限制、参考和地面设置是初始化配置；修改后重建控制器，不支持原地热修改这些张量。

### 性能复现与边界

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python examples/go2_constrained_mpc.py --steps 200 --require-deadline
# 独立 plant 在已配置的 croco_env 中运行
PYTHONPATH=.dev-tools/croco-numpy126:src OPENBLAS_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 \
  /Users/zhengyuanhao/anaconda3/envs/croco_env/bin/python \
  examples/go2_constrained_mpc.py --plant pinocchio --steps 200 --require-deadline
```

示例设置 Torch 单线程，并分别记录初始化/预热和控制周期。控制延迟包含 QP 与完整
非线性可行性检查，不含传感器、plant/模拟器、执行器 IO；GPU 计时时同步仅在示例中。
`--require-deadline` 在任何周期超过 20 ms 时退出失败，不会把平均延迟当成完整验收。
本机 CPU B=2/T=2 的 200 周期独立 plant 测量：中位 9.50 ms、P99 14.85 ms、
最大 15.88 ms、零超时，扭矩/力约束违反量均为零。完整数据见
`benchmarks/go2_50hz_cpu_2026-10-06.json` 和 `validation.md`。

该结果仅对应记录的配置；Python/macOS 不是硬实时系统，不能据此保证任何运行的最坏延迟。
更长 horizon/更多环境/CUDA 需分别测试。该固定模式控制器尚无离地互补、碰撞检测、
真实状态估计或硬件联调；单边约束是固定支撑模式中的法向力约束，并不实现足端自动离地。

## 显式接触事件与模式控制器

`Go2HybridDynamics` 使用 FL/FR/RL/RR 四个固定力槽，`contact_mask` 为 bool `[4]`
或 `[B,4]`。激活足保持世界锚点和点接触 KKT；非激活足移除约束，其力/力导数为零。
全 False 支持无约束腾空动力学。位置门禁使用单独的几何递推，避免计算无关的刚体量。
模式/锚点是模型初始化配置，不可在已冻结的 TorchScript 模型上原地修改。

`transition(state, previous_mask)` 是纯接触事件函数，不修改输入或模型：

- 释放保留 q/v，不施加虚构冲量；激活足的锚点不能被重新定位。
- 新触地足必须接近平地及目标锚点（默认 2 mm），法向速度不能明显朝上。
- 有新增接触时计算完全非弹性、粘着点接触冲量：
  `M dv - J.T p = 0`、`J(v+dv)=0`；同时投影保留足和新触地足的速度。
- 所有碰撞冲量必须满足 `pz>=0`、`|px|+|py|<=mu*pz`；检查接触速度残差和动能不增。
  需要拉力或超摩擦的粘着碰撞被拒绝，不裁剪冲量，也不默默改为滑动/释放另一足。
- q 不做触地位置吸附；拒绝的环境保持输入状态，冲量为零。

`Go2ContactSwitchingMPC` 管理预构建的模式。每个环境独立记录模式索引。
`prepare_mode()` 在控制循环外计算导数、约束矩阵并编译/预热图；当前局部站立 MPC
要求支撑集能平衡参考，因此腾空仅有动力学，尚无腾空 MPC。
参考姿态的激活足位置必须与准备的锚点一致，切换时保留足锚点必须完全一致。
非支撑腿可以设置不同参考姿态，用于抬腿/下降；当前不是连续摆腿轨迹优化。

```python
import torch
from crocoddyl_batched_mpc.models import Go2ContactSwitchingMPC

torch.set_num_threads(1)
controller = Go2ContactSwitchingMPC(
    batch_size=2, horizon=2, minimum_normal_force=0.0, force_margin=1e-4
)
pose = controller.controllers[0].reference[:19].clone()
pose[9] -= 0.05  # FL calf 姿态目标，抬足约毫米量级
controller.prepare_mode("lift_fl", [False, True, True, True], reference_configuration=pose)
state = controller.controllers[0].reference.expand(2, -1).clone()
event = controller.switch("lift_fl", state, mask=torch.tensor([True, False]))
# accepted=False 的环境保留原状态/模式。仿真器必须实际应用事件后的速度。
state = event.state
action, result = controller.compute(state)
# 仅执行 result.usable 的动作，并使用当前模式推进 plant；下一周期读新测量。
```

切换还要求目标模式从碰撞后状态得到通过全部硬约束检查的 MPC 轨迹。
接受切换时清除该环境所有模式的原变量、对偶和有效性缓存；未选中/被拒绝环境的缓存
保持不变。被拒绝的触地请求需要停止/重新规划，不能继续用旧模式忽略已经接近的地面。
`reset(mask)` 只清 warm start，不会把接触模式重置成四足支撑。
当前 `compute()` 按模式调用各模式控制器；每次预测时域内仍为同一个支撑模式。
这个模式控制器尚未在预测节点间规划释放/触地，也没有自动接触检测或离地互补。
连续平地慢速 walk 已由独立的 `Go2SlowWalkController` 实现，见 [walk 文档](go2_walk.md)。

```bash
PYTHONPATH=.dev-tools/croco-numpy126:src OPENBLAS_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 \
  /Users/zhengyuanhao/anaconda3/envs/croco_env/bin/python \
  examples/go2_contact_switching.py --plant pinocchio
```

示例仅环境 0 切换，环境 1 保持站立；两轮抬 FL、下降、四足触地。
触地周期显式注入向下速度扰动以验证非零碰撞冲量。独立 Pinocchio plant 按当前支撑集
推进，约束门禁继续逐周期执行。记录实际抬足高度、零非支撑力、能量、约束与延迟。
新增路径的计时包含接触事件和 compute，另行排除模型准备/plant/传感器/IO；
上一节固定站立 50 Hz 的测量不能推广到混合模式或切换周期。详见 validation.md。
