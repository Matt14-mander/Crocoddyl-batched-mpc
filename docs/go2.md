# 宇树 Go2：模型与固定接触 CPU 参考

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

下一步是实现可接入 DDP/FDDP 的 Go2 Torch 批量固定接触离散动力学，并以此 oracle 核对
加速度、力、积分和导数，再加入姿态/足端代价、站立闭环及约束求解。
本轮尚无 Go2 MPC 站立闭环、GPU 接触动力学或真实 Isaac Lab 联调。
