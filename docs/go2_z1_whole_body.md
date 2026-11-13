# Go2–Z1 W 全身规划器 CPU 首版

2026-11-13。在独立 Go2–Z1 Pinocchio 模型上新增
[`standing_whole_body.py`](../src/crocoddyl_batched_mpc/planners/standing_whole_body.py)，
实现固定四足支撑的 Crocoddyl Box-FDDP 离线预测问题。
本轮完成静态平衡、数值导数和小幅手端运动验收；尚未接入统一 Guidance、WBC 或物理环境。
这是阶段 A 的 W 参考实现，不能据此认定已完成 C0/C1/W 对照或实机控制。

已新增 [Go2–Z1 三维预测回放 demo](go2_z1_demo.md)，展示独立求解的 3 s 手端往返任务、
整机姿态、质心和接触力；该示例保持原模型、规划器和单 Go2 训练路径不变。

## 模型与求解

- 沿用已冻结的装配、TCP、固定夹爪、零额外载荷和零被动力矩假设，质量 21.60796983 kg；
  细节见 [模型记录](go2_z1_model_plan.md)。不下载 mesh，不改变原 URDF 或现有依赖。
- 物理状态为 q25/v24，切空间 48；18 个腿臂驱动关节力矩，各节点独立优化。
  世界系四足锚点在规划器构造时固定，基座仍可自由运动；首版只接受水平共面支撑。
- 原生 `ContactModel3D(LOCAL_WORLD_ALIGNED)`，无接触切换。使用明确的 Baumgarte 稳定项：
  `J*a + dJ*v + 100*(p-p_anchor) + 20*J*v = 0`。
  另行记录未加稳定项的接触加速度和足端漂移，不把稳定项解释为真实足端作用力。
- 使用 Crocoddyl 半隐式 Euler：`q_next=integrate(q,dt*v+dt²*a)`、`v_next=v+dt*a`；
  默认 dt=0.05 s、10 节点、0.5 s 时域、最多 100 次迭代、停止阈值 1e-7。
- 代价包含世界 TCP 位置/朝向/速度、质心、基座、名义姿态/速度、静态力矩参考及力矩变化。
  TCP 测试任务为五次平滑位移，起终点速度/加速度为零，朝向固定。
  质心、基座和名义关节参考仍取初始姿态，尚非共同运动学参考生成器。
- 真正增广上一节点力矩 `z`：`z_next=u`，每个运行节点增加
  `dt * 0.01 / 2 * ||u-z||²`。求解状态/切空间变成 67/66，控制仍为 18。
  `z0` 可传入上一实际力矩，默认采用通过支撑检查的静态力矩；终端没有变化代价。
  这不等价于每节点对冻结“上一力矩”的独立惩罚。
- 力矩上下界由 Box-FDDP 原生 box 处理。摩擦使用内接菱形
  `|fx|+|fy| <= 0.6*fz`，法向力范围为 [2,200] N；它们在优化内均为软 barrier。
  关节位置/速度、基座位移和姿态范围同样是软 barrier，全部通过末端检查决定可行性。
  默认基座相对初始位置界为 [0.08,0.08,0.05] m，旋转 log 各分量界为 0.25 rad。
  基座线速度/角速度各分量界为 0.5 m/s、1 rad/s；关节边界取 URDF。

静态初始化采用 Crocoddyl `quasiStatic`，再用独立 Pinocchio KKT 方程检查加速度、力矩、
摩擦和法向力。此初始化不是带不等式的完整静态可行性优化器；若种子不通过，会明确拒绝，
不会据此宣称该姿态无解。名义姿态的静态法向力约 46.74–59.25 N，最大力矩占 URDF 限值
21.04%，独立加速度最大绝对值约 6.73e-13。

## 输出和拒绝规则

`solve()` 返回 `WholeBodyPlan`。其中 `mpc` 沿用现有 `MPCResult`，物理状态/力矩采用
FL、FR、RL、RR、Z1 J1–J6 的公共顺序；另存 native 增广状态/控制用于复核。
原生关节排序通过模型元数据转换。`forces_world` 的四足顺序为 FL、FR、RL、RR，
`accelerations` 为公共顺序 24 维广义加速度。

每个节点使用独立 Pinocchio 稠密 KKT 重新计算接触力、加速度，检查动力学残差和
流形积分 gap，包括初值、力矩记忆和终端状态。终端力检查采用最后一节点力矩保持的假设。
力矩/摩擦/法向力/关节/速度/基座边界违反容差为 1e-6，gap 容差为 1e-6，
动力学及带稳定项的接触方程残差容差为 1e-7，足位移容差 1 mm、足速度容差 0.01 m/s。
这些是离散预测节点门槛，不提供节点之间的连续约束或碰撞保证。

分别报告三个结论：

1. `converged` 是优化器数值收敛；达到预算后返回 `MAX_ITERATIONS`。
2. `diagnostics.feasible` 是独立物理/边界检查。即使求解器收敛，违反门槛仍返回
   `NO_FEASIBLE_CANDIDATE`，`mpc.usable=False`；数值失败返回 `NUMERICAL_FAILURE`。
3. `tracking_passed` 要求全时域 TCP 位置峰值误差不超过 5 mm、旋转误差不超过 0.02 rad；
   `accepted_for_task` 还要求可行。目标不可达时不能仅凭 `SUCCESS` 判定任务完成。

现有 `MPCResult.usable` 允许可行的预算耗尽候选，不包含任务误差门槛。
验收示例要求同时收敛、可行和任务跟踪通过，并在失败时退出非零。
本模块不执行力矩，也不提供控制回退；`MPCResult.action` 失败时返回零的既有通用行为
不能直接当作站立带臂控制器。后续 Guidance/WBC 必须实现旧有效参考保留、超时和终止逻辑。

## 本机验收

复用已有 Python 3.11.4 / NumPy 1.26.4 / Pinocchio 4.0.0 / Crocoddyl 3.2.1 / Torch 2.2.2。
原生依赖仅在新路径构造时加载。新测试 **30 passed**；旧 Go2 adapter **3 passed / 1 skipped**，
跳过原因是本机无 CUDA。没有再次运行既有 Go2 完整站立闭环或训练任务。

导数中心差分覆盖完整 66 维增广切空间和 18 维控制，包括非零基座姿态/速度、接触力导数、
激活的力约束 barrier、终端成本和相邻控制交叉项。最大绝对误差：Fx 4.84e-8、Fu 3.24e-8、
Lx 1.24e-6、Lu 9.01e-13、接触力对状态/控制导数 1.98e-8/1.47e-8。
另包含完整独立 KKT rollout、全局合力/力矩平衡、异常候选和失败状态测试。

四个场景各执行一次，均从同一名义姿态开始，独立初始化，没有 warm start：

| 场景 | TCP 世界位移 m | 迭代 | 位置 RMSE mm | 位置峰值 mm | 求解 ms | 构建+求解+检查 ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| hold | [0,0,0] | 1 | <1e-9 | <1e-9 | 9.36 | 47.56 |
| reach | [0.01,0,0] | 3 | 0.0233 | 0.0444 | 23.74 | 56.82 |
| side | [0,0.02,0] | 4 | 0.0726 | 0.1091 | 33.12 | 65.69 |
| combined | [0.01,0.02,0] | 4 | 0.0736 | 0.1107 | 29.38 | 58.58 |

四例的力矩、摩擦、法向力、关节及基座边界违反量均为零。最大预测 gap 8.13e-14，
动力学残差 5.69e-14，带稳定项接触残差 5.91e-15，足漂移 5.53e-7 m，
TCP 朝向峰值 4.60e-4 rad，最大力矩占限值 21.64%。
这些是离线模型预测误差，不是物理仿真或实机跟踪误差。
计时为单次样本，不是实时频率验收；表中总时间不含一次性初始化约 20.04 ms。

逐节点状态、力矩、接触力、参考、软件版本、参数、源码/URDF 哈希、原始日志和 JUnit
均保存在 [验收目录](benchmarks/go2_z1_whole_body_2026-11-13/summary.json)。
本轮开始前的 98 个既有代码/资产/测试/示例/配置文件及 82 个依赖元数据文件哈希均保持一致。
未修改单 Go2 训练代码、包入口、`pyproject.toml` 或任何依赖环境，未操作集成仓库或训练服务器。

## 复现与后续

在源码仓库执行，不重新安装任何包；新模型资产的 wheel/sdist 收录仍待处理。
只把 cmeel 的 site-packages 加到子进程，不能把其上级目录中的 NumPy 2.x 加入本环境。

```bash
(
  export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
  export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1
  export PYTHONPATH=.dev-tools/crocoddyl321-pin400/cmeel.prefix/lib/python3.11/site-packages:src
  .venv/bin/python -B -X faulthandler -m pytest -p no:cacheprovider \
    -q -s tests/test_standing_whole_body.py
  .venv/bin/python -B -X faulthandler examples/go2_z1_whole_body.py \
    --output /tmp/go2-z1-whole-body.json
)
```

后续应先建立共同运动学参考、Guidance 与统一 WBC，再验证在线重规划、旧参考回退、时戳/
有效期和闭环状态扰动。本版是单次离线时域求解，尚无 horizon shift/warm start。
Isaac Lab/PhysX、碰撞、非零载荷、12/36 场景对照、C0/C1、实时吞吐和 Torch/CUDA 批处理均待验收。
不把此 CPU 规划器接到现有单 Go2 RL-guided MPC 训练入口。
