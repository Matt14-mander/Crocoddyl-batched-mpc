# 项目验证记录

## 2026-10-06 M2.6 Go2 局部硬约束站立控制与 50 Hz

- 新增批量 predictor/corrector 不等式 QP，与独立 NumPy 活跃集穷举的最优解对照。
  可行性与 KKT 收敛分别报告；没有软约束或不可行性证明。
- 新增 `Go2ContactConstraints`、`Go2ConstrainedMPC`：缓存站立工作点导数和凝聚矩阵，
  在线约束扭矩、非负法向力及内接摩擦棱锥。完整非线性模型逐节点复核所有候选。
  不可用动作返回 NaN，拒绝无条件保持旧扭矩；局部使用域越界有独立状态码。
- 四腿运动学由 12 次串行递推改为 3 层并行递推，保留原有独立刚体量/导数 oracle 容差。
  在线检查使用启动时编译和预热的 TorchScript；完整 DDP/FDDP 路径继续作为离线基线。

### 实际测试

用户指定的 croco_env（Python 3.10.18 / Torch 2.2.2 / Crocoddyl 3.0.1 /
Pinocchio 3.6.0，独立 NumPy 1.26.4 路径）完整回归：
**106 passed, 22 skipped**，437.59 秒；全部跳过项为 CUDA 不可用。
新增用例覆盖活跃摩擦/法向力/扭矩边界、完整预测轨迹复核、无可行候选、异常观测、
工作域越界、mask reset 与环境隔离，以及 120 周期独立 Pinocchio plant 闭环恢复。
三层并行化后的 Torch Go2 专项在 Crocoddyl 3.2.1 / Pinocchio 4.0.0 下
**8 passed, 2 skipped**，原 oracle 门槛不变。
`ruff check .`、`git diff --check`、wheel/sdist 构建通过；从 wheel 解包路径
实际导入并实例化局部控制器，平衡动作及其硬约束检查通过，无需 Pinocchio。

```bash
PYTHONPATH=.dev-tools/croco-numpy126:src \
  OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 \
  /Users/zhengyuanhao/anaconda3/envs/croco_env/bin/python \
  -X faulthandler -m pytest -q -ra
```

### 20 ms 周期实测

独立 Pinocchio KKT/integrate plant，macOS Intel x86_64 / Python 3.11.4 /
Torch 2.2.2 / Crocoddyl 3.2.1 / Pinocchio 4.0.0，Torch/BLAS 单线程。
B=2、T=2、dt=0.02、float64，200 周期；初始 roll ±0.02 rad、关节位移
0.015 rad、速度 0.02；第 32 周期注入角速度 `[0.08,-0.06,0.04]` rad/s。

```bash
PYTHONPATH=.dev-tools/crocoddyl321-pin400/cmeel.prefix/lib/python3.11/site-packages:src \
  OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python examples/go2_constrained_mpc.py \
  --plant pinocchio --steps 200 --require-deadline
```

| 指标 | 实测 |
| --- | ---: |
| 在线延迟中位 / P95 / P99 | 9.50 / 11.19 / 14.85 ms |
| 在线最大延迟 / 超过 20 ms 的周期数 | 15.88 ms / 0 of 200 |
| 扭矩 / 接触不等式最大违反 | 0 Nm / 0 N |
| 最小法向力 | 32.90 N |
| 最大 `(|fx|+|fy|)/fz`（限制 0.6） | 0.10323 |
| 最大扭矩 / URDF effort | 0.13521 |
| 最终状态切空间误差（两个环境） | 3.323e-5 / 3.328e-5 |

原始数据见 [CPU 50 Hz 测量](benchmarks/go2_50hz_cpu_2026-10-06.json)。
计时包含 compute、QP 和全部预测节点的非线性约束检查；启动准备 7.90 秒、plant、
传感器和通信/执行器 I/O 不计入。短测 T=3 曾通过，但延长到 200 周期出现超时，
因此默认预测时域选为 T=2（40 ms），上述结论只限此实测配置。
这不是操作系统最坏延迟或真实硬件控制链的硬实时保证。

当前为平地四足固定支撑、缓存局部工作点，约束力采用圆形 Coulomb 锥的保守内近似。
没有离地互补、接触切换、关节位置硬约束或硬件联调；CUDA/更大批量与预测时域仍需实测。

## 2026-10-05 M2.5 Go2 Torch 批量接触与站立 MPC

### 实现

- 新增 `Go2TorchDynamics`：独立解析官方 URDF，包括固定 rotor/foot 的完整惯性合并。
  在线仅使用 Torch 计算运动学、M/h/J/classical drift、双边接触 KKT 和自由浮基积分。
  nx=37、ndx=36、nu=12；支持 shared/per-environment 世界锚点及足序/支撑子集。
- 半隐式积分和 48 方向 `vmap(jvp)` 自动局部导数，直接接入 DDP/FDDP。
  `solve_ex` 将奇异环境的解标为 NaN；求解器继续按环境报告数值失败。
- `Go2StandingCost`、平衡扭矩初始化及 `go2_standing_problem`；
  `examples/go2_standing_mpc.py` 提供 Torch/独立 Pinocchio plant、扰动和验收指标。
- 新增 float32/float64 平衡与批量隔离、独立锚点、CPU 外部 oracle、离散导数、
  DDP/FDDP warm start/reset/异常四元数隔离、站立扰动恢复与 CUDA stream 测试。

### 外部对照与闭环

Python 3.11.4 / Torch 2.2.2 / NumPy 1.26.4 / Crocoddyl 3.2.1 / Pinocchio 4.0.0，
macOS x86_64，单线程 CPU。使用项目独立 cmeel 目录，原 croco_env 保持不变。

```bash
PYTHONPATH=.dev-tools/crocoddyl321-pin400/cmeel.prefix/lib/python3.11/site-packages:src \
  .venv/bin/python -X faulthandler -m pytest -m crocoddyl -q -s
PYTHONPATH=src .venv/bin/python examples/go2_standing_mpc.py --steps 65
```

- 新 Torch 专项：**8 passed, 2 skipped**；跳过项为 CUDA 不可用。
- 全部 Crocoddyl marker：**21 passed, 96 deselected**，173.39 秒。
  其中站立闭环使用独立 NumPy KKT 连续动力学与 Pinocchio integrate 推进 plant。
- 在用户指定的 croco_env（Python 3.10.18 / Crocoddyl 3.0.1 / Pinocchio 3.6.0，
  `PYTHONPATH=.dev-tools/croco-numpy126:src`）执行完整回归：
  **96 passed, 21 skipped**，433.29 秒；21 项全部为 CUDA 不可用。
- 四足/两足（RR、FL）与零/非零稳定化：M/h/J/drift/a/f 与 CPU KKT 的
  atol/rtol 门槛 1e-9；连续 a/f 自动导数与 Crocoddyl 的门槛 2e-8；均通过。
- 离散局部 Fx/Fu 与独立 Pinocchio plant 的 48 列中心差分对照：门槛 2e-7，通过。
- 单次随机扰动观测误差：M=3.55e-15、h=2.84e-14、J=3.33e-16、drift=1.12e-15、
  a=3.38e-14、f=3.55e-14。它们不代替测试中的明确容差。

站立闭环：B=2、dt=0.02、T=8、每周期 1 次 DDP，65 周期；初始两环境 roll
分别 ±0.02 rad、12 关节位移 0.015 rad、18 维速度 0.02；第 32 步注入
base body 角速度 `[0.08,-0.06,0.04]` rad/s。没有执行最终动作裁剪。

| 独立 plant 观测 | 环境 0 | 环境 1 |
| --- | ---: | ---: |
| 初始状态切空间误差范数 | 0.101489 | 0.101489 |
| 最终状态误差范数 | 0.000801 | 0.000881 |
| 最后 5 周期最大状态误差 | 0.001393 | 0.001507 |
| 最终配置误差范数 | 0.000107 | 0.000122 |
| 最终速度范数 | 0.000794 | 0.000872 |
| 最大扭矩/URDF effort | 0.135744 | 0.135580 |
| 最小法向力 (N) | 30.148577 | 30.766677 |
| 最大切向力/法向力 | 0.078683 | 0.102579 |
| 最大接触加速度残差 | 5.65e-15 | 7.47e-15 |
| 最终最大足端位置误差 (m) | 3.05e-5 | 2.80e-5 |

65 周期墙钟耗时 **166.65 秒**（同时有另一回归进程）；这是离线正确性演示，
不代表 dt=0.02 的实时频率。另一次 Torch plant 演示为 155.92 秒，扰动在第 30 步。
FDDP CLI smoke（B=2、T=2、4 周期）全部动作可用，3.19 秒；不作为长期恢复验收。

### 打包与适用边界

- `ruff check .`、`git diff --check`、wheel/sdist 构建通过。
- 从 wheel 解包路径实际实例化 Torch Go2，无需 Pinocchio；平衡加速度最大 1.21e-12。
  float32 Fx/Fu 对 float64 的平衡状态误差分别 1.49e-6 / 7.46e-8。
- CUDA 用例已编写，本机无 CUDA，未执行 GPU/stream 或实时性能验收。
- 模型是双边固定点接触。没有摩擦锥、单边力、扭矩/关节 box 约束求解、接触切换、
  碰撞检测或 Isaac Lab/硬件联调。轨迹中的正法向力、摩擦比和扭矩余量仅为观测门禁。
- Torch 刚体量与 Pinocchio 独立实现，但共享官方 URDF，不验证实际硬件惯性参数。
  下一轮优化自动导数、刚体递推与分配开销，再推进受约束接触和模拟器闭环。

## 2026-10-04 Crocoddyl CI 段错误修复

用户提供的 Linux/Python 3.12 日志显示 Crocoddyl 3.2.1 在 Go2 接触 action 的
首次 `model.calc()`（tests/test_go2.py）处退出码 139；此前 FDDP 对照正常运行。
本机使用项目独立目录安装相同 Crocoddyl wheel 系列进行版本对比，未修改 croco_env。

- Python 3.11.4 / macOS x86_64 / Crocoddyl 3.2.1 / Pinocchio 4.1.0：
  在同一 `calc()` 位置复现原生段错误；`pip check` 无法检查二进制兼容性。
- 保留 Crocoddyl 3.2.1，将 Pinocchio 改为 4.0.0：Go2 7 项原有对照全部通过。
  这确认本机故障取决于原生依赖版本组合；具体 C++ 崩溃机制未用调试器确认。
- `crocoddyl` extra 固定 Crocoddyl 3.2.1 / pin 4.0.0，pin 自身要求
  libpinocchio 4.0.0。Go2 factory 在 3.2.1 搭配 Pinocchio >=4.1 时提前报错。
- CI 增加 `pip check` 和 Python faulthandler；没有删减接触对照或放宽数值容差。

实际验证（Torch 2.2.2，测试使用 Python 3.11 的 NumPy 1.26.4）：

```bash
PYTHONPATH=.dev-tools/crocoddyl321-pin400/cmeel.prefix/lib/python3.11/site-packages:src \
  .venv/bin/python -X faulthandler -m pytest -m crocoddyl -q -s
PYTHONPATH=.dev-tools/croco-numpy126:src \
  /Users/zhengyuanhao/anaconda3/envs/croco_env/bin/python \
  -X faulthandler -m pytest -m crocoddyl -q -s
```

- 新组合：**15 passed, 88 deselected**，6.89 秒。
- 原 croco_env（Crocoddyl 3.0.1 / Pinocchio 3.6.0）：**15 passed, 88 deselected**，7.05 秒。
- 在实际 3.2.1 / 4.1.0 环境运行新增防崩溃测试：**1 passed, 8 deselected**。
- 新组合安装目录的依赖元数据检查通过；`ruff check .`、`git diff --check` 通过。

本机验证为 macOS/Python 3.11 与原有 Python 3.10；GitHub 的 Linux/Python 3.12
结果仍须推送后重新运行确认。独立目录由 Git 忽略，干净安装使用 extra 中的版本固定。

## 2026-10-01 M2.4 Go2 固定接触 CPU 参考

首个机器人模型已选定宇树 Go2，固定官方 `unitreerobotics/unitree_ros` commit
`5994d4faef0a9cadd3287f8de0199a67eeb2a259` 的原始 URDF，并保存 BSD-3-Clause
许可证及 SHA256 来源信息。详情见 `go2.md`。

### 本轮实现与环境

- Go2 加载器：自由浮基、12 个标量驱动关节，nq=19、nv=18、nx=37、ndx=36。
  显式 FL/FR/RL/RR、hip/thigh/calf 映射，四个 foot frame，URDF 位置/effort 限制。
- CPU 固定足端 KKT：Pinocchio M/h/J/drift，世界轴点接触力，位置/速度稳定化项。
- 静态平衡扭矩初值、加速度/力中心差分导数及独立 Crocoddyl action 对照。
- 模型资产随 wheel/sdist 分发；Pinocchio/NumPy 为延迟导入的可选 `robotics` extra。

执行环境为此前 croco_env（Python 3.10.18、Torch 2.2.2、Pinocchio 3.6.0、Crocoddyl 3.0.1），
继续通过项目内 NumPy 1.26.4 路径执行测试，没有新增修改原有环境依赖。

### 实际执行

```bash
PYTHONPATH=.dev-tools/croco-numpy126:src \
  /Users/zhengyuanhao/anaconda3/envs/croco_env/bin/python -m pytest tests/test_go2.py -q
PYTHONPATH=.dev-tools/croco-numpy126:src \
  /Users/zhengyuanhao/anaconda3/envs/croco_env/bin/python -m pytest -q -rs
PYTHONPATH=.dev-tools/croco-numpy126:src \
  /Users/zhengyuanhao/anaconda3/envs/croco_env/bin/python examples/go2_contact_reference.py
```

- Go2 专项：**8 passed**，2.71 秒。覆盖模型资产、驱动/足端顺序、浮基几何、静态平衡，
  四足/两足（含逆序）支撑和零/非零稳定化的加速度、力及局部导数，错误输入和结果隔离。
- 完整回归：**83 passed, 19 skipped**，259.20 秒；跳过项全部为 CUDA 不可用。
- 新 Go2 action 加速度/力门槛为 atol/rtol=1e-9，中心差分导数门槛为 2e-5；均通过。
- `ruff check .` 和 `git diff --check` 通过。
- wheel 构建成功，验证包含 `go2.urdf/LICENSE/SOURCE.json` 与正确 SHA256；
  从 wheel 解包路径实际加载 Go2 得到 nq=19、nv=18。
- 基础 `.venv` 未安装 Pinocchio，仍可导入模型模块，实际调用加载器时给出可选依赖提示。

四足静态示例：模型惯性总质量 16.087 kg，根部高度 0.296797 m，
每个前足竖直力约 39.2713 N、后足约 39.6354 N，总计 157.81347 N；
最大扭矩约 5.8653 Nm，低于该 URDF 的关节 effort 限制。
最大加速度=5.09e-13，动力学残差=8.88e-16，接触加速度残差=6.49e-15。

| 与 Crocoddyl 的对照（静态示例） | 最大绝对误差 |
| --- | ---: |
| 广义加速度 | 3.05e-13 |
| 世界坐标接触力 | 1.42e-14 |
| 加速度对状态导数 | 1.32e-7 |
| 加速度对控制导数 | 4.90e-8 |
| 接触力对状态导数 | 1.74e-8 |
| 接触力对控制导数 | 1.27e-8 |

### 当前边界

该模型仅实现双边固定点接触 CPU oracle；尚无 Go2 Torch/GPU 接触动力学、
离散积分及 DDP/FDDP 站立闭环。没有实现摩擦锥、单边力、扭矩约束、接触切换或硬件控制。
静态示例的正竖直力与 effort 余量不能代替一般姿态下的不等式约束。
KKT 与 Crocoddyl 复用 Pinocchio 的刚体量，对照并不验证实际 Go2 硬件惯性参数。
下一步实现 Torch 批量固定接触模型并使用这些 oracle 门禁核对，之后验收 Go2 站立闭环。


## 2026-10-01 M2.3 浮基流形与切空间验收

### 实现

- 新增 `SO3Manifold`、`SE3Manifold`、`FloatingBaseManifold`，采用 xyzw 单位四元数、
  右侧机体系增量与 SE(3) 指数映射，支持自由浮基和欧氏标量关节。
- dynamics/cost 显式支持 `ndx`；旧欧氏模型保留缺省 `ndx=nx`。状态轨迹仍为 nx，
  导数、gap、反馈增益使用 ndx，并检查返回导数的形状。
- FDDP backward pass 在非平坦流形上用差分 Jacobian 将动力学输出导数转换到
  下一名义状态的 residual chart，再注入 gap；欧氏热路径保持原有递推。
- 新增局部 Torch JVP 导数辅助函数、manifold residual 二次代价及 Gauss-Newton Hessian。
- 求解器使用有效 neutral 状态保护异常环境；零/非单位四元数属于逐环境数值失败。
  horizon-shift warm start、reset 及初始化辅助函数支持状态/切空间维度不同。

### 实际验证

执行环境仍为本机 croco_env：Python 3.10.18、Torch 2.2.2、Pinocchio 3.6.0、
Crocoddyl 3.0.1，测试路径中的 NumPy 1.26.4，CPU 单线程。

```bash
PYTHONPATH=.dev-tools/croco-numpy126:src \
  /Users/zhengyuanhao/anaconda3/envs/croco_env/bin/python -m pytest -q -rs
```

- 完整回归：**75 passed, 19 skipped**，153.85 秒；全部跳过项为 CUDA 不可用。
- 新增 24 项测试中 CPU/外部 oracle 的 18 项通过，6 项 CUDA 几何用例跳过。
- Pinocchio/Crocoddyl 对照：0/2 关节浮基，多种初态和增量，integrate/diff 的
  `atol` 门槛 2e-12、Jdiff 门槛 2e-11，全部通过；未放宽原有 oracle 容差。
- SO(3)/SE(3)/浮基局部差分 Jacobian、动力学局部导数和代价梯度均通过独立有限差分。
- 覆盖小角度、identity、接近 pi、四元数正负等价、float32/float64。
- 验证 nx=4/ndx=3 的 DDP/FDDP，独立环境求解一致、不可行姿态初值、
  非零 gap 下首轮反馈增益、浮基缓存/reset/异常四元数隔离。
- 原有线性 LQR、摆闭环及 Crocoddyl FDDP 外部对照全部通过。
- `ruff check .` 和 `git diff --check` 通过。
- `PYTHONPATH=src .venv/bin/python examples/floating_base_state.py --joints 12`：
  nq=19、nv=18、nx=37、ndx=36，B=4，float64 往返最大误差 **1.39e-17**。

### 适用边界与下一步

本轮交付浮基几何和求解接口，不含刚体接触动力学。合成测试直接命令切空间增量，
不代表浮基可直接施加控制，也不代表机器人已能站立。
新代价使用 Gauss-Newton Hessian；当前仍忽略二阶动力学/残差曲率，没有扭矩与摩擦约束。
SE(3) 对数的主值切割、SE(3) Jacobian 级数和 JVP 基线的 GPU 性能尚未在本机验证。
下一步为选定机器人模型，实现固定接触动力学，核对加速度、接触力和导数，之后验收站立闭环。
接口、坐标约定及复现说明见 `floating_base.md`。


## 2026-10-01 本机 croco_env 外部 oracle 验收

### 环境与兼容处理

使用 `/Users/zhengyuanhao/anaconda3/envs/croco_env/bin/python`：Python 3.10.18，
Crocoddyl 3.0.1，Torch 2.2.2，pytest 9.1.1，macOS x86_64，CPU 单线程。
原环境缺少 Torch/pytest，本次已补齐。原有 NumPy 2.2.6 已恢复并保留；测试使用
项目内 `.dev-tools/croco-numpy126` 的 NumPy 1.26.4。原因是该 Intel macOS 的 Torch 2.2
与 NumPy 2.x 桥接不兼容，而环境中的 bezier/cmeel-boost 声明依赖 NumPy 2.x。
恢复后在不加测试路径的环境运行 `python -m pip check`：**No broken requirements found**。

本机 EigenPy 将单列 `Fu/Lxu/K` 作为一维数组暴露，导致首次运行的 3 项非线性测试
在数组索引处失败。现已兼容其数学形状，fake action 自检与实际 bindings 均可使用。
没有修改求解算法、接受规则或数值容差。

复现（从项目根目录执行；独立 NumPy 路径被 Git 忽略）：

```bash
/Users/zhengyuanhao/anaconda3/envs/croco_env/bin/python -m pip install torch==2.2.2 'pytest>=8'
/Users/zhengyuanhao/anaconda3/envs/croco_env/bin/python -m pip install \
  --target .dev-tools/croco-numpy126 numpy==1.26.4
PYTHONPATH=.dev-tools/croco-numpy126:src \
  /Users/zhengyuanhao/anaconda3/envs/croco_env/bin/python -m pytest -m crocoddyl -q -s
PYTHONPATH=.dev-tools/croco-numpy126:src \
  /Users/zhengyuanhao/anaconda3/envs/croco_env/bin/python -m pytest -q -rs
```

### 外部对照结果

`-m crocoddyl -q -s`：**5 passed, 65 deselected**，没有 Crocoddyl 跳过项。
完整回归 `-q -rs`：**57 passed, 13 skipped**，耗时 151.45 秒；13 个跳过项全部为
CUDA 不可用，DDP/FDDP 持续闭环、物理约定、fake action 自检与实际外部 oracle 均通过。
`ruff check .` 和 `git diff --check` 通过。

| 对照 | 状态最大绝对误差 | 控制最大绝对误差 | 代价绝对误差 |
| --- | ---: | ---: | ---: |
| Torch LQR vs Crocoddyl（B=2、T=5） | 5.55e-17 | 3.89e-16 | 0 |
| Torch DDP vs Crocoddyl（同线性场景） | 5.55e-17 | 2.50e-16 | 2.22e-16 |
| Torch FDDP vs Crocoddyl（摆、T=12、预算25） | 6.40e-6 | 7.05e-5 | 1.13e-9 |

摆的 action 值与导数通过原有 `atol=1e-12` 门槛，首轮反馈增益
`K_torch` 与 `-K_crocoddyl` 最大绝对误差 **8.71e-5**，通过原有 `atol/rtol=3e-3`。
初态为 `[2.6, -0.3]`，不连续初值最大 gap=0.08，gap_penalty=100。

| 迭代预算 | Torch cost | Crocoddyl cost | Torch gap | Crocoddyl gap | Torch 有效迭代数 | Crocoddyl iteration_index |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 41.837775499 | 41.837826962 | 0 | 1.11e-16 | 1 | 1 |
| 2 | 38.440848675 | 38.440848673 | 0 | 0 | 2 | 2 |
| 5 | 38.440820497 | 38.440820496 | 0 | 0 | 3 | 4 |
| 25 | 38.440820497 | 38.440820496 | 0 | 0 | 3 | 4 |

双方首轮闭合 gap，后续代价改善，最终轨迹通过原有数值门槛。
上述有效迭代计数与 Crocoddyl 返回的 iteration_index 语义不同；接受与停止规则也不完全相同。
结论限于这些模型与初值，不能据此声明完整 FDDP 数值规则已对齐。


## 2026-09-30 物理摆修正与持续闭环验收

### 进度复核与本次开发

- 当前阶段仍为 M2.2：LQR、DDP/FDDP、horizon-shift warm start、逐环境参数更新已实现。
- 修正 Pendulum 的重力方向：角度从向下垂直测量时，重力项为 `-(g/l)*sin(theta)`。
  同步修正解析 Jacobian 和 Crocoddyl oracle 的独立 action 公式。旧模型会使向上平衡点
  自然稳定，旧数值与闭环记录不能用于证明物理倒立摆稳定性。
- 新增 3 项物理门禁：共享/逐环境参数的平衡点恢复方向，以及非单位质量、长度、阻尼的
  扭矩平衡。有限差分只能验证代码和导数一致，不能独立发现共同的物理符号错误。
- 新增 2 项闭环门禁：CPU DDP 与 FDDP，各运行 B=3、100 周期，独立物理 plant、
  horizon=30、每周期最多 6 次迭代。第 60 周期施加单环境状态扰动，验证新测量、warm start
  和持续直立。详见 `ddp_acceptance.md`。
- 摆示例新增 dtype/backend/线程数及持续稳定窗口选项，稳定判据同时检查角度、角速度、
  有限性和动作可用性；未满足验收时返回非零退出码。

### 实际执行结果

环境：macOS x86_64，Python 3.11.4，Torch 2.2.2，NumPy 1.26.4；项目独立 `.venv`。
本机没有 CUDA，该测试环境未安装 Crocoddyl。未更改已有 Anaconda 环境。

- 新闭环测试加入前的完整回归（含新增 3 项物理测试）：**50 passed, 18 skipped**。
- `python -m pytest tests/test_pendulum_closed_loop.py -q`：**2 passed**，耗时约 160 秒。
  两次运行合计覆盖当前收集的全部 70 项：52 项通过，18 项因 CUDA/Crocoddyl 不可用跳过。
- `ruff check .` 与 `git diff --check`：通过。
- `PYTHONPATH=src .venv/bin/python examples/pendulum_swingup.py --batch-size 3`：
  CPU float32，100 周期，最后 20 步持续稳定 **3/3**，不可用求解 **0**，
  最终最大角度误差约 **1e-6 rad**、最大速度约 **4e-6 rad/s**。
- 标准 T=40 的 float64 单次求解：零控制 cost=2419.973216，优化后 cost=509.286965，
  独立 adjoint 最大控制残差=3.47e-5，终端状态=[3.125814, 0.026646]，5 次迭代收敛。

### 下一阶段仍需完成

- 在同时提供 Torch/Crocoddyl 的环境执行修正后的外部 FDDP oracle，记录版本和误差。
- 在 CUDA 环境复验修正后的非线性模型、持续闭环和无 host sync 调用路径。
- 本次闭环使用无约束扭矩及同参数 plant，尚未验收控制约束、模型失配和真实 Isaac Lab。
- M3 尚需先进行非线性求解器 profiling，再决定原生内核与 CUDA Graph 的实现范围。

以下为历史执行记录；本次没有重新验证其中的 GPU 与跨平台性能结论。

## 2026-09-20 FDDP 外部 oracle 准备

- 新增非线性 Crocoddyl `SolverFDDP` 对照：相同半隐式摆模型的运行/终端值与导数、
  不连续初值首轮反馈增益、1/2/5/25 次预算的 gap 与代价、最终轨迹。
- 本机完整测试：**60 passed, 5 skipped**；FDDP 专项为 **8 passed, 3 skipped**。
  新增三个跳过项均为 Crocoddyl 外部 oracle；其余两个是原有 Crocoddyl 对照。
- 本机 `base` 未安装 Crocoddyl；`croco_env` 亦未安装 Crocoddyl/Torch。
  因此目前只有 oracle 模型公式自检通过，**尚无跨库误差或迭代数据**。
- 外部执行命令：`python -m pytest -m fddp_oracle -q -s`。CI 的
  `crocoddyl-reference` 任务运行 `-m crocoddyl`，会纳入新增三个测试。

## 2026-09-12 工程基线复核

- 解决 `models.py` 与 `models/` 同名造成的导入冲突，线性模型现位于 `models/linear.py`。
- 根目录调试和手工运行脚本已归入 `examples/`。
- 完整测试：**59 passed, 2 skipped**；两个跳过项均为未安装的 Crocoddyl bindings。
- DDP 正确性验收：**42 passed, 1 skipped**，包含模型导数、状态机、独立数值门槛
  和 FDDP gap handling。
- M2.2 验收：**18 passed**，覆盖 horizon shift、局部 reset、逐环境参数、FDDP gap 收缩、
  批量隔离及 CPU/CUDA 一致性。
- Ruff 检查通过，M0 LQR 和 M1 模型导数测试可以在同一次测试运行中完成。
- DDP 新增 8 项状态机测试：与精确 LQR 对齐、当前 `x0` 轨迹一致性、逐环境失败、
  Cholesky 失败的正则化重试、线搜索拒绝、固定预算可用解、模型 dtype 校验和
  CUDA 无 host sync。
- Pendulum 单次求解（B=1、T=20、最多 50 次迭代）在 5 次迭代收敛，
  cost=519.4505，终态 θ=3.2642 rad、角速度 1.2589 rad/s。它证明单次求解明显改善，
  尚不能替代长时闭环稳定验收。

日期：2026-09-03。下列为本机执行结果，CI 配置尚未在远程运行。

## 环境

- Windows，Python 3.12.3：`E:\anaconda3\python.exe`。
- PyTorch 2.7.0+cu128，CUDA runtime 12.8。
- NVIDIA GeForce RTX 4070 Laptop GPU，驱动 555.97，显存约 8 GB。
- 系统默认 `python` 指向另一套 MSYS Python，未安装 Torch；本次使用上述 Anaconda 解释器。
- 没有修改已有 Python 环境；Ruff/build 辅助工具置于被 Git 忽略的 `.dev-tools/`。

## 数值与调用路径

`E:\anaconda3\python.exe -m pytest -q`：**17 passed, 1 skipped**。

通过项包括：

- T=1/5、含仿射动力学和线性代价的随机 LQR，与消元后完整控制 Hessian 的独立稠密解对齐。
- 正则项 λ=0/0.2 的控制解和目标值；动力学可行性；逐环境独立求解一致性。
- float32/float64、非连续状态 view、输入不变和无梯度输出。
- 非有限输入/模型与非正定控制 Hessian 的逐环境失败隔离。
- 控制器保持上次有效动作、选择性 reset、返回值不暴露内部缓存。
- 闭环状态误差下降；错误 shape/dtype/参数明确拒绝。
- 真实 CUDA 上 float32/float64 与 CPU 对齐，非默认 stream 及同 stream 即时消费者。
- 预热后 profiler 中无 `cudaStreamSynchronize`、`cudaDeviceSynchronize`、
  `aten::item`、`aten::_local_scalar_dense`。该检查针对当前 Torch 环境，并非所有平台保证。

Crocoddyl 对照测试因未安装 bindings 跳过。参考适配器已编写，运行兼容性及数值对照仍待验证。
独立 CI 任务会先明确导入 Crocoddyl，避免缺依赖导致 CI 静默跳过。

真实 CUDA 示例：

- double integrator，B=1024，100 个闭环步骤：0 次失败，位置 RMS 从 0.577914 降至 0.006807。
- tensor bridge，B=4096：返回 `[4096, 1]` CUDA 动作，全部求解成功。
- 此处的 bridge 没有启动 Isaac Sim；真实机器人环境尚未联调。

## 初步性能数据

最终三角求解版本，B=1024、T=20、nx=2、nu=1、float32，CPU threads=1，
warmup=2、repeats=5。使用 `benchmarks/bench_lqr.py`；含有限性检查、分配、完整轨迹与代价，
CUDA 在计时边界同步，排除模型构建和模拟器。

| 设备 | 中位延迟 ms | 最小/最大 ms | 环境求解数/秒 |
| --- | ---: | ---: | ---: |
| CPU | 25.362 | 23.564 / 25.472 | 40,375 |
| CUDA | 36.758 | 32.644 / 56.126 | 27,858 |

这是小规模烟雾基准，采样数量少，不作硬件或训练性能结论。此配置未显示 GPU 加速优势。
后续需测量更多维度/批量，并减少 Python 调度和临时分配。初次 profiler 发现
`torch.cholesky_solve` 内部同步，现已用两次 `solve_triangular` 替代并加入回归测试。

## 工程检查与未覆盖项

- Ruff 检查通过；sdist 与 wheel 可构建。
- 尚未验证：Crocoddyl bindings、真实 Isaac Lab/Isaac Sim、原生 CUDA 内核、CUDA Graph、
  非线性/接触/约束 MPC、多 GPU、可微求解以及最小 Torch 2.2 版本兼容性。
