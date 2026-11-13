# Go2–Z1 共同参考、Guidance 与固定支撑 WBC CPU 闭环

2026-11-13。在原 W CPU 规划器之外增加独立执行路径：

```text
脚本 TCP 任务 → 共同运动学参考 → 共同 Guidance / W 预测转 Guidance
                                      ↓
实际 q/v → 全身逆动力学 QP → 18 关节力矩 → 独立 Pinocchio CPU 动力学
    ↑___________________________________________________________|
```

这次完成固定四足支撑的 CPU 功能闭环。C0/C1、在线滚动 MPC、Isaac Lab/PhysX、
载荷、行走 WBC 和实时性仍是后续工作。已有单 Go2 训练入口、模型、adapter、
QP 求解器、W 规划器与依赖环境均保持原样；所有新类通过新模块显式导入。

## 实现位置

| 文件（位于 `src/crocoddyl_batched_mpc/`） | 职责 |
|---|---|
| `references/standing_arm.py` | 共同固定支撑 IK、C2 TCP 脚本任务、W 参考适配 |
| `guidance.py` | 不可变参考快照、单位/坐标/模型约束、插值、有效期及回退 |
| `controllers/whole_body_qp.py` | 54 变量逆动力学 QP、约束检查、显式失败结果 |
| `models/go2_arm_cpu.py` | 独立 Pinocchio 接触 KKT 与实际状态积分 |
| `controllers/standing_arm_cpu.py` | 参考缓冲、WBC、plant、终止及指标记录 |

## 共同运动学参考

`StandingArmReferenceGenerator.generate()` 接收初始 q、时间序列和世界 TCP 位姿/
速度/加速度。首版固定 TCP 朝向、基座朝向和四足世界锚点，允许基座平移及 18 关节
协调。采用带弱姿态/时间先验的阻尼 IK，基座相对初态的平移范围为 ±[0.06,0.06,0.04] m，
关节范围来自 URDF；逐节点核查 FK 和足锚点，未通过即拒绝参考。
IK 失败只是当前算法未找到满足门槛的候选，不是不可达证明。

共同参考不依赖 Crocoddyl。它同时提供完整名义状态序列给 W，以及任务空间/关节
参考给 Guidance。TCP 导数来自解析五次曲线；名义关节、基座平移及质心的节点导数
由运动学序列差分构造，首尾静止。默认接触力是均分重力的软参考，不宣称完成 C1 的
准静态承载验证。实际支撑可行性由 WBC 硬约束检查。

## Guidance 契约与生命周期

- SI 单位；世界位姿、世界坐标表达的线/角速度与加速度均以对应任务帧原点为参考。
- 关节顺序是 FL/FR/RL/RR，随后 Z1 J1–J6；接触力按 FL/FR/RL/RR，采用世界坐标。
- 绝对单调时间戳、签发时间、有效期、序列号、有效标志、来源与原因。
- 模型身份包含组合 URDF、重力和关节/浮基坐标约定的 SHA256；同一包内足锚点固定。
- 包含手端/质心/基座参考及其导数、18 关节参考及其导数、四足力参考。
- 复制并冻结数组，检查形状、有限值、SE(3)、时间顺序和模型/接触一致性。
- 平移、质心和关节采用五次 Hermite 插值；姿态在 SO(3) 上插值并计算世界角速度/
  角加速度，节点间保持 C2 连续。相邻旋转接近 π 时拒绝。接触力采用线性插值。

`GuidanceBuffer` 拒收失效、过期、旧序列或错误模型/接触的包；拒收不会刷新旧包
有效期。旧包仍有效时继续执行，过期后从最后有效参考平滑过渡到已配置的初始站立
目标，默认过渡 1 s，随后保持站立。`reset()` 清除包、过渡和时间状态。
这不代表在线滚动 MPC 或新规划包交接已经验收。

W 适配器保留共同 TCP 目标与名义关节参考，使用 W 预测的质心、基座与接触力；
通过模型动力学和足锚点复核后才形成包。W 的关节轨迹仅在显式启用
`use_optimized_joints=True` 时作为消融输入。W 力矩不与 WBC 力矩叠加。

## 统一 WBC QP

每周期以当前 q25/v24 和 GuidanceSample 求解 `z=[a24,f12,tau18]`，共 54 个变量。
全身动力学与 12 个支撑足加速度约束为硬等式：

```text
M*a + h = S*tau + Jc.T*f
Jc*a + dJc*v + 100*(p-foot_anchor) + 20*Jc*v = 0
```

硬不等式包括：18 关节 URDF 力矩范围、各足法向力 2–200 N、
`|fx|+|fy| <= 0.6*fz` 的内接菱形摩擦锥，以及下一步关节位置和速度范围。
下一步位置采用与 plant 相同的半隐式公式 `qj_next=qj+dt*vj+dt²*aj`。

手端、质心、基座、名义关节、接触力和力矩变化是加权任务代价，冲突时可以产生
任务残差。任务反馈采用参考加速度加位置/速度误差反馈。配置与全部权重记录在报告中。
这里的加权任务没有硬性的任务优先级保证，也没有把任务 slack 当作物理约束 slack。

通过 SVD 消去硬等式，通常剩 18 个变量；复用已有 `solve_inequality_qp` 的
Torch float64 CPU 原始-对偶求解器。对不等式同时按系数与上界缩放，避免微小 dt²
形成巨大 slack；在完整物理变量中重新检查等式、不等式、有限性和 QP 收敛。
正常 warm start 来自上一被接受的完整解，重新投影到当前等式零空间。

失败返回 `torques=None`，执行循环终止并记录原因，不输出全零力矩、强行裁剪
候选或继续执行失败解。该求解器不提供不可行性数学证书，因此失败状态称为
“没有被接受的候选”，而不是断言该物理问题无解。

## 固定支撑 CPU 闭环

每步从当前实际 q/v 重新构建 WBC。`FixedSupportCPU` 只接收力矩和可选外力，
重新用原始 Pinocchio 计算质量矩阵、非线性项、接触雅可比与偏置，解独立稠密 KKT，
然后以 dt=0.005 s 积分得到下一状态；不会把 IK 或 W 预测姿态写入实际状态。
CPU plant 使用相同刚性点足/接触稳定假设，不是 PhysX。

本机五个场景：静态保持、共同参考往返、W→Guidance→WBC、5 N 世界 x 方向
推力持续 0.1 s、在 0.65 s 使参考过期并返回站立。任务为 TCP 前伸 1 cm 与侧移
2 cm 的组合往返，2 s 运动加 0.5 s 后续执行；每例 500 个控制步。
外力只加到 plant，WBC 不读取该外力。

记录实际 TCP/姿态误差、足漂移/速度、关节/力矩/力约束违反、独立动力学残差、
QP 残差、末端速度、参考模式和全部失败。名义加速度一致性仅在没有外力的步检查。
末端速度门槛为 0.05（广义速度各分量的数值上界，分别具有 m/s 或 rad/s 单位）。

仿真积分频率为 200 Hz，实际执行是离线运行。求解耗时和包含参考采样、审计及
plant 的整周期耗时分别报告，包含首次调用、P50/P95/P99、最大值和 5 ms 超时计数。
**当前没有通过 200 Hz 实时验收。** 不用仿真 dt 倒数代替计算性能。

## 复现与结果

不安装或升级依赖，使用现有隔离的 Python/NumPy/Torch/Pinocchio/Crocoddyl：

```bash
(
  export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
  export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1
  export PYTHONPATH=.dev-tools/crocoddyl321-pin400/cmeel.prefix/lib/python3.11/site-packages:src
  .venv/bin/python -B examples/go2_z1_wbc.py
  .venv/bin/python -B -m pytest -p no:cacheprovider -q \
    tests/test_guidance.py tests/test_whole_body_qp.py
)
```

原始参考/实际状态/控制/力/外力轨迹和逐步 QP 诊断位于
`artifacts/go2-z1-wbc/`，汇总、哈希、测试与隔离检查见
[本机基线记录](benchmarks/go2_z1_wbc_2026-11-13/README.md)。
只运行 `--cases hold reference push expired` 时无需构造 Crocoddyl 规划器。
