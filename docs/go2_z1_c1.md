# Go2–Z1 C1 构型感知质心规划与 CPU 对照

开发起点 `d20ff6c5255d93a5bc7aeaf5c6148e9417920aed`。本轮全部控制实现为新增模块，
复用 C0 的质心动力学/SCP，冻结 W、共同参考、Guidance、WBC、plant、旧单 Go2
训练入口和依赖均不修改。实验记录及状态见
[C1 验收归档](benchmarks/go2_z1_c1_2026-11-14/README.md)。

## 模型增量与边界

C1 仍优化 `[c,l,k]` 和四足世界接触力；控制不是 18 个关节力矩。
在共同运动学预览的每个未来 `(q_ref,v_ref)` 上，由 Pinocchio 计算世界坐标下、
关于总质心的 `Ag(q)`，并按 FL/FR/RL/RR/Z1 顺序排列速度列。基座速度仍为局部
线/角速度，动量排列为 `[linear,angular]`，角动量单位 kg·m²/s。

```text
[l_ref, k_ref] = Ag(q_ref) v_ref
c_ref = COM(q_ref)
Ag = [Ab, Aj]
v_base_pred = inv(Ab(q_ref)) ([l,k] - Aj(q_ref) v_joint_ref)
```

与 C0 的零角动量参考不同，C1 参考包含未来机械臂/腿/基座运动产生的角动量。
每个未来节点约束映射后的局部基座线速度分量 ≤0.5 m/s、角速度分量 ≤1 rad/s，
并在原始候选验收中复核初始与终端节点。基座矩阵条件数大于 1e8 时拒绝。
这不是常量基座惯量或只计算机械臂的近似映射。

`q_ref,v_joint_ref` 是共同预览，**不是 C1 的新增优化变量**。C1 没有积分预测基座
姿态/关节运动，也不保证映射后的速度同时满足足端速度和 TCP 运动学；这些仍由
WBC 完整动力学执行与 CPU 闭环门槛验收。映射随每次滚动的未来构型重新计算，
但不对一个时域内优化后的 COM 再联合求解全身构型。因而这是构型条件化的质心
近似，不能代替 W 的全身状态/动力学预测。

C0 的 24×0.05 s 时域、状态/力权重、COM 区域、信赖域、力变化率、SCP/QP 预算
以及原非线性验收均沿用。没有修改 C0 或 W 的数学问题来迁就 C1。

## 准静态能力与力矩包络

每个未来构型（含终端）先检查完整 24 自由度重力平衡：

```text
g(q) = S tau + Jf(q)^T f
|fx| + |fy| <= 0.6 fz,  2 <= fz <= 200 N
|tau_j| <= model_torque_limit_j
```

求解器消去浮基 6 维平衡等式，优化 12 个支撑力与最大归一化力矩变量 `t`。
约束 `|g_j - J_j^T f| <= t tau_limit`，最小化
`t + 0.5e-6 t² + 0.5e-8 ||f||²`，得到超限评分与静态支撑分配。
接收仍要求真实力矩界通过；增大 `t` 不是放宽物理验收。
QP 最多 60 轮、容差 1e-8；全部原始平衡、摩擦、法向力、力矩、关节界与
固定足位置再次检查，物理残差容差 1e-6、足位置 1 mm。
`QUASI_STATIC_LIMIT` 与 `NO_ACCEPTED_CAPABILITY_QP` 分开，不把数值失败当作
全局不可行性证明；足位置/关节几何不合格单列 `REFERENCE_GEOMETRY_INVALID`。

C1 的规划 QP 还对**实际规划的力**加入每节点
`|g_j(q_ref) - J_j(q_ref)^T f_plan| <= tau_limit`；终端使用末段保持力。
浮基静态平衡仅用于参考承载能力检查，不强加到运动中的预测力上，否则会消除
质心加速度。上述力矩包络忽略惯性/科氏项；不能把它称为动态力矩证书。
最终执行力矩由原 `WholeBodyQP` 求出，包含完整惯性、接触与硬力矩约束。

静态检查支持 TCP 局部偏置处的点载荷重力探针：将 `J_tcp^T w_payload` 从所需
重力广义力中减去一次。模型自身机械臂、夹爪和支架质量已经计入，不重复添加。
该参数只用于静态能力/重分配专项；本轮 C1/C0/W 动态闭环均为空载，原 CPU
模型仍拒绝新增动态 payload。此探针不代表真实 Z1 额定载荷、抓取、碰撞、惯量
或热力矩能力认证。

## 共同参考与姿态重分配

`prepare_capable_reference` 接在共同运动学生成器之后；已经可行时原对象和全部
参考字段原样返回。若未来节点超出能力，在共同参考允许的基座/关节范围内做
确定性任务零空间搜索：保持四足锚点和 TCP 位姿，对每个试探解重新做非线性
运动学投影及完整静态检查。初始实测姿态不可改。仍不能接受时返回带节点、
搜索次数和计时的明确失败；搜索预算耗尽不等价于全局不可行。
原共同任务若禁止基座旋转，重分配也保留其固定朝向。

重分配后重新计算 COM、基座和关节导数；超过速度界的序列拒绝。该处理是离线
共同参考准备，调整后的同一个 `CommonReference` 必须交给 C0/C1/W 三组。
滚动 C1 worker 不私自改变名义关节/基座参考：若实际滚动/恢复预览静态检查失败，
该包拒收并进入原有旧包沿用/平滑回站立机制。三组首轮对照禁用姿态重分配，
验证原共同任务均有静态能力；重分配另列专项，不与模型质量收益混算。

参考调整只保证采样节点的运动学/静态能力；Guidance 节点间插值和动态力矩还需
执行层检查。运动学搜索、动量映射、静态 QP、规划/验收与导出分别计时。

## Guidance、滚动与真实执行

`C1WorkerEngine` 使用当前实测 `(q,v)` 计算初始 COM/动量，以实际上一 WBC 力矩
估计当前支撑力。只用接收器 ACK 的计划移位热启动；初始化重置为实测状态。
包交接、超期、连续故障、回退、恢复和前台 WBC 都复用冻结执行路径。

C1→Guidance 与 C0 一样只提供 COM 位置/速度/加速度和接触力；TCP、基座和
名义关节字段保持共同预览。未来 `Ag` 影响参考和规划约束，映射基座速度不另作
WBC 监督任务。导出再次计算全部未来 `Ag`、重力、接触 Jacobian，复核静态支撑
与原始规划候选，拒绝陈旧/篡改构型、映射、静态力或模型身份。
预测构型和质心状态不写回 plant；实际轨迹由 WBC 的 18 关节力矩推进。

`ScheduledC1Worker` 与 C0/W 配对入口使用相同规定交付时间；原生求解确实运行，
其耗时另报。此实现是给足预算的模型质量 CPU 对照，尚未完成 C1 后台 spawn、
同实际墙钟预算或硬实时验证。不能将规定的 250 ms 交付等同于实测计算延迟。

## 本轮结果

保持、伸手、高低目标、连续规划失败、过期包五任务，W/C0/C1 首批 15/15 通过；
边界复核后的 C1 五例复测通过，轨迹逐元素一致。高低任务 TCP 峰值
W/C0/C1 为 0.448393 / 0.435975 / 0.435977 mm，均保持原 5 mm 门槛。
每法 66 次请求、61 包接收；C1 的 62 个实际候选收敛，1550 个静态节点通过。
完整相关回归 242 passed / 1 CUDA skipped，其中新增 31 项 C1 测试。

C1 原生完整请求各任务 p50 约 517–624 ms，C0 175–235 ms，W 143–165 ms。
当前 C1 无速度优势，且这组空载任务远离静态承载边界，没有出现可归因于 C1 的
明显执行改善。控制质量、静态能力、真实计算时间与规定交付时间分别报告。

62 次 C1 快照的状态、上一实际力矩、动量及独立 KKT 力估计最大差均为 0；
W/C0 与已有基线、C1 首批与复测共 180 组执行数组一致。303 个原跟踪文件仅更新
开发计划，其余 302 个与 88 个依赖元数据、冻结 W 全清单不变。
最初审计报告出现 ndarray 序列化错误，修正保存格式后复测成功；错误日志、
首批实验、复测及全部故障事件均保留，详见验收归档。

## 复现

使用现有环境，不安装/升级依赖，输出必须选新目录。

```bash
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONPATH=.dev-tools/crocoddyl321-pin400/cmeel.prefix/lib/python3.11/site-packages:src
.venv/bin/python -B examples/go2_z1_c1_compare.py \
  --output artifacts/go2-z1-c1-comparison-rerun
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -B -m pytest -p no:cacheprovider -q \
  tests/test_configuration_centroidal.py tests/test_configuration_guidance.py \
  tests/test_centroidal.py tests/test_centroidal_guidance.py \
  tests/test_async_timing.py tests/test_async_cpu.py tests/test_rolling_cpu.py \
  tests/test_stance_demo.py tests/test_guidance.py tests/test_whole_body_qp.py \
  tests/test_standing_whole_body.py tests/test_go2_tensor_adapter.py \
  tests/test_walking_whole_body_demo.py tests/test_w_workspace.py
```

后续仍需同真实预算/后台 C0/C1、独立带臂 adapter/Isaac Lab、带载动态模型与
碰撞检查、正式阶段 A 配对；固定支撑结果不等于行走 WBC 或硬件验收。
