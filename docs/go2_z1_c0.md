# Go2–Z1 C0 质心规划与初步 CPU 对照

2026-11-14，开发起点为 `cd00a27ecf736bf0856ad21d299aa5e0ef929ade`。
本轮新增独立 C0 模块和实验入口，复用已经冻结的共同参考、Guidance、接收器、WBC
和固定接触 CPU plant。冻结 W 的全部源码/资产/依赖/证据哈希保持不变，单 Go2 训练
路径和环境没有改动。C1、PhysX、载荷、碰撞、行走以及硬实时验收不在本轮结论内。

## 模型和求解

`models/centroidal.py` 的状态为世界坐标下 `[c, l, k]`：总质心位置、线动量和关于
质心的角动量，共 9 维。控制是 FL/FR/RL/RR 四足世界接触力，共 12 维，**不是关节
力矩**。总质量来自同一个 Go2–Z1 CPU 模型（21.60796983 kg），包括机械臂、固定夹爪
和支架；不重复添加机械臂重力或手端外力。

连续模型：

```text
c_dot = l / m
l_dot = m g + sum_i f_i
k_dot = sum_i (p_i - c) × f_i
```

接触位置 `p_i` 在一个 episode 内固定。每步在常力假设下精确积分 COM/线动量，
角动量使用步首力矩的显式 Euler；`dt=0.05 s`，24 步、1.2 s 时域。这是明确的离散
预测模型，不声称是连续角动量的精确积分。非线性项是 `-c × sum(f)`，A/B 为解析导数，
由有限差分和坐标平移不变性测试核对。

`planners/standing_centroidal.py` 实现顺序凸 QP：每轮围绕上一条非线性 rollout
线性化，消去线性状态递推，保留 288 个力决策变量。复用现有 Torch float64 内点 QP，
不引入新求解器依赖。目标包含 COM、线速度 `l/m`、角动量、接触力先验及相邻力变化。
角动量目标为零；C0 不利用未来全身构型的惯量或质心动量矩阵。

| C0 参数 | 值/解释 |
| --- | --- |
| 物理接触界 | `|fx|+|fy| ≤ 0.6 fz`，`2 ≤ fz ≤ 200 N`，与 WBC 的接触条件一致 |
| 力变化率 | 每个世界分量 ≤ 400 N/s；首步相对于当前接触力估计 |
| COM 区域 | 初始总 COM 的 xyz 各 ±0.08 m；整个 episode 固定，不随实测误差移动 |
| 绝对 COM 高度 | 相对足端平面 0.15–0.65 m，与上面的移动区域取交集 |
| 信赖域 | 每个未来 COM 分量 ±0.02 m，拒绝不良步骤后缩半，最小 1e-5 m |
| 状态权重 | COM 10000、线速度 100、角动量 100；各项带 dt，终端乘 5 |
| 力权重 | 力先验 1e-4、相邻力变化 1e-3；各项带 dt |
| 求解预算 | 最多 8 轮 SCP；必要时另做 1 次初值恢复 QP；每次 QP 最多 60 轮 |
| 数值门槛 | QP KKT 1e-8；初值/非线性动力学 1e-8；物理界 1e-6；线性化缺陷 1e-6 |

这些是本轮 C0 模型的显式假设和参数，不是对冻结 W 或 WBC 的修改。COM 区域只是该
固定支撑任务域，不能替代全身关节、力矩、碰撞或可达性证明。

移位力初值在新的实测状态上重新 rollout；如果不满足硬约束，先用不含**算法信赖域**
的初始化 QP 恢复可行性，物理力/变化率/COM 界完整保留，再进入正常信赖域 SCP。
恢复阶段不作为 SCP 收敛。独立记录恢复次数、SCP 轮数和全部 QP 内迭代/耗时。

每个普通 SCP 步骤比较实际非线性目标下降与线性化预测下降；不良步骤缩小信赖域。
最终逐节点及终端重算原始离散非线性方程、初始条件、摩擦、法向力、力变化率和 COM
界；线性 QP 的成功标志不代替验收。有限预算内通过原始门槛的候选可标为
`MAX_ITERATIONS` 并继续使用，收敛与可行性分开报告。QP 失败不构成全局不可行性证明。

## 共同参考、Guidance 与实际执行

`references/centroidal.py` 使用同一共同参考的 `c_ref` 和 `m * v_com_ref`，名义力采用
共同重力先验。实测 `(q,v)` 通过 Pinocchio 读取当前 COM 和世界动量；这一步是当前
状态观测，不是对未来构型的 C1 增强预测。

C0→Guidance 只替换 COM 位置/速度/加速度与接触力。TCP、基座和名义关节参考完全
保留共同生成器的值。现有 Guidance 没有角动量任务，C0 的 `k` 用于预测、代价和诊断，
不通过假定基座惯量把它变成姿态监督。导出前再次检查模型身份、时域、测量初值和
完整非线性候选；终端加速度采用末段保持力。

`C0WorkerEngine` 接受与 W 相同的实测快照和实际上一力矩。当前接触力由独立 CPU KKT
根据这些反馈量估计，使用冻结接触稳定化参数；估计步骤返回的下一状态被丢弃，不能
写回 plant。该估计不含未来关节/力矩可行性，也不假定能观测未知外力。
仅接收器 ACK 的 C0 计划能够用于移位热启动，过期初值丢弃。恢复预览、旧包沿用、
过期回站立、C2 交接和 WBC 失败终止均复用冻结路径。

实际状态由同一个 WBC 输出的 18 关节力矩驱动独立 KKT plant 推进；C0 的预测状态或
共同 IK 姿态不会被当作实际状态。C0 只证明质心模型内的可行性，全身力矩/关节/接触
可行性与 TCP/姿态误差继续由原 WBC 和 CPU 闭环验收。

## 对照范围

入口 `examples/go2_z1_c0_compare.py` 先验证冻结清单，再运行静态保持、小幅往返、
高低目标、四次连续规划失败、过期包五个任务。两种方法使用同一个生成好的参考、
相同初始状态、1.2 s 时域、2 Hz 请求、20 ms 名义 WBC 周期、≤5 ms plant 子步，
以及同一套增益、动作/约束门槛、回退和恢复策略。包内全部共同参考字段及名义状态
一起参与哈希核对。

这是**给足计算预算的模型质量初步对照**：`ScheduledC0Worker` 与冻结
`ScheduledWWorker` 都实际执行原生求解、Guidance、WBC 和动力学；交付时间统一规定为
250 ms，前台采样前/求解后时间分别为 2/8 ms。计时模型与原生求解耗时分开报告，
不能据此声称 C0 已完成后台 spawn、真实墙钟 deadline 或硬实时验收。未知延迟下的
性能及同墙钟计算预算对照仍待后续实验。跨模型代价函数和迭代含义不同，不把它们的
原始数值直接排成优劣。

最终五组配对（10 个回合）均通过原闭环/故障策略验收，每法 66 次请求、61 包接收。
完整相关回归 211 通过、1 CUDA 跳过；36 项为新增 C0 测试。当前 C0 请求耗时中位数
178–259 ms，W 为 159–171 ms，未体现速度优势。

完整场景/性能表与反馈核对见 [验收归档](benchmarks/go2_z1_c0_2026-11-14/README.md)。
冻结 W 的数学问题、增益、门槛和既有失败记录不因本轮对照而改变。

## 首次失败与修正

最早的 hold 控制回合通过，但报告打印遇到 NumPy `bool_` 无法直接 JSON 编码，脚本
未完成其余任务。保留 partial JSON 和异常日志；修复为使用已有 `json_safe`。

首个完整配对批次，W 五例全通过，C0 高低例失败：接收 19/24 包，TCP 峰值
6.8326 mm > 原 5 mm 门槛。分析发现初版把 COM 的 z 范围设为 ±5 cm，而共同高低
任务的**总质心**变化为 -59.127/+33.526 mm。COM 与基座平移是不同物理量，W 的
基座界不能直接充当 C0 的 COM 界。现明确使用上表的 ±8 cm COM 区域和绝对高度界；
原始执行验收界没有放宽。另有移位 rollout 越界后信赖域与全局区域无交集的问题，
通过独立的可行初值恢复步骤处理，不修改物理约束或伪造 QP 成功。

专项高低复测 24/24 接收，TCP 峰值 0.4360 mm、姿态峰值 0.005052 rad。
首次失败、部分运行和复测单独归档，不将它们覆盖成最终结果。最终五任务没有触发
初始化恢复 QP；该分支由显式不可行移位初值的专项测试覆盖。

## 复现

使用已有环境，不安装依赖；输出到新目录保留前一轮结果。

```bash
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONPATH=.dev-tools/crocoddyl321-pin400/cmeel.prefix/lib/python3.11/site-packages:src
.venv/bin/python -B examples/go2_z1_c0_compare.py \
  --output artifacts/go2-z1-c0-comparison-rerun
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -B -m pytest -p no:cacheprovider -q \
  tests/test_centroidal.py tests/test_centroidal_guidance.py \
  tests/test_async_timing.py tests/test_async_cpu.py tests/test_rolling_cpu.py \
  tests/test_stance_demo.py tests/test_guidance.py tests/test_whole_body_qp.py \
  tests/test_standing_whole_body.py tests/test_go2_tensor_adapter.py \
  tests/test_walking_whole_body_demo.py tests/test_w_workspace.py
```

下一步是 C1 的未来构型相关动量映射和准静态能力检查，并继续复用同一 Guidance/WBC。
相同实际墙钟预算、独立带臂执行接口和正式物理配对实验仍需另行验收。
