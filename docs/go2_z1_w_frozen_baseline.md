# Go2–Z1 固定支撑 W CPU 冻结基线

2026-11-14；优化前版本 `5a16a96510bb9dd126583725a626787a2bebc01e`。
本轮限定于 W 问题构造、独立验收、Guidance 导出与前台包校验。数学问题、代价权重、
控制增益、约束、求解预算和验收门槛保持不变。基线 `W-CPU-2026-11-14-v1` 以
源码哈希与实验协议冻结，并随本次 Git 提交归档；未创建 release tag，也不代表完成
硬实时、PhysX 或实机验收。包含清单的提交可用
`git log -1 --format=%H -- docs/benchmarks/go2_z1_w_frozen_2026-11-14/manifest.json` 定位。

## 实现范围

- `CachedWholeBodyPlanner` 由每个 W worker 独占。缓存各节点接触/代价模型，逐次更新
  TCP 位姿/速度、COM、基座位姿、名义全身状态及相应关节/速度 barrier。
  设置、锚点、静态力矩或力矩界变化时清空模型缓存。每次仍创建新的 ShootingProblem、
  原生计算 Data 和 Box-FDDP solver，不继承求解器的隐藏正则化或搜索状态。
- W 参考的 SE(3) 检查批量执行；形状、有限性、每个四元数的原始点积检查和所有容差保留。
  独立验收复用私有 Pinocchio Data 与本次计算出的 base/TCP FK；每个节点及终端的
  KKT、动力学残差、接触约束、状态衔接、力矩、姿态和跟踪门槛继续逐项检查。
- `WholeBodyGuidanceExporter` 只计算导出和复核需要的量，复用私有 Data；不再调用包含
  独立重力、COM Jacobian、TCP Jacobian 等未用输出的完整 `evaluate`。
  每个节点仍重新检查 `M a + h - S tau - Jᵀ f` 和足端锚点，保留原来的
  `1e-6` 动力学残差与 `1e-3 m` 锚点门槛，以及相同的 canonical 求和顺序。
- `guidance_batch` 批量验证所有样本字段的形状、有限性和 SE(3)，然后建立只读、与
  发送端分离的数据；原 Guidance 包级校验继续执行。序号、时间、模型、来源、固定
  足端锚点、相邻旋转和有效期均不依赖发送端声明。保留原样本时间的 `1e-9` 容差。
- 接收端在校验完成后重新读取单调时钟，再检查原超时和状态年龄，并用该时刻进行
  buffer 交接。校验跨过 deadline 或有效期的包被拒收；门槛未放宽。

优化默认作用于实际异步 CPU worker 和接收路径。`optimized=False` 的 worker 选项
保留未缓存构造与旧导出，用于诊断；同名的 CPU episode 选项仍只控制上一轮 plant 优化。
同步/离线 W 会得到通用参考校验与 FK 复用的收益，但不自动使用 worker 私有模型缓存。
缓存内的原生 action 会随下一次问题构造更新，不能跨构造重算旧 ShootingProblem；
已返回的计划数组和 Guidance 自行拥有数据，可以继续保存或交接。worker 模型/拓扑在
一个 episode 内固定；需要更换模型时新建 worker。

## 数值与性能证据

[配对记录](benchmarks/go2_z1_w_frozen_2026-11-14/paired.json) 从本地 Git 读取优化前
planner，在同一进程交替运行两版。两个任务（平缓 reach、高低目标）各四个滚动窗口，
覆盖首包、ACK 移位初值和超出旧时域后重新构造初值。

8/8 对照通过：全部 144 组轨迹/Guidance 数组逐元素相同，最大绝对差为 **0**；代价、
迭代次数、收敛标志、所有非计时验收指标及逐节点失败列表一致。与动手修改前另存的
144 组数组也完全一致。这里的精确相同是当前固定原生栈和主机的实测结果，不是跨平台
浮点位相同的承诺。另与上一轮确定性 CPU 闭环的八份原始轨迹核对，
全部数组也逐元素相同，覆盖 W→Guidance→WBC→plant 和失败回退。

配对微基准预热后交替五组，每版构造/验收/导出各 25 次，包校验各 100 次；下面为中位数。

| 阶段 | reach：前 → 后（ms） | 高低任务：前 → 后（ms） |
| --- | ---: | ---: |
| W 问题构造（包含新建 problem Data） | 38.703 → 25.251 | 36.683 → 22.815 |
| 独立全节点验收 | 31.456 → 15.326 | 31.442 → 15.518 |
| Guidance 导出 | 37.381 → 10.316 | 38.347 → 10.689 |
| 前台样本/包校验与分离复制 | 10.612 → 1.751 | 10.806 → 1.815 |

微基准最后一行只测包数据校验，不含 IPC 传输及 buffer 的 C2 换包构造。
完整 worker 请求仍包含任务预览、初值、Box-FDDP 迭代及求解器初始化，不能把上述四项
收益直接相加当作闭环频率。保留首次冷构造成本，缓存不跨进程重启。

新增测试覆盖变化的全部目标、名义状态、初始实测状态、上一实际力矩、活动 barrier、
终端与运行节点的 cost/xnext/Fx/Fu/Lx/Lu/Lxx/Lxu/Luu，缓存失效、完整独立验收、
导出对坏终端力/加速度的拒绝、pickle 后修改的异常包、只读隔离，以及校验过程中
越过超时/状态年龄/过期边界的拒包。

闭环和延迟结果、完整测试计数见[验收归档](benchmarks/go2_z1_w_frozen_2026-11-14/README.md)。
历史失败与复测记录保留，不覆盖原有两轮证据。

## 冻结协议

以 [manifest.json](benchmarks/go2_z1_w_frozen_2026-11-14/manifest.json) 中的源码、资产、
依赖元数据、model identity、全部设置和原验收界定义本轮 W 基线。

- W：24 × 0.05 s = 1.2 s，2 Hz 请求，Box-FDDP 最多 100 次迭代，停止门槛 `1e-7`。
- Guidance：换包 0.4 s、回站立 1.0 s、恢复 0.8 s；默认交付超时 0.6 s、状态年龄
  0.65 s、硬超时 2.0 s。过期/重启注入场景沿用之前明确记录的专用超时设置。
- WBC：名义周期 20 ms；plant 最大积分步长 5 ms，补算上限 200 ms，命令年龄上限
  80 ms。所有 WBC 增益/权重以及 CPU 跟踪门槛与上一版相同。
- 保持公共 TCP 与名义关节参考，W 不额外使用优化关节轨迹作监督。
- 现阶段冻结的是这套软实时 CPU 实验条件。错过控制槽仍报告；不声称每个 20 ms
  deadline 都能满足，也不通过扩大超时或误差界来宣称通过。

下一步按同一共同参考、Guidance、WBC、模型与初始条件接入 C0，再接入 C1。
先给足求解预算比较模型/执行质量，再单独比较相同计算预算。新增方法不能静默改变
本清单中的 W 数学问题、增益、门槛；需要修改时建立新的有编号基线并重复配对与延迟验收。
独立带臂 adapter、Isaac Lab、载荷、碰撞和行走 WBC 继续独立排期。

## 复现

沿用已有环境，不安装或升级依赖。请使用新的输出目录保留前一次记录。

```bash
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONPATH=.dev-tools/crocoddyl321-pin400/cmeel.prefix/lib/python3.11/site-packages:src
.venv/bin/python -B examples/go2_z1_w_baseline.py \
  --reference-commit 5a16a96510bb9dd126583725a626787a2bebc01e \
  --output artifacts/go2-z1-w-paired-rerun
.venv/bin/python -B examples/go2_z1_async_cpu.py \
  --output artifacts/go2-z1-w-seven-rerun
.venv/bin/python -B examples/go2_z1_delay_regression.py --mode virtual --repeat 2 \
  --output artifacts/go2-z1-w-virtual-rerun
.venv/bin/python -B examples/go2_z1_delay_regression.py --mode wall --repeat 2 \
  --output artifacts/go2-z1-w-wall-rerun
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -B -m pytest -p no:cacheprovider -q \
  tests/test_async_timing.py tests/test_async_cpu.py tests/test_rolling_cpu.py \
  tests/test_stance_demo.py tests/test_guidance.py tests/test_whole_body_qp.py \
  tests/test_standing_whole_body.py tests/test_go2_tensor_adapter.py \
  tests/test_walking_whole_body_demo.py tests/test_w_workspace.py
```

`go2_z1_w_baseline.py` 只读取指定的本地 Git 版本，不切换工作区。归档内另有原始配对
数组 `paired_trajectories.npz`，键前缀 `0_` 为优化前、`1_` 为本轮。真实墙钟轨迹不会因
数学一致而逐元素一致；确定性虚拟时钟回归另外逐数组检查重复性。

在仓库根目录核对冻结文件（不运行控制器）：

```bash
python3 - <<'PY'
import hashlib, json
from pathlib import Path
manifest = json.loads(Path(
    'docs/benchmarks/go2_z1_w_frozen_2026-11-14/manifest.json'
).read_text())
for group in ('source_sha256', 'asset_sha256', 'dependency_metadata_sha256', 'evidence_sha256'):
    for name, expected in manifest[group].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
print('W baseline hashes match')
PY
```
