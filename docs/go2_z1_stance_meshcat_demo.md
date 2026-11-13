# Go2–Z1 固定支撑高低目标 Meshcat 闭环演示

2026-11-13。演示执行链为共同运动学参考 → W 单次全时域规划 → Guidance →
统一 WBC → 18 关节力矩 → 独立 Pinocchio 接触动力学 → 实际状态反馈。
Meshcat 展示该执行状态的 FK，W 的预测状态和 IK 名义姿态另存用于诊断。

## 观看与复现

```bash
.venv/bin/python -B examples/run_go2_z1_stance_meshcat.py serve
```

打开 <http://127.0.0.1:8009/>。独立 HTML 为
`artifacts/go2-z1-stance-meshcat/index.html`，内嵌完整官方网格、Meshcat viewer 和动画，
可离线打开。支持播放/暂停、时间轴、调速、中位/低位/高位/返回跳转及侧视；
默认暂停，点击“播放”开始，避免打开页面后持续占用渲染资源。
宽窗口附带世界 XZ 轨迹和四足实际法向力。绿色支撑力标记按 0.002 m/N 显示。
页面同时报告实际 TCP 误差、机身高度、俯仰和力矩使用率。

重新求解及导出使用已有环境，不安装依赖：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -B examples/run_go2_z1_stance_meshcat.py solve
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -B examples/run_go2_z1_stance_meshcat.py export
```

求解读取已有 `.venv` 和 `.dev-tools/crocoddyl321-pin400/cmeel.prefix/...`；
导出额外读取已有 `.cache/go2-z1-viewer-python` 与基础 Python 的 transport 依赖。
不将 viewer 依赖传入控制求解路径。模型/安装参数、官方资产来源与固定版本沿用
[Go2–Z1 网格记录](go2_z1_meshcat_demo.md)。26 个机器人外观对象使用完整官方 DAE，
另有项目定义的安装支架；不抽稀网格或用简单几何替换机器人。

## 与论文 Fig. 3 的对应

灵感来自 [Whole-Body Inverse Dynamics MPC for Legged Loco-Manipulation，Fig. 3](https://arxiv.org/html/2511.19709v1#S6.SS2)：
手端访问高低目标时，躯干与腿协同调整。原论文使用 B2–Z1 小跑和直接力矩 MPC；
本演示为 Go2–Z1 固定四足、W→Guidance→WBC 的独立实现，不宣称复现原论文控制器
或小跑实验性能。保留全部 18 个驱动关节，无额外物体载荷。

## 任务与数值配置

- 演示专用初始臂姿 `[0,1.1,-1.1,0,0,0] rad`，模型默认初始姿态保持不变。
- 12 s 目标序列：0–1 s 中位；1–3.5 s 前伸 6 cm、下降 20 cm；3.5–4.5 s 低位保持；
  4.5–7.5 s 到初始上方 12 cm；7.5–8.5 s 高位保持；8.5–11 s 回中位；11–12 s 保持。
- 各段使用五次曲线，停留边界速度和加速度连续且为零。手端世界朝向保持初始值。
- 共同 IK 可选开放基座朝向，平移边界 ±[0.08,0.06,0.075] m，旋转向量范数上界
  0.3 rad。任务仅指定手端轨迹；身体动作由加权 IK 和 W 任务共同决定，不预编身体动画。
  它依赖共同运动学先验，不宣称是论文意义下纯 MPC 自发产生的同等行为。
- W 为 240 节点、dt=0.05 s、最多 150 次迭代；使用同一动力学、代价和逐节点验收。
  长时域使用共同参考姿态与逐节点静态支撑力矩作有 gap 初值；FDDP 负责闭合 gap。
  仅增加可选 `initial_guess`，未提供时保留原静态力矩 rollout 初始化。
- Guidance 保留共同 TCP/名义关节参考，使用 W 的质心、基座与力预测；不执行 W 力矩。
- WBC/CPU 积分 dt=0.005 s；目标包 12 s 过期后用 1 s 回到初始站姿，再保持 0.5 s。
  全程 13.5 s、2700 个力矩控制步，显示按实际状态降采样至 40 Hz。

共同生成器新增 `allow_base_rotation`、基座范围与运动正则权重设置，默认仍保持
原固定朝向行为。世界角速度/角加速度通过 SO(3) 左雅可比从旋转向量导数转换；
名义状态中的浮基速度转回 body/local 表达。旧单 Go2 控制、adapter、训练入口、
依赖声明与环境均不修改。

## 验收与范围

W 数值收敛，独立动力学/接触/状态/力矩门槛通过；2700 步 CPU 执行全部通过。
手端高度约 0.669–0.989 m，机身高度约 0.254–0.319 m，俯仰约 −0.96°–7.39°。
TCP RMSE 约 0.0482 mm、峰值约 0.0964 mm；足漂移最大约 0.00416 mm；
力矩、关节位置/速度、摩擦与法向力违反量为零，最高力矩使用率约 27.62%。
这些是相同刚体模型内的 CPU 数值结果，不能作为硬件或 PhysX 精度结论。

本轮相关测试 84 passed、1 skipped；跳过 CUDA adapter 非默认 stream。
新增 8 项测试覆盖高低任务 C2 连续、姿态导数/坐标、固定足锚点、可配置运动范围、
带 gap 初始化与独立 FDDP 验收。旧 W、共同参考/WBC、行走 demo 和 Go2 adapter 回归通过。

在仓库根目录用同一套已有环境复现相关测试：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 \
PYTHONPATH=.dev-tools/crocoddyl321-pin400/cmeel.prefix/lib/python3.11/site-packages:src \
  .venv/bin/python -B -m pytest -p no:cacheprovider -q \
  tests/test_stance_demo.py tests/test_guidance.py tests/test_whole_body_qp.py \
  tests/test_standing_whole_body.py tests/test_go2_tensor_adapter.py \
  tests/test_walking_whole_body_demo.py
```

当前整周期计算耗时仍超过 5 ms，200 Hz 仅为积分频率，未通过实时验收。
在线滚动 MPC、接触切换 WBC、碰撞检测、Isaac Lab/PhysX 和实机仍待验证。
导出器拒绝失败报告、模型/轨迹哈希不一致、缺失网格和 TCP 与实际状态 FK 不一致。
本地 `trace.npz` 保存实际/规划/名义状态、力矩、接触力、参考与足端轨迹；
`report.json` 保存全部逐步诊断，`viewer.json` 保存资产和输出校验值。

汇总、日志和代码/依赖隔离检查见
[本轮记录](benchmarks/go2_z1_stance_meshcat_2026-11-13/summary.json)。
