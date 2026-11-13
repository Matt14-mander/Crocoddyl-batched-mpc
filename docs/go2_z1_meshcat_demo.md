# Go2–Z1 行走 + 竖直 8 字 Meshcat 演示

这是独立的带臂 CPU 演示：Crocoddyl 规划完整浮基动力学和接触切换，再由
独立 Pinocchio KKT 积分器执行名义反馈回放。显示的是回放状态的 FK，IK
只用于求解初值。已有单 Go2 训练代码、依赖环境、站立 W 规划器均保持原样。

## 打开与重建

在仓库根目录运行：

```bash
.venv/bin/python -B examples/run_go2_z1_meshcat.py serve
```

打开 <http://127.0.0.1:8008/>。也可以直接用支持 WebGL 和 DecompressionStream
的浏览器打开 `artifacts/go2-z1-meshcat/index.html`。HTML 内含完整网格、
Meshcat JavaScript 和轨迹，不需要外网或运行求解器。支持拖动视角、播放/暂停、
时间轴、调速、正视曲线、Meshcat 原生控件和 YZ 轨迹小图。

重新求解和导出：

```bash
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -B examples/run_go2_z1_meshcat.py solve --iterations 250
.venv/bin/python -B examples/run_go2_z1_meshcat.py export
```

求解输出 `report.json` 和 `trace.npz`；导出输出 `index.html` 和 `viewer.json`。
求解器达到迭代上限时仍明确标记 `converged=false`；只有独立动力学与任务门限
通过才允许导出。退出码 0 表示这些门限通过，不代表求解器达到收敛阈值。
如需继续优化，可用 `solve --warm-start artifacts/go2-z1-meshcat/trace.npz`。

依赖来自已有 `.venv` 与 `.dev-tools/crocoddyl321-pin400/cmeel.prefix/...`，
即 NumPy 1.26.4、Pinocchio 4.0.0、Crocoddyl 3.2.1。不要把整个
`.dev-tools/crocoddyl321-pin400` 放进 PYTHONPATH，其中另有 NumPy。
Meshcat 0.3.2、u-msgpack-python 2.8.0、pyngrok 7.2.1 只安装到新建的
`.cache/go2-z1-viewer-python`。导出器将现有 venv 基础 Python 的 site-packages
追加到搜索路径末尾，只读取已有 ZMQ、Tornado、Pillow、IPython；不覆盖 venv
的 NumPy，也不向这些环境安装包。求解命令完全不加载 viewer 依赖路径。

若在新机器上准备网格，运行 `assets` 子命令，它从固定官方版本下载并校验：

```bash
.venv/bin/python -B examples/run_go2_z1_meshcat.py assets
```

## 模型与资产

使用现有 `go2_z1_fixed_gripper.urdf` 及其质量、惯量和 18 关节映射。
26 个机器人外观对象全部使用官方 DAE，另有 1 个项目自定义安装支架。
共 16 个不同的官方网格，不抽稀、不用基本几何替代机器人。

来源为 [Unitree 官方 unitree_ros 固定提交](https://github.com/unitreerobotics/unitree_ros/tree/5994d4faef0a9cadd3287f8de0199a67eeb2a259)：
Go2 的 7 个网格由 `examples/go2_z1_visual_meshes.json` 固定 Git blob SHA1，
Z1 的 9 个网格复用现有 `assets/go2_arm/SOURCE.json`。缓存的 `MANIFEST.json`
记录 Git blob SHA1 与 SHA256；导出时逐文件再验 SHA256。
`viewer.json` 保存资产、代码、HTML 与轨迹哈希。HTML 内保留 Unitree BSD-3-Clause
和 Meshcat viewer MIT 许可。

沿用的**仿真假设**：Go2 base 到 Z1 根部平移 `[0.05, 0, 0.10] m`、旋转为零；
支架质量 0.30 kg，尺寸 `[0.16, 0.09, 0.04] m`，COM 为 base 下
`[0.05, 0, 0.08] m`；TCP 是 `z1_gripperStator` 的局部 +x 方向 0.10 m
虚拟点；夹爪固定，无额外负载。这些不是实机安装/TCP 标定。

该任务将初始臂姿设为 `[0, 1.6, -1.6, 0, 0, 0] rad`，以保持世界固定绘图
平面的可达性；只作用于此演示，没有修改模型或站立任务的默认臂姿。

## 轨迹、动力学与验收含义

- 7.8 s、40 Hz；RR → FR → RL → FL 两轮，共 8 次迈步。
- 每次足端向前 8 cm、抬高 4 cm，摆动 0.5 s；相邻摆动间有 0.35 s 四足转移段。
- TCP 世界 x 固定；`y=y0+0.04 sin(2θ)`、`z=z0+0.06 sin(θ)`，是上下两瓣的
  竖直 8 字，宽 8 cm、高 12 cm；姿态保持初始世界姿态，θ 采用五次平滑时间标定。
- 浮基 q/v = 25/24，物理状态 49；另有 18 维上一时刻扭矩记忆，优化状态 67。
  连续段控制 18 维，零时长碰撞段控制 0 维，碰撞不重置扭矩记忆。
- 刚性点足、平地、预设接触顺序；接触 Kp/Kd 为 100/20，摩擦系数 0.6 的
  内接菱形摩擦锥，每个支撑足法向力 2–200 N；完全非弹性着地冲量。
- 独立回放从同一初始状态开始，使用 `u=u_plan-K·diff(x_plan,x)`，每一步重解
  Pinocchio 质量矩阵/雅可比 KKT，并做半隐式积分；没有逐帧赋值为 IK/规划状态。
- 数值门限包括：足端支撑位置 1 mm、速度 1 cm/s、TCP 峰值 5 mm、旋转 0.02 rad、
  力锥违反 1 mN、冲量锥违反 0.1 mN·s。所有实际残差完整记录。
  地面软代价允许 0.5 mm 离散漂移，独立几何门限为 1 mm，并非零穿透认证。

演示循环播放会回到初始位置。它不是周期稳态步态，末端状态没有单独的静止
终端约束；报告另列末端速度。反馈回放使用相同初态，因此只验证模型与实现
一致性，不能据此声称已验证扰动鲁棒性。

## 验收状态

当前可接受项：官方完整外观网格、世界坐标 TCP 曲线、接触切换、18 关节映射、
独立 CPU 动力学回放与数值任务门限。新增测试覆盖调度、参考曲线、冲量动量方程、
流动段 KKT、状态导数、扭矩记忆和终端/初始/非有限状态检查。

跳过：Isaac Lab / PhysX、CUDA 和现有单 Go2 训练执行。

待验收：严格优化收敛、滚动时域 MPC、带臂 WBC、碰撞检测、扰动鲁棒性、停止稳定、
实时性能与实机。当前演示是通用的行走画曲线任务，不宣称复现某篇论文的控制器
或其实验性能。最新数值与环境隔离记录见
[本次基线报告](benchmarks/go2_z1_meshcat_2026-11-13/README.md)。
