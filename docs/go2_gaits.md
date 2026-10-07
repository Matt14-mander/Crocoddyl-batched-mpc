# Go2 的四种 Crocoddyl 示例步态

将 Crocoddyl 的 `SimpleQuadrupedalGaitProblem` 四种步态适配到项目固定的官方 Go2 URDF。
保留示例步序与首次半步规则，添加真实平地落足、世界坐标锚点、完整支撑集碰撞和
显式执行门禁。原示例工具来自 [Crocoddyl](https://github.com/loco-3d/crocoddyl)，
属于示例辅助类；本包将依赖集中在离线适配器，并以数值测试核对其行为。

| 步态 | 摆动足顺序 | 摆动时支撑 |
| --- | --- | --- |
| walk | RR → FR → RL → FL | 三足 |
| trot | FR+RL → FL+RR | 对角两足 |
| pace | FR+RR → FL+RL | 同侧两足 |
| bound | FL+FR → RL+RR | 前后两足 |

`FL/FR/RL/RR` 分别为左前、右前、左后、右后。
walk 首轮 RR/FR 半步，trot/pace 首轮第一组半步；bound 全步。
默认步长 4 cm、抬足 2 cm、dt=20 ms，摆动 20 节点、四足支撑 10 节点，
每轮增加 20 节点的四足 COM 目标收敛阶段。后续轮次以实际优化终态重新建模，使用全步。
旧慢速 walk 的 RL→FL→RR→FR 步序、COM 三角形转移和接触等待接口继续保留。

## 运行与导出

离线规划需要 Crocoddyl 和 Pinocchio，使用用户 croco_env：

```bash
PYTHONPATH=.dev-tools/croco-numpy126:src OPENBLAS_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 \
  /Users/zhengyuanhao/anaconda3/envs/croco_env/bin/python \
  examples/go2_quadrupedal_gaits.py --gait all --cycles 2
```

`--gait walk|trot|pace|bound|all` 选择步态，`--plant pinocchio|torch` 选择 plant。
两种示例都逐周期使用独立 native 动力学核对 Torch 一步输出；Native plant 从实测状态
和实际反馈扭矩推进。没有姿态覆盖或人工落足速度注入。
`--save-dir /tmp/go2-gaits` 导出每种步态的 `.pt` 文件，只包含 Tensor 和基础元数据，
不序列化 native solver。规划参数可通过 `--step-length`、`--step-height`、`--dt`、
`--step-knots`、`--support-knots`、`--iterations` 设置；改变参数后必须重新检查可执行性。

```python
import torch
from crocoddyl_batched_mpc.models import Go2GaitController, load_go2_gait_plan

torch.set_num_threads(1)
plan = load_go2_gait_plan('/tmp/go2-gaits/trot.pt', device='cpu')
controller = Go2GaitController(plan, batch_size=2)
state = plan.states[0].expand(2, -1).clone()
control = controller.compute(state)
# 先检查 control.result.usable，再执行实际扭矩或零时长碰撞。
# 仿真器以实测状态独立积分/投影；不要用参考配置更新 plant。
predicted = control.result.xs[:, 1]
```

载入和执行导出的 plan 只依赖 Torch，不需要 NumPy/Pinocchio/Crocoddyl；可用
`plan.to('cuda')` 移动 Tensor。native 对象不会进入执行循环。
本机没有 CUDA，相应测试尚未运行。

## 优化与硬约束执行

离线采用原生 `SolverBoxFDDP`，URDF 扭矩上下界进入优化器。
使用逐阶段固定世界锚点，接触位置/速度稳定增益为 100/50；每次落足更新新落足足的
锚点，保留足锚点继续固定。后续四足模型不复用初始阶段的旧锚点。
落足目标 z=0；碰撞同时投影新足和保留足，避免碰撞后原支撑足产生分离/穿地速度。
碰撞节点的物理时长为零，执行数据中的扭矩/增益占位为零。

力、冲量与地面软 barrier 用于引导优化，分阶段提升惩罚强度，碰撞显式开启冲量 Jacobian。
它们不构成可行性证书。导出前重新 rollout，清除可能残留的 FDDP 动力学间隙，
并在导出状态上重新计算反馈增益；`plan.converged` 单独报告各轮是否达到优化停止阈值。
预算耗尽可以产生可执行轨迹，也可以产生被执行器拒绝的轨迹。

在线使用流形误差和 Crocoddyl 增益 `u=uff-K*dx`。
如果反馈扭矩违反当前力/扭矩界，求解到目标反馈扭矩的最小距离 QP，使用实测状态下的
精确控制仿射接触力映射。没有最终动作裁剪。所有输出再次检查：

- URDF 扭矩限制、法向力 0–200 N、`abs(fx)+abs(fy)<=0.6*fz`，非支撑力为零。
- 观测/预测状态和关节范围、足端不低于 -2 mm。
- 落足位置与目标锚点误差不超过 2 mm、新足接近方向正确。
- 全部目标支撑冲量的单边/摩擦不等式、碰撞后接触速度残差、动能不增；容差 1e-7。

失败返回不可用结果和 NaN 动作，停止该环境推进并保持失败状态，直到 `reset(mask)`。
其他环境可继续。重置控制器时，调用者需同步将指定 plant 重置到 `plan.states[0]`。
这种动态步态不使用慢速 walk 的静态三角形离地门禁或触地相位等待。
有限轨迹结束后 `finished=True`，调用者需停止该序列或交接其他控制器；继续调用不提供保持扭矩。

## 验证与适用边界

四种步态均有独立 native/Torch 状态、接触力和冲量对照，atol/rtol=1e-9，
零时长碰撞保持配置完全不变。测试覆盖双环境反馈前进、物理门禁、选择性重置、
安全 Tensor 导出加载，以及冲量 barrier Jacobian 的独立有限差分（1e-8）。
详细结果见 [验证记录](validation.md)。

这是 **Crocoddyl CPU 离线接触序列优化 + Torch 批量反馈/约束执行**。
当前尚未实现每周期重求整段步态的在线滚动非线性 MPC、任意时刻步态切换、
速度/转向指令、地形自适应或硬件/Isaac Sim 联调。规划时间不计入在线控制延迟，
控制延迟也不含传感器、通信和 plant；不能据此保证真实机器人全链路 50 Hz。
