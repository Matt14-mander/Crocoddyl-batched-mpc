# Go2 平地慢速 walk

提供扭矩驱动的直行 crawl：默认步序 **RL→FL→RR→FR**，每次只摆动一条腿，
其余三足支撑。每足每轮前进 2 cm，摆足高度 15 mm，控制模型 dt=20 ms。
参考先将整机 COM 转移到剩余三足的支撑三角形内，再抬足、向前摆动和下降。
每轮结束后回到居中四足支撑；有限轮数结束后继续站立保持。

```bash
python examples/go2_slow_walk.py --cycles 2 --batch-size 2
# 可选独立 Pinocchio 刚体动力学、积分和碰撞投影对照
python examples/go2_slow_walk.py --plant pinocchio --cycles 2 --progress
```

Torch plant 无需 Pinocchio。初版使用 float64；用户 croco_env 的原生对照命令：

```bash
PYTHONPATH=.dev-tools/croco-numpy126:src OPENBLAS_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 \
  /Users/zhengyuanhao/anaconda3/envs/croco_env/bin/python \
  examples/go2_slow_walk.py --plant pinocchio --cycles 2 --progress
```

## 参考规划与在线控制

`go2_slow_walk_plan()` 在循环外生成有限轨迹、接触掩码及逐帧世界坐标锚点。
身体转移和水平摆腿采用五次平滑曲线；竖直摆腿曲线在两端具有零速度和零加速度。
基于 URDF 的腿部 IK 检查足端误差、可达性和关节范围。默认转移/摆动各 1 s、
落足后保持 0.4 s，默认两轮计划时长 21.6 s，触地等待会延长实际运行。
配置位置始终来自 plant 积分，参考只用于计算扭矩。

`Go2WalkMPC` 在当前实测状态下重算精确控制仿射映射：
`a=a0+Au*u`、`f=f0+Fu*u`。同一扭矩用于默认两节点短预测时域，
采用冻结当前动力学的局部运动学预测与凝聚 QP，包含状态及辅助 PD 加速度跟踪目标。
每周期更新动力学、接触掩码和锚点，共享编译图；无需逐步构建新的模式控制器。

QP 内约束 URDF 扭矩上限、支撑法向力 0.5–200 N、
内接摩擦棱锥 `abs(fx)+abs(fy)<=0.6*fz`，默认接触不等式余量 2 N。
非支撑足力严格为零。候选扭矩还必须通过完整非线性时域复核：实际接触力、扭矩、
观测和预测关节范围、足端地面容差及流形有效性。可复用旧扭矩时也重新执行全部复核。
状态/约束无效或没有可行候选返回不可用结果和 NaN 动作。
这仍是短时域局部约束控制，不是跨接触序列的完整非线性 SQP。

## 接触事件和批量生命周期

离地前要求实测 COM 在三足支撑三角形内，边界余量至少 2 cm。
触地沿用严格粘着塑性碰撞检查：平地高度与锚点接近、接近方向、单边/摩擦冲量、
接触速度残差和动能不增；需要拉力或滑动的解被拒绝，冲量容差为 1e-7 N·s。
事件通过后还需目标支撑集下的 MPC 可行。接受事件只修改碰撞速度和接触元数据，
不修改配置位置。落足锚点只在实际接受触地时更新，保留足锚点不动。

条件未满足时维持旧支撑集，停止该环境相位并跟踪静止的落足目标，默认最多等待 2 s。
超时/无可行控制后该环境进入失败状态；其他环境仍可推进。
各环境独立保存相位、锚点和 QP 缓存。`reset(mask)` 重置指定环境；调用者需同步
将对应 plant 重置到 `plan.reference[0]`。不要在运行中修改已准备的 plan。

```python
import torch
from crocoddyl_batched_mpc.models import Go2SlowWalkController, go2_slow_walk_plan

torch.set_num_threads(1)
plan = go2_slow_walk_plan(cycles=2)
controller = Go2SlowWalkController(plan, batch_size=2)
state = plan.reference[0].expand(2, -1).clone()
control = controller.compute(state)
# 先检查 control.result.usable；只执行可用动作。
# 仿真 plant 先应用已验证的碰撞后状态，再按当前支撑集进行扭矩积分。
state = controller.mpc.model.calc(
    control.state, control.action,
    contact_mask=control.contact_mask, contact_positions=control.anchors,
)
# 下一周期读取新测量；finished 后继续 compute 会保持最终站立。
```

## 验证与边界

独立 Pinocchio plant 已完成 B=2 的两轮连续前进，包含每环境 8 次离地/触地，
逐周期一步预测对照和每次塑性碰撞速度对照均使用 1e-9 容差。
实测前进约 4 cm、抬足约 15 mm；力和扭矩约束违反为零。
完整延迟和测试结果见 [验证记录](validation.md) 与
[CPU 测量](benchmarks/go2_slow_walk_cpu_2026-10-06.json)。

整段 walk 尚未通过 50 Hz 的 20 ms 验收，计时还未包含传感器、通信和实际机器人。
当前为非常保守的平地直行，小步距和 COM 转移导致平均前进速度很低；尚无速度/转向命令、
地形自适应、通用步态切换或真实硬件/Isaac Sim 验证。CUDA 用例已添加，本机没有 CUDA，
尚未执行。预测时域内固定当前接触集，事件由当前时刻的调度器验证提交。
