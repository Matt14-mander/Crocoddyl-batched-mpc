# Go2–Z1 共同参考 / Guidance / WBC CPU 基线

2026-11-13，前序工作基线为 `b1b290df14a66f704d8fca94c0f214c178928bbc`。
新代码由 [summary.json](summary.json) 的 `source_sha256` 精确标识。

## 结果

五类固定支撑 CPU 闭环均通过；每例仿真 2.5 s、500 个控制步，dt=0.005 s。
往返任务是在世界坐标系前伸 1 cm、侧移 2 cm；外力为基座原点世界 x 方向
5 N、持续 0.1 s；过期场景在 0.65 s 停止采纳参考并用 1 s 平滑回到初始站立目标。

| 场景 | TCP RMSE (mm) | TCP 峰值 (mm) | 整周期 P50/P95 (ms) | 结果 |
|---|---:|---:|---:|---|
| 静态持臂 | 0.000394 | 0.000417 | 9.978 / 12.373 | 通过 |
| 共同参考往返 | 0.033364 | 0.052094 | 9.734 / 12.520 | 通过 |
| W → Guidance → WBC | 0.037562 | 0.055075 | 9.940 / 11.529 | 通过 |
| 5 N 外力扰动 | 0.037096 | 0.072944 | 9.616 / 11.403 | 通过 |
| 参考过期回退 | 0.040082 | 0.089423 | 10.149 / 13.559 | 通过 |

全部场景的实际力矩、关节位置/速度、摩擦与法向力违反量为零。足端漂移最大
小于 1.5e-6 m，独立 plant 动力学残差小于 9e-14，名义步的 WBC/plant 加速度
差小于 3.6e-11。验收门槛、各任务残差、末端速度及完整指标见 JSON。
这里的误差是模型内的数值闭环结果，不代表真实硬件或 PhysX 精度。

W 的单次 40 节点规划数值收敛，且通过独立约束与任务门槛；它使用同一共同任务，
转成 Guidance 后由 WBC 根据执行状态重新计算力矩。名义关节参考仍来自共同生成器。
该简单场景中 W 的跟踪误差并未优于直接共同参考，不能据此宣称 W 的模型收益；
C0/C1 对照和困难工况尚未开展。

**200 Hz 实时验收未通过**：各例整周期全部 500 步都超过 5 ms。报告分别记录 WBC
求解和包含参考采样、指标审计及 plant 推进的整周期耗时。积分频率不是实际吞吐。

## 测试、跳过与未验收

本轮选择的测试为 **89 passed / 1 skipped / 0 failed**，其中 32 项为新增
Guidance/WBC 测试。覆盖共同 IK、SO(3)/C2 导数、独立 KKT、硬约束饱和、
模型/接触映射、错误包拒收、过期回退与控制失败终止。跳过原因是本机无 CUDA，
无法运行旧 Go2 adapter 非默认 stream 测试。

仍待验收：在线滚动 MPC 与新包交接、C0/C1、PhysX、非零物体载荷、碰撞、
行走 WBC 和实时性。此前单 Go2 完整站立基线未在本轮重复执行。

## 环境及隔离

已有 Python 3.11.4、NumPy 1.26.4、Torch 2.2.2、Pinocchio 4.0.0、Crocoddyl 3.2.1，
Intel macOS、CPU 单线程。未安装或升级依赖；仅对子进程设置现有原生库路径。
117 个既有源文件/示例/测试/配置等文件、82 个依赖元数据文件前后哈希一致，详见
[preservation.json](preservation.json)。旧 Go2 控制、adapter、QP、W 规划器和训练接口
未修改；未操作集成仓库或训练服务器。

## 复现命令

在仓库根目录使用已有环境：

```bash
(
  export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
  export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1
  export PYTHONPATH=.dev-tools/crocoddyl321-pin400/cmeel.prefix/lib/python3.11/site-packages:src
  .venv/bin/python -B -m pytest -p no:cacheprovider \
    tests/test_guidance.py tests/test_whole_body_qp.py tests/test_qp.py \
    tests/test_standing_whole_body.py tests/test_walking_whole_body_demo.py \
    tests/test_go2_arm.py tests/test_go2_tensor_adapter.py
  .venv/bin/python -B examples/go2_z1_wbc.py
)
```

两条验收命令均退出 0。测试和基准顺序执行，避免并发测试影响时延。
基准默认写入本地忽略目录 `artifacts/go2-z1-wbc/`，含完整 `report.json` 和五个 NPZ
实际状态/参考/力矩/接触力/外力轨迹；哈希记录在汇总 JSON。这些原始大文件没有提交，
可按上述命令重新生成（时延及包含时延的报告哈希会随运行变化）。

仓库保存 [测试日志](tests.log)、[JUnit](tests.xml)、[闭环日志](closed_loop.log)、
[汇总与配置](summary.json) 和隔离校验。实现说明见 [WBC 文档](../../go2_z1_wbc.md)。
