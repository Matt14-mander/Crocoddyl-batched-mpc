# Go2 + Z1 模型选型与资产准备

2026-11-13 按用户选型，阶段 A 使用 **Go2 + Unitree Z1 六自由度机械臂 + 固定 Unitree 夹爪**。
已完成资产冻结、仿真安装/TCP 假设、组合控制 URDF 和独立 Pinocchio CPU 模型核查；
固定支撑力分配、WBC 和 PhysX 验收仍待实现。模型实现与参数见下文。
此前单 Go2 库级基线见 [验证记录](validation.md#2026-11-13-go2-本机库级基线adapter-与站立回归)。

## 权威来源与版本

统一使用 [Unitree 官方 unitree_ros](https://github.com/unitreerobotics/unitree_ros)，冻结提交
`5994d4faef0a9cadd3287f8de0199a67eeb2a259`，与现有 Go2 `SOURCE.json` 的上游提交相同。
选用该固定快照便于复现，不声明它是最新版本。

| 内容 | 官方固定版本来源 | 本地用途 |
| --- | --- | --- |
| 六轴机械臂 | [z1.urdf](https://github.com/unitreerobotics/unitree_ros/blob/5994d4faef0a9cadd3287f8de0199a67eeb2a259/robots/z1_description/xacro/z1.urdf) | 几何、惯量、关节轴和限位 |
| 可选夹爪 | [robot.xacro](https://github.com/unitreerobotics/unitree_ros/blob/5994d4faef0a9cadd3287f8de0199a67eeb2a259/robots/z1_description/xacro/robot.xacro)、[const.xacro](https://github.com/unitreerobotics/unitree_ros/blob/5994d4faef0a9cadd3287f8de0199a67eeb2a259/robots/z1_description/xacro/const.xacro) | 夹爪安装变换、两个刚体的质量与惯量 |
| 网格 | [meshes](https://github.com/unitreerobotics/unitree_ros/tree/5994d4faef0a9cadd3287f8de0199a67eeb2a259/robots/z1_description/meshes) | 后续可视化、碰撞和 USD 转换 |
| 许可 | [仓库 LICENSE](https://github.com/unitreerobotics/unitree_ros/blob/5994d4faef0a9cadd3287f8de0199a67eeb2a259/LICENSE) | BSD-3-Clause，随资产保留 |

本地资产目录为 [`assets/go2_arm/`](../src/crocoddyl_batched_mpc/assets/go2_arm/)。
7 个上游文本文件原样保存，逐一核对 Git blob SHA-1，并记录 SHA-256；许可证与文件来源见
[`SOURCE.json`](../src/crocoddyl_batched_mpc/assets/go2_arm/SOURCE.json)。
上游 `package.xml` 的 license 字段为 `TODO`，此处保留该事实，许可证依据仓库根 `LICENSE`。

18 个官方网格约 32.86 MB，已固定路径、Git blob SHA-1、大小和下载 URL，本轮未下载。
因此当前资产能用于 Pinocchio 的运动学/惯量模型加载，尚不是可直接用于 Isaac Lab 的完整视觉与碰撞包。
后续网格和 USD 均从同一快照派生，不另换一套动力学参数。

## Z1 版本与参数口径

[产品页](https://www.unitree.com/z1/)区分六轴 Z1 AIR/PRO，列出不同重量和载荷。
所选 URDF 没有给出与 AIR/PRO 的唯一对应关系，本项目先标识为“官方 Z1 固定资产版本”，
不把产品页数字覆盖到 URDF，也不据此宣称已经验证某一硬件配置或 Go2 的带臂承载能力。

从所选资产逐 link 求和：

| 项目 | 质量 |
| --- | ---: |
| Z1 主体，含 `link00`，不含夹爪 | 4.41872026 kg |
| 夹爪 stator | 0.52603655 kg |
| 夹爪 mover | 0.27621302 kg |
| Z1 加固定夹爪 | **5.22096983 kg** |
| 现有 Go2 | 16.087 kg |
| 已知刚体质量合计，不含支架与物体 | 21.30796983 kg |

空载表示额外物体质量为零，不能把夹爪也设为零质量。载荷作为额外固定惯性体加入，
给定安装变换、质量、质心与惯量；不重复加入“物体重量”手端外力。

六个关节的所选 URDF 参数如下。速度界均为 `3.1415 rad/s`。

| 控制模型关节 | 轴（关节局部系） | 位置范围（度） | URDF effort（N·m） |
| --- | --- | --- | ---: |
| `z1_joint1` | z | -150 ～ 150 | 30 |
| `z1_joint2` | y | 0 ～ 170 | 60 |
| `z1_joint3` | y | -165 ～ 0 | 30 |
| `z1_joint4` | y | -87 ～ 87 | 30 |
| `z1_joint5` | z | -77 ～ 77 | 30 |
| `z1_joint6` | x | -160 ～ 160 | 30 |

这些是固定源文件的模型界限，**不是已核实的持续硬件力矩额定值**。上游 URDF 的某些
位置范围与产品页不同，当前 CPU 模型保留 URDF 数值并显式记录；硬件/仿真实验的执行器界限
在后续配置中核对，所有对照组使用相同界限。

## 已准备的独立控制资产

[`z1_fixed_gripper.urdf`](../src/crocoddyl_batched_mpc/assets/go2_arm/z1_fixed_gripper.urdf)
由离线脚本
[`build_z1_fixed_gripper.py`](../src/crocoddyl_batched_mpc/assets/go2_arm/build_z1_fixed_gripper.py)
从冻结文件生成，无需 ROS、xacro 安装或网络请求。变换步骤为：

1. 保留原六轴机械臂 links/joints；移除独立示例的 `world`、`base_static_joint`、Gazebo 插件和 transmissions。
2. 从同一版本 xacro 加入 `gripperStator` 和 `gripperMover`，保留两者全部惯量。
3. 将原 `jointGripper` 锁定在 **0 rad**；这只是首版固定角约定，不宣称对应实物抓取开口标定。
4. 为所有 link/joint 加 `z1_` 前缀。两个夹爪固定关节命名为
   `z1_gripper_mount_joint` 和 `z1_gripper_lock_joint`，避免上游固定关节与刚体同名带来的 frame 查询歧义。
5. 保留六轴的原始 damping/friction 标签及视觉/碰撞引用；运行时的被动阻尼/摩擦口径尚需统一。

最终为 9 个有质量刚体、6 个转动关节和 2 个固定关节。根为 `z1_link00`，
夹爪安装面为 `z1_gripperStator`，它相对 `z1_link06` 的平移为 `[0.051, 0, 0] m`。
夹爪安装面不等于已标定的抓取中心；手端任务 TCP 需独立设置。

生成文件 SHA-256：`ef8676e0c10e626837f534e6068f3c950e46b9c0b637a2189136439a01a0409d`。
原始文件与派生文件分开保存，不覆盖现有 `assets/go2/`。

## 本次核查结果

使用既有 `.venv` 的 Python 3.11.4/NumPy 1.26.4，以及已有独立目录的 Pinocchio 4.0.0，
未安装或升级依赖。运行
[`audit_z1_asset.py`](../src/crocoddyl_batched_mpc/assets/go2_arm/audit_z1_asset.py)，结果见
[`ASSET_AUDIT.json`](../src/crocoddyl_batched_mpc/assets/go2_arm/ASSET_AUDIT.json)。

- 自由浮基独立 Z1 模型为 `nq=13`、`nv=12`；驱动关节为有序 `z1_joint1…6`。
- 9 个刚体质量为正；惯量正定且满足主惯量三角关系；Pinocchio 总质量与 XML 求和一致。
- 零位、折叠和非对称三个配置通过关节范围检查，质量矩阵均正定。
- 全部 12 个切空间方向的安装面雅可比对中心差分，最大绝对误差约 `1.83e-10`，门槛 `1e-7`。
- 重力项对势能梯度中心差分，最大绝对误差约 `1.28e-9`，门槛 `1e-6`；也核对了零速零加速度 RNEA。
- 记录各配置的 FK、质心、机械臂重力力矩；这些静态代数核查不代表全身重力补偿闭环已通过。

初次审查发现上游夹爪固定 joint 与 link 同名会使未指定类型的 Pinocchio frame 查询有歧义；
派生资产已区分名称，审查脚本还显式使用 `FrameType.BODY`，最终核查通过。
没有进行网格加载、自碰撞、Go2 四足支撑可行性或 PhysX 验收。

## 仿真装配假设 v1

安装参数写入独立的
[`SIMULATION_CONFIG.json`](../src/crocoddyl_batched_mpc/assets/go2_arm/SIMULATION_CONFIG.json)。
这些是项目的仿真假设，未使用硬件安装 CAD、实测参数或网格碰撞认证。
xyz 为子 frame 原点在父 frame 中的位置；rpy 按 `Rz(yaw) Ry(pitch) Rx(roll)` 定义。

| 项目 | 首版数值 | 解释 |
| --- | --- | --- |
| Go2 `base` → `z1_link00` | xyz = `[0.05, 0, 0.10] m`，rpy = `[0, 0, 0] rad` | 臂根前移 5 cm、抬高 10 cm，轴向与机身一致 |
| 等效支架质量 | 0.30 kg | 包含紧固件的集中等效模型，非特定材料或实物支架 |
| 支架盒体尺寸 | `[0.16, 0.09, 0.04] m` | 均匀质量分布，盒体轴与 Go2 `base` 一致 |
| 支架质心，在 Go2 `base` 中 | `[0.05, 0, 0.08] m` | 盒体上表面 z=0.10 m，与臂根平面相接 |
| 支架质心惯量 | diag(`[0.0002425, 0.0006800, 0.0008425]`) kg·m² | 按均匀盒体 `m/12` 公式计算，非机身原点惯量 |
| `z1_gripperStator` → `z1_tcp` | xyz = `[0.10, 0, 0] m`，rpy = `[0, 0, 0] rad` | 沿夹爪安装面 +x 的无质量虚拟任务点，非已标定的抓取中心 |
| 夹爪关节 | 固定在 0 rad | 保留 stator/mover 的全部质量与惯量 |
| 额外物体 | 0 kg | 当前 loader 明确拒绝非零载荷配置，后续再扩展 |
| 世界重力 | `[0, 0, -9.81] m/s²` | CPU 模型和后续仿真应一致 |
| 被动关节阻尼/摩擦 | 全部 18 个驱动关节均为 0 | 理想刚体基线，仅修改组合 URDF；原始 Go2/Z1 资产保持不变 |

支架下表面 z=0.06 m，现有 Go2 base 简化碰撞盒的上表面为 z=0.057 m；两者相差 3 mm。
这只是相对于现有简化盒体的装配参考，不代表完整外壳、线缆或所有臂姿态均无碰撞。
支架通过固定关节连接到机身，模型不模拟连接件柔性。

组合模型空载总质量为 **21.60796983 kg**，已计入支架和夹爪。
名义腿姿态沿用 `[0, 0.8, -1.6] rad` 每腿；名义臂姿态取
`[0, 1.2, -1.8, 0.6, 0, 0] rad`。基座高度由四足几何接地求得，不固定基座。
它是几何初值，尚未经支撑力、摩擦、关节力矩或自碰撞可行性筛选。

## 独立 Pinocchio CPU 模型与契约

实现位于 [`models/go2_arm.py`](../src/crocoddyl_batched_mpc/models/go2_arm.py)，直接导入使用；
未修改 `models/__init__.py`，旧 Go2 控制路径不会自动加载它。
[`MODEL_SPEC.json`](../src/crocoddyl_batched_mpc/assets/go2_arm/MODEL_SPEC.json) 保存来源、
组合参数、质量、关节元数据和已实现的模型契约：

- `nq=25`、`nv=24`、`nx=49`、`ndx=48`、`nu=18`；夹爪不增加驱动维度。
- 配置为基座 xyz、xyzw 四元数、12 个腿关节、6 个臂关节；浮基速度采用机身系线速度和角速度。
- 腿顺序沿用 FL/FR/RL/RR，各 hip/thigh/calf；臂顺序为 `z1_joint1…6`。
- Pinocchio 和输入的关节数组都按名称建立映射，不假设树遍历顺序等于控制顺序。
- 旧 Go2 的 q(19)/v(18)/12 维扭矩接口保持独立。

公开 `q/v` 和所有返回的动力学量使用上述控制顺序；原生 `model` 使用 Pinocchio 顺序，
通过 `to_native_q/v`、`from_native_q/v` 转换。当前冻结模型两者恰好一致，代码仍保留显式映射。
`pack_configuration`、`pack_velocity`、`scatter_torques` 处理任意命名顺序的 CPU NumPy 数组，
额外关节槽输出零力矩；这还不是 Isaac Lab Tensor adapter。

`evaluate(q,v)` 返回世界系 TCP 位姿、四足位置、质心、TCP/足端/质心雅可比、
质量矩阵、广义重力和非线性项。TCP 雅可比为 `LOCAL_WORLD_ALIGNED`、线/角速度顺序，
速度参考点在 TCP；足端和质心雅可比为世界系线速度。
`actuation_matrix()` 默认与这些公开动力学量的顺序一致；只有直接调用 Pinocchio 时才使用
`actuation_matrix(native=True)`。

```python
from crocoddyl_batched_mpc.models.go2_arm import load_go2_arm

robot = load_go2_arm()  # 默认加载 SIMULATION_CONFIG.json，也可传独立配置路径/dict
q = robot.standing_configuration()
terms = robot.evaluate(q)
print(robot.total_mass, terms.tcp_pose, terms.com)
```

loader 校验两个源 URDF 的 SHA-256，在内存中组合模型，不覆盖源文件。导出的
[`go2_z1_fixed_gripper.urdf`](../src/crocoddyl_batched_mpc/assets/go2_arm/go2_z1_fixed_gripper.urdf)
供后续对照和仿真导入使用，保留网格引用，但本轮没有下载/加载网格。

广义重力的前 6 维是浮基所需的未驱动广义力；只取后 18 维直接施加关节力矩，
不等于求出了四足支撑下的站立平衡。模型没有附加支撑约束或进行 WBC 求解。

## 组合模型验证

新增 [`tests/test_go2_arm.py`](../tests/test_go2_arm.py) 共 **10 项通过**。
参考计算从独立 Go2/Z1 源 XML 出发，用 NumPy 实现关节变换与逐 link 质量加权，
不读取生成的组合 XML，也不复用组合模型的 FK/质心计算。
覆盖名义姿态、非零浮基位置/姿态和非对称臂姿态，以及另一组旋转安装/TCP/支架质量/重力配置。

| 核查 | 最大绝对误差 | 门槛 |
| --- | ---: | ---: |
| TCP 齐次位姿对独立 XML FK | 5.56e-16 | 2e-12 |
| 质心对逐 link 质量加权 | 5.56e-17 m | 2e-12 m |
| TCP 雅可比对 24 维中心差分 | 2.17e-10 | 2e-8 |
| 四足雅可比对中心差分 | 5.07e-11 | 2e-8 |
| 质心雅可比对中心差分 | 1.11e-10 | 2e-8 |
| 广义重力对独立势能梯度 | 3.23e-8 | 2e-6 |

此外核查总质量、质量矩阵对称/正定、零速非线性项与重力一致、非零速度的原生项映射、
全部 18 个关节的状态/力矩映射和虚功一致性。外部关节数组刻意反序并加入两个额外槽位。
异常四元数、NaN、缺失/重复关节名、错误扭矩维度和非法配置均被拒绝。
独立模块导入在禁止 NumPy/Pinocchio/Crocoddyl 新导入的测试中通过，构造模型时才加载原生依赖。

旧 `test_go2_tensor_adapter.py` 在基础 `.venv` 补跑为 **3 passed / 1 skipped**；
跳过项为 CUDA 非默认 stream，因为本机没有 CUDA。
本轮总计 **13 passed / 1 skipped**，不包含此前的 Go2 站立完整闭环重跑。
配置、映射表、名义 FK/质心/重力、逐项误差、JUnit 与兼容性校验见
[CPU 模型结果](benchmarks/go2_z1_cpu_2026-11-13/summary.json)。

后续已完成名义姿态固定四足支撑力/力矩检查和四个小幅任务的
[W CPU 参考规划验收](go2_z1_whole_body.md)。统一 WBC、非零载荷模型、碰撞检查、
完整模拟器 18 关节状态/力矩与 reset、URDF/USD 一致性和 PhysX 仍待验收。

## 对现有训练的隔离

本阶段新增独立模型、测试、示例和 `assets/go2_arm/` 下的组合资产/配置，并更新规划文档。
未修改旧 Go2 资产、控制器、adapter、求解器、包导入、训练配置或依赖声明。
没有操作训练服务器，没有运行安装命令；只对核查子进程设置 `PYTHONPATH`。

这些资产目前按源码路径使用；`pyproject.toml` 的显式 package-data 配置仍仅列出既有 Go2 资产，
新 URDF 等非 Python 资产的 wheel/sdist 收录仍待单独处理；当前模型按源码路径使用，本轮不改变安装内容。
尚未向 Isaac Lab 发布 USD，也未接入现有单 Go2 RL-guided MPC 执行链。
来源校验、可重复生成和既有代码/依赖元数据核对结果见
[`PREPARATION_CHECKS.json`](../src/crocoddyl_batched_mpc/assets/go2_arm/PREPARATION_CHECKS.json)。

## 复现

从仓库根目录执行；只写入新资产目录内的派生 URDF 和核查 JSON：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B \
  src/crocoddyl_batched_mpc/assets/go2_arm/build_z1_fixed_gripper.py

PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
  OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 \
  PYTHONPATH=.dev-tools/crocoddyl321-pin400/cmeel.prefix/lib/python3.11/site-packages:src \
  .venv/bin/python -B src/crocoddyl_batched_mpc/assets/go2_arm/audit_z1_asset.py
```

组合 CPU 模型与数值核查（无需重新安装）：

```bash
(
  export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
  export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1
  export PYTHONPATH=.dev-tools/crocoddyl321-pin400/cmeel.prefix/lib/python3.11/site-packages:src

  .venv/bin/python -B -X faulthandler -m pytest -p no:cacheprovider -vv -ra -s tests/test_go2_arm.py
  .venv/bin/python -B examples/go2_arm_reference.py --output /tmp/go2-z1-reference.json \
    --urdf-output /tmp/go2-z1.urdf
)
```
