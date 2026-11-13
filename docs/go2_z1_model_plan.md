# Go2 + Z1 模型选型与资产准备

2026-11-13 按用户选型，阶段 A 使用 **Go2 + Unitree Z1 六自由度机械臂 + 固定 Unitree 夹爪**。
本次完成资产冻结、独立机械臂控制 URDF 和数值核查；Go2 安装参数、组合模型及固定支撑控制仍待实现。
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

## 组合模型契约与待确定项

[`MODEL_SPEC.json`](../src/crocoddyl_batched_mpc/assets/go2_arm/MODEL_SPEC.json) 保存来源、
质量、派生规则、关节元数据以及下列**尚未实现的组合模型契约**：

- `nq=25`、`nv=24`、`nx=49`、`ndx=48`、`nu=18`；夹爪不增加驱动维度。
- 配置为基座 xyz、xyzw 四元数、12 个腿关节、6 个臂关节；浮基速度采用机身系线速度和角速度。
- 腿顺序沿用 FL/FR/RL/RR，各 hip/thigh/calf；臂顺序为 `z1_joint1…6`。
- native 模型和模拟器都按名称建立映射，不假设树遍历顺序正好等于控制顺序。
- 旧 Go2 的 q(19)/v(18)/12 维扭矩接口保持独立。

以下参数在元数据中明确为 `null`，组合模型实现应要求显式配置，不能当作零值：

| 参数 | 需要确定的内容 |
| --- | --- |
| 安装变换 | Go2 `base` → `z1_link00` 的 xyz/rpy；来自明确的仿真装配配置或安装 CAD/测量 |
| 支架 | 质量、质心与惯量；若仿真先忽略，须显式声明并作为模型假设 |
| 手端 TCP | `z1_gripperStator` → 抓取点的 xyz/rpy；先完成夹爪几何核对 |
| 执行器与被动动力学 | 最终力矩界、速度界、damping/friction 与 effort 模式；与 WBC/PhysX 一致 |

选定的官方 Z1 独立资产没有 Go2 安装变换；不能用 AlienGo/B1 的安装值冒充 Go2 标定。
CPU 的刚体重力/RNEA 核查没有验证 URDF 阻尼与摩擦的仿真执行效果；也不能再叠加一套位置 PD
来执行 WBC 的力矩而不计入实验定义。

下一步先完成显式安装配置和独立 `models/go2_arm.py` 的 Pinocchio CPU 装配/加载，核对
全身质量、FK、质心、雅可比、重力及全部 18 关节映射，再进入固定支撑 WBC。
现有 `Go2TorchDynamics` 固定为 12 腿关节，不在本次把它直接改成带臂模型。

## 对现有训练的隔离

本次只新增 `assets/go2_arm/` 资产和离线准备/核查脚本，并更新规划文档。
未修改旧 Go2 资产、控制器、adapter、求解器、包导入、训练配置或依赖声明。
没有操作训练服务器，没有运行安装命令；只对核查子进程设置 `PYTHONPATH`。

这些资产目前按源码路径使用；`pyproject.toml` 的显式 package-data 配置仍仅列出既有 Go2 资产，
新 URDF 等非 Python 资产的 wheel/sdist 收录应随独立带臂模型实现处理，本轮不改变安装内容。
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
