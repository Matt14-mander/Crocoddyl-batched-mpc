# Go2 网页动画（Meshcat）

slow walk 和 walk / trot / pace / bound 示例现在支持与 Crocoddyl
`MeshcatDisplay` 相同的 Meshcat / Pinocchio 三维查看器。动画来自**闭环 plant
执行后的状态**，包括接触等待；不是参考姿态动画。默认展示环境 0，
`--display-env 1` 可观察第二个扰动环境。绿色标记表示支撑足，橙色表示摆动足，
蓝线表示整个执行过程的足端路径。

## 运行示例并打开网页

本机已有 `croco_env`，从项目根目录先设置验证时使用的路径：

```bash
conda activate croco_env
export PYTHONPATH="$PWD/.dev-tools/croco-numpy126:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1
```

这里的 `src` 使未安装的项目包可被导入；项目内 NumPy 1.26.4 路径解决
该环境 Torch 2.2.2 与 NumPy 2.x 的桥接不兼容。该路径已在本机准备好，
不随 Git 分发，也不修改原环境的依赖。其他环境安装可选依赖：

```bash
python -m pip install -e ".[crocoddyl,viewer]"
```

然后运行：

```bash

# 慢速 crawl：在线 MPC + 独立 Pinocchio plant
python examples/go2_slow_walk.py --cycles 1 --plant pinocchio --display

# 四步态：生成 walk/trot/pace/bound 切换页面
python examples/go2_quadrupedal_gaits.py --gait all --cycles 1 --display

# 单个步态，显示扰动环境
python examples/go2_quadrupedal_gaits.py --gait trot --display --display-env 1
```

`--display` 在执行完成后导出网页并打开默认浏览器。默认文件位于
`artifacts/go2-viewer/`。四步态入口为 `index.html`，慢速 walk 为
`slow_walk.html`。鼠标拖动旋转、滚轮缩放；点击右上角 **Open Controls**，
展开 **Animations → default** 后可播放、暂停、重置、拖动 time 和调整
timeScale。页面在本机运行，无需上传轨迹或连接远程服务。

## 保存轨迹和合并展示

`--html` 指定 HTML 输出位置：slow walk 接受文件路径，四步态接受目录。
`--trace` 同理，分别接受 JSON 文件路径和 JSON 输出目录。
记录轨迹不依赖 Meshcat；从轨迹重新生成网页无需重新规划或运行控制器。

```bash
python examples/go2_slow_walk.py --cycles 1 --plant pinocchio \
  --trace artifacts/go2-traces/slow_walk.json --html artifacts/go2-viewer/slow_walk.html
python examples/go2_quadrupedal_gaits.py --gait all --cycles 1 \
  --trace artifacts/go2-traces --html artifacts/go2-viewer

# 合并五种步态（实际闭环轨迹）
python examples/go2_web_viewer.py artifacts/go2-traces/slow_walk.json \
  artifacts/go2-traces/walk.json artifacts/go2-traces/trot.json \
  artifacts/go2-traces/pace.json artifacts/go2-traces/bound.json --open
```

HTML 包含 Meshcat 脚本、几何和动画，可直接双击打开。
也可以在项目根目录运行下面的命令，然后访问 <http://127.0.0.1:8765>：

```bash
python -m http.server 8765 --bind 127.0.0.1 --directory artifacts/go2-viewer
```

## 外观网格

仓库仅包含宇树 Go2 官方 URDF、许可和来源记录，未包含外观网格。
默认用 URDF 的 box / cylinder / sphere 碰撞几何呈现完整关节运动，
页面明确标注这一模式。已有 `unitree_ros/robots/go2_description/dae/`
资产时，给上述命令添加 `--mesh-dir /path/to/unitree_ros/robots`，即可通过
Pinocchio 加载原始 Go2 外观模型。路径必须包含 `go2_description/dae/`；
网格缺失会报错，不会悄悄替换成其他机器人。

## 时间与验证范围

控制计时只包围原有控制器调用；轨迹拷贝、FK、几何加载和 HTML 导出均在计时
范围之外。动画采用 50 fps，按累计仿真时间设置关键帧。零时长碰撞节点保留在
JSON 中，渲染时同时间戳使用碰撞后的状态，不额外延长动画。
网页是有限轨迹回放，循环时会回到初始状态；不是在线网页遥控或无限续步。
失败时也可以导出已执行部分，并显示“部分轨迹 / 未完成”。

新增记录器验收覆盖选定环境、快照独立性、零时长冲击、序列保存加载和网页
名称转义。本机 `croco_env` 中另运行五个示例、导出实际轨迹和 HTML，并在
浏览器检查场景、回放控件和步态切换。四步态仍然是离线 Crocoddyl 规划加
Torch 反馈与约束门禁；可视化不改变其收敛、实时性或真机验证边界。

五个示例均完成双环境独立 Pinocchio plant 执行，记录环境 0；
slow walk 为 14.46 s，walk 为 2.40 s，其余三种为 1.60 s。
记录器及相关 Torch 回归检查共 7 项通过。结果见
[网页轨迹验证记录](benchmarks/go2_viewer_cpu_2026-10-09.json)。
此次运行存在并发负载，记录只用于回放正确性验证，不作为 50 Hz 性能基线。
