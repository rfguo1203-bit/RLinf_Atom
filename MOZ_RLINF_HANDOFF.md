# MOZ × RLinf 真机 RLPD 交接说明

本文记录当前对 RLinf、MOZ SDK 与 OpenPI 真机推理栈的代码理解、已经确认的开发决策，以及首版适配的实现状态。后续会话应先阅读本文，再继续实现。本文中的“已实现”均指当前工作区代码，尚未替代现场的标定和真机验收。

## 0. 当前实现状态（2026-09-22）

首版代码已在当前工作区完成，目标是安全地跑通 demo → replay → update → checkpoint，而非直接承诺抓取成功率。

- 已将 ``packages/mozrobot`` 的 ``MOZ1Robot.connect`` 扩展为 ``auto_reset=False`` 可选模式，并公开 ``disable_external_following_mode()``，使 RLinf 可以只读连接和显式安全停止。
- 已新增 ``MOZConnection`` / ``MOZRobot``，并注册 ``MOZ`` hardware。连接层是 SDK 的唯一写入者：真实硬件使用本地 120 Hz 保持流；未显式打开 motion gate 时拒绝下发目标；关闭或异常时停用 teleop 与 external following。dummy 模式不导入厂商 SDK。
- 已新增 ``MozPickLift-v0``：右臂 7D delta action，14D state（右夹爪、右臂关节、右 TCP），头部与右腕各一路 224×224 RGB。真机配置只有在 ``motion_enabled``、``safety_calibrated``、home、工作空间和夹爪范围均齐备时才允许复位和运动。
- 已新增 ``moz_native`` teleop 适配。它只读取 MOZ 原生 teleop 的目标，将其转换为同一 action contract，再经 ``MOZConnection`` 下发，不创建第二个 SDK 写入者。
- 已新增采集、两节点异步 CNN RLPD 和 dummy e2e 配置；H100 的 actor/rollout 固定在 GPU 0，MOZ 节点仅运行 environment worker。配置中所有真实序列号和标定值均为禁动占位符。
- 已补齐 MOZ 安装路由、dummy Docker/CI 路由、单元测试、英中文档、real-world 索引和 README 入口。通用安装不会安装 ROS、驱动或 ``draccus``；MOZ 必须复用已经跑通 π0.5 的运行时。
- 已完成 Python 语法编译、YAML/Shell 静态检查、Git whitespace 检查和文档符号检查；本地默认 Python 缺少 ``torch``、``draccus``、``docutils`` 及完整 RLinf 依赖，因此尚未在本机执行 RLinf 单测/e2e、Sphinx build 或真实 SDK 集成测试，也没有擅自安装/升级依赖。

## 1. 当前目标与边界

最终目标是在千寻 MOZ 真机上，使用 RLinf 完成简单物体抓取的真机强化学习训练。当前实施目标是先跑通安全、可重复的真实机器人 CNN RLPD 闭环；训练效果优化（Pi0.5 在线微调、自动奖励、多 GPU、双臂等）放在闭环稳定之后。

首个任务已锁定为：桌面固定标记区域内的标准物块，**右臂定点抓取并抬升**。

已确认的资源：

- MOZ 控制端带一张 4090，已能运行 Pi0.5 真机推理。
- 训练服务器为 8 张 H100；首轮只使用 1 张 H100，因为真机采样速率而非训练算力是瓶颈。
- 两台机器能通过 SSH 连接；运行期的观测、动作和权重通信应由 Ray 完成，SSH 只用于部署、运维和日志排查。

本仓库根目录为 `/Users/rkos/Workspace/RLinf_Atom`。OpenPI MOZ 推理仓库为 `/Users/rkos/Workspace/openpi-openpi_moz`。用户指定的 OpenPI 真机入口是：

```bash
python3 examples/moz1_real/main.py
```

截至写本文时，尚未为该方案修改任何代码。`git status` 中已有未跟踪的 `packages/` 和 `PLAN.md`；它们不能被误认为是本轮新建文件。

## 2. 已理解的 MOZ / OpenPI 通信架构

### 2.1 OpenPI 真机推理路径

关键代码：

- `/Users/rkos/Workspace/openpi-openpi_moz/examples/moz1_real/main.py`
- `/Users/rkos/Workspace/openpi-openpi_moz/examples/moz1_real/env.py`
- `/Users/rkos/Workspace/openpi-openpi_moz/src/openpi/serving/websocket_policy_server.py`
- `/Users/rkos/Workspace/openpi-openpi_moz/packages/mozrobot/`

现有 Pi0.5 推理是一个独立闭环：

```text
MOZ1Robot / 相机 -> Moz1RealEnvironment -> WebsocketClientPolicy
    -> ActionChunkResampleBroker -> Runtime -> MOZ1Robot.send_action
```

- `main.py` 创建 `WebsocketClientPolicy`，向远端或本地 policy server 请求 Pi0.5 action chunk。
- `ActionChunkResampleBroker` 默认将约 30 Hz 的模型 action chunk 重采样到 120 Hz 机器人控制节奏；示例使用 `action_horizon=200`、`resample_ratio=4.0`。
- `Moz1RealEnvironment` 在构造时实例化 `MOZ1Robot(MOZ1RobotConfig(..., robot_control_hz=120))`，读取相机/本体状态，再将完整 action 交给 SDK。
- WebSocket 使用 msgpack + NumPy；它是 OpenPI 推理协议，不是 RLinf 真机训练接口。

因此，RLinf 真机训练**不能**把现有 `main.py` 当作环境进程同时运行。RLinf rollout 已经生成动作；MOZ 的 RL 环境应直接调用 SDK。Pi0.5 进程与 RLinf MOZ 环境不得同时占用 ROS 控制器、相机或同一机器人连接。

### 2.2 MOZ SDK 的状态、动作和控制器

`packages/mozrobot` 在两个仓库中曾经内容一致。用户已决定后续把它纳入 RLinf 版本控制，并以 RLinf 为唯一源码；OpenPI 运行时应通过 editable install 或 `PYTHONPATH` 引用 RLinf 的这一份，不再维护独立副本。

关键观察：

- `MOZ1RobotConfig` 支持 `structure`、`realsense_serials`、相机分辨率、`robot_control_hz`、观测频率、`virtual_robot`、`no_camera`、软实时/CPU 绑定等配置。
- `examples/moz1_real/main.py` 字面默认 `structure="wholebody"`；但 README 写明实际常见配置为 `wholebody_without_base`。本项目已确认首版使用 **`wholebody_without_base`**，不控制底盘。
- 现有示例相机序列号字符串为 `2-2,2-8,2-3`，顺序为头部、左腕、右腕。不要把该字面值硬编码进通用代码，应从实际 MOZ 节点配置注入。
- `capture_robot_observation()` 可提供左右 TCP cartesian pose（位置 + rotvec，6 维）、关节位置、夹爪和相机帧；`capture_images()` 提供多相机帧。
- SDK cartesian action 使用绝对 `[x, y, z, rx, ry, rz]`，单位为米、弧度；夹爪命令单位为米。SDK 会把完整 action 转换为 ROS `MechUnitCmdArray`，并分别处理夹爪/底盘命令。
- 底层 `real_env_moz1.py` 的 `send_action(action, action_time)` 可以发送完整整机 command；控制器有带时间戳的队列/服务端。文档要求外部跟随模式下持续以机器人控制频率发送统一 action。
- `MOZ1Robot` 已公开 `connect()`、`reset_robot_positions()`、`start_teleop()`、`stop_teleop()`、`teleop_step()` 和 `enable_external_following_mode()`。底层控制器存在 `disable_external_following_mode()`，但 `MOZ1Robot` 缺少对应公开封装；实现时必须补齐公开的停用/安全保持 API，不能从 RLinf 访问私有 `_env`。

### 2.3 原生遥操作

`mozrobot.teleoperations.moz_teleop_adapter.MozTeleopAdapter` 是 ROS 服务/话题适配器：它连接 `moz_teleop`，订阅机械单元、夹爪、底盘命令和 teleop 状态。`MOZ1Robot.teleop_step(record_data=True, record_image=True)` 会返回：

1. 当前机器人/相机 observation；
2. 原生遥操作产生的完整 command target。

它读取 target，不应假设自动完成 RLinf 的轨迹写入或安全过滤。采集时必须让 **一个**本地 command owner 接收该 target、冻结左臂/躯干/底盘、进行同一套安全校验后下发。绝不能同时由遥操作、环境零动作和 policy 三方写控制器。

## 3. 已理解的 RLinf 真机与 RLPD 架构

### 3.1 RealWorldEnv 和机器人抽象

关键路径：

- `rlinf/envs/real/env.py`：`RealWorldEnv`，负责加载真实 task、包装 observation、交给 Gym/Ray 环境 worker。
- `rlinf/envs/real/registry.py`：真实 task 创建与 wrapper 组装。
- `rlinf/envs/real/__init__.py`：真实 task package 的 lazy loader；MOZ 应在这里注册。
- `rlinf/robotics/`：硬件抽象。`Connection` 负责延迟打开设备；`Robot`/parts/views 负责将逻辑 robot 部位映射到同一硬件连接。
- `rlinf/workers/env/env_worker.py` 和 `rlinf/workers/env/async_env_worker.py`：同步/异步环境交互。

真实环境仍然使用 `env_type: real`，MOZ 是 real task/robot/hardware resource，而不是另建一个独立的 `SupportedEnvType`。现有 Turtle2/DOSW1 的共享连接设计是较适合的参考；Franka task 的任务逻辑不可直接复用。

`RealWorldEnv` 将 task observation 展平为 policy observation：state 按 key 组合，主相机由 `main_image_key` 指定，其他相机成为额外视角。因此 MOZ task 应稳定输出确定的键名与 shape，不要让 SDK 原始字段名直接泄漏为训练 schema。

### 3.2 现有真机 RLPD 配置

参考配置为：

- `examples/embodiment/config/realworld_pnp_rlpd_cnn_async.yaml`

它是 Franka 的双节点异步 CNN RLPD 示例：GPU 节点放 actor/rollout，机器人节点放 env；使用 7D action、19D state、双图像、demo buffer + online replay buffer、10 个 Q head。MOZ 配置应沿用它的训练/replay/placement 骨架，但不能沿用 Franka 的状态、动作、SDK 或 task。

RLPD 冷启动数据通过 `algorithm.demo_buffer.load_path` 载入，必须与在线 action、state、image schema 完全一致。Pi0.5 的 20D 双臂/躯干 action 不能直接充当本项目 7D 单臂 RLPD demo。

### 3.3 人工奖励和采集工具

RLinf 已有 headless evdev 键盘机制，不需要新建 Web UI：

- `single_stage`：`a=-1` 失败、`b=0` 中性、`c=+1` 成功；没有开始门控。
- `start_end`：用于演示采集，`a` 开始/中止录制、`b` 切分 segment、`c` 成功结束。
- `eval_control`：复位后保持 idle 并等待 `a` 开始；运行中 `b` 以 `0` 结束失败，`c` 以 `+1` 结束成功。

用户已经选择：

- 采集 demonstrations 使用 `start_end`；
- 在线 RLPD 使用 `eval_control`，确保操作员摆放物体后才开始动作；
- 使用 MOZ 节点本地物理键盘/evdev 的 `a/b/c`，不是标准输入文本；必要时通过 `RLINF_KEYBOARD_DEVICE` 指定稳定的 `/dev/input/by-id/...` 设备。

`examples/embodiment/collect_real_data.py` 已能通过 `TrajectoryAccumulator` 和 `TrajectoryReplayBuffer` 产出可供 replay 使用的 `.pt` trajectory。新建 MOZ 原生 teleop 适配时应复用该格式，而不是新造数据格式。

## 4. 已锁定的首版设计决策

| 项目 | 已确认决策 |
| --- | --- |
| 机器人构型 | `wholebody_without_base` |
| 控制范围 | 仅右臂；左臂和躯干固定，底盘禁用 |
| 任务 | 固定区域物块抓取并抬升 |
| 模型/算法 | CNN RLPD（先不做 Pi0.5 在线微调） |
| 相机 | 头部 + 右腕，两路 RGB |
| policy 节奏 | 10 Hz，最长 15 秒 / 150 RL step |
| 底层控制 | MOZ 本地 120 Hz stream，保持最新安全目标 |
| 人工回报 | `eval_control`：a 开始，b 失败 0，c 成功 +1 |
| demonstrations | MOZ 原生遥操作，先收集 50 条有效成功轨迹 |
| 训练算力 | H100 节点 1 卡，actor/rollout colocate |
| SDK ownership | `packages/mozrobot` 跟随 RLinf 版本控制；RLinf 为唯一源码 |
| 安全基线 | 未标定即禁动；先完成真机校准再生成可动配置 |
| 首轮验收 | 稳定闭环：demo、真机执行、回报、replay、更新、checkpoint 都跑通；暂不承诺成功率 |

## 5. 拟实现的公开接口与数据 contract

### 5.1 机器人连接与命令流

新增 `MOZConnection`（以及必要的 `MOZRobot`/硬件配置注册），由它独占一个 `MOZ1Robot` 实例。公开职责应包括：

- lazy `open`/`close`；
- 读取带本地单调时钟的状态/图像 snapshot；
- 显式 reset 到已标定姿态；
- 开启/停用 external-following 和 native teleop；
- 提交已验证的完整 command target；
- 超时 hold/安全停止。

在 `MOZConnection` 内部或其紧邻的本地组件中实现 `MOZCommandStreamer`：

- 环境/teleop 以 10 Hz 提交最新 target；
- streamer 以 120 Hz 发送最新完整目标给 SDK；
- command stale、frame stale、异常或关闭时，不再接受新动作并进入 hold/安全停止；
- policy、teleop、reset 是互斥状态，streamer 是唯一下发者。

### 5.2 `MozPickLift-v0` 环境 contract

新增 `MozPickLift-v0` 真实 task，并在 real task loader、robot registry、Ray hardware resource 中注册。

固定 policy state 为 14 维：

```text
[right_tcp_xyz(3), right_tcp_rotvec(3), right_joint_pos(7), right_gripper(1)]
```

固定 policy action 为归一化 7 维：

```text
[delta_x, delta_y, delta_z, delta_rx, delta_ry, delta_rz, gripper]
```

环境将 action 映射为已裁剪的绝对 `rightarm_cmd_cart_pos` 与右夹爪位置。全身 command 中左臂/躯干保持最近安全目标，永远不写底盘。状态键固定，图像键固定为 `head_rgb`、`right_wrist_rgb`；CNN 配置使用 `state_dim: 14`、`image_num: 2`。

需要定义 typed/configurable 的 MOZ task 配置，至少含：结构、相机序列号、控制频率、home/reset 位姿、工作空间边界、单步平移/旋转阈值、夹爪范围、最大图像/动作时延、`motion_enabled`。当 `motion_enabled=true` 而标定字段不完整时，配置校验必须失败。

### 5.3 原生 teleop 的轨迹转换

新增 `teleop: moz_native` 适配：

1. 从 `teleop_step(record_data=True, record_image=True)` 得到 observation 和原生 command target；
2. 将右臂目标相对当前 observation 转为同一 7D 归一化 RLPD action；
3. 将经安全过滤后的完整 target 唯一地交给 streamer；
4. 以 10 Hz 记录 `(obs, action, next_obs, reward, done)`，使用 `start_end` 标注并只保留成功 episode；
5. 导出的 trajectory 必须能直接用于 `demo_buffer.load_path`。

## 6. 安全与部署要求

可动代码默认禁用。启用前由现场操作员完成并填入配置：安全工作空间、home/reset 位姿、夹爪开闭范围、单步增量限制、速度限制、实际相机序列号，以及 e-stop 操作流程。

必须保证：

- 构造连接不移动机器人；只有显式 reset 或通过 motion gate 后才可能动作。
- 所有完整 command 在同一位置做 workspace、增量、姿态、夹爪和新鲜度检查。
- 退出、异常、Ray actor 终止时先停止 teleop/external-following 或 hold，再回收相机/ROS/子进程；清理应幂等。
- 真机执行中不能同时运行 OpenPI 的 `examples/moz1_real/main.py`。
- OpenPI 和 RLinf 可以保持各自 Python 环境（OpenPI 要求较新 Python/Torch），但 MOZ SDK 源必须相同；不要尝试将两套环境强行合并。

Ray 部署目标：H100 为 `RLINF_NODE_RANK=0` 的 head，MOZ 为 `RLINF_NODE_RANK=1` 的 env 节点；必须在每台机器 `ray start` 前设置相应环境变量、Python 环境和 SDK 路径。配置中 actor/rollout placement 指向 H100 第 0 卡，env placement 指向 MOZ hardware resource。

## 7. 实施顺序

1. 将 `packages/mozrobot` 正式纳入版本控制，建立 MOZ 节点的 RLinf 运行环境和 OpenPI 对同一 SDK 源的引用方式；增加启动时 SDK 来源/版本诊断。
2. 补齐 SDK 的公开安全停止能力，新增 `MOZConnection`、唯一 120 Hz streamer、robot/hardware 注册与 dummy/mock 支持。
3. 实现 `MozPickLift-v0`、14D/双图像 schema、7D action 映射、motion gate 与 `moz_native` teleop 适配。
4. 新增 demo 收集配置和异步 CNN RLPD 配置：H100 一卡 + MOZ env、`eval_control`、10 Hz、150 step、双图像、50 demo 路径。
5. 先运行 dummy/mock e2e，再进行无运动真机预检、标定、受限微小动作、teleop demo 收集，最后运行两节点训练闭环。
6. 闭环稳定后，再单独设计训练效果提升实验：demo 质量/数量、reward 自动化、图像增强、训练超参、Pi0.5 行为先验或在线微调、多 GPU 等不属于首版适配。

## 8. 测试与验收清单

代码级测试：

- `MOZConnection` mock SDK：状态读取、完整 command 打包、右臂动作转换、左臂/躯干 hold、超时 hold、幂等 close。
- 任务环境：7D action、14D state、双 RGB 图像、边界裁剪、缺失/过期帧和禁动配置错误。
- native teleop：输出 action 与在线 policy schema 相同，且无双写入；trajectory 可被 replay 加载。
- dummy real-world e2e：异步 CNN RLPD 完成 demo 加载、采样、actor/critic 更新和 checkpoint 恢复。

真机分阶段验收：

1. 只读预检：ROS、SDK、相机、evdev 键盘、Ray 节点、两图像色彩/尺寸/新鲜度。
2. 标定后：reset、hold、极小受限平移/旋转/夹爪动作，验证 action age 保护和安全停止。
3. 原生遥操作：收集并回放检查 50 条成功 demos。
4. 两节点 RLPD：操作员按 `a` 开始，`b/c` 标注；确认在线轨迹进入 replay、发生更新、指标持续记录且 checkpoint 可恢复。

## 9. 后续实现前仍需由现场提供的事实

这些不是架构选择，而是启用真实运动前必须填写的现场参数：

- 实际 MOZ 节点上三路 Realsense 的稳定序列号和头部/右腕对应关系；
- 右臂安全工作空间、home/reset joint pose、夹爪有效范围、最大每步位移/旋转和速度；
- `moz_teleop` 设备配置及其 ROS service/topic 是否与当前 SDK 默认值一致；
- 可读的 evdev 设备路径（如需要设置 `RLINF_KEYBOARD_DEVICE`）；
- H100 与 MOZ 的实际 IP、网卡/NCCL 配置及两个节点上的 Python/Ray 可执行路径。

在这些值缺失时，只能推进 mock/dummy、只读探测和禁动检查，不能用示例默认值驱动真机。
