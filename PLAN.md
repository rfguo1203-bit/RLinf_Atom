# MOZ 单臂定点抓取的 RLPD 真机开发计划

## 总体方案

首版固定为：`wholebody_without_base`、右臂、头部 + 右腕 RGB 相机、CNN RLPD、10 Hz policy / 150 step episode、50 条成功 demonstrations、单张 H100 训练。验收重点是稳定训练闭环，而非首版抓取成功率。

现有 MOZ 推理是“WebSocket policy + 120 Hz action resample + MOZ SDK”；RLinf 真机训练改为直接通过 Ray 把 H100 的 rollout 动作送到 MOZ 环境，不启动或复用 `examples/moz1_real/main.py`。

```text
H100（Ray head；actor + rollout，1 GPU）
        ⇅ Ray：观测、动作、权重、指标
MOZ（Ray env worker；4090 不参与首版训练）
        ⇅ 本地 120 Hz 命令流
MOZ SDK / ROS / 相机 / 右臂
```

## 核心实现

- 将 `packages/mozrobot` 纳入 RLinf 版本控制，作为唯一源码；MOZ 上的 OpenPI 环境通过 editable install 或 `PYTHONPATH` 引用这份源码。OpenPI 与 RLinf 保持独立 Python 环境，但不得各自维护 SDK 副本。

- 在 RLinf real-world 架构中新增 `moz` 任务包、MOZ hardware resource、`MOZConnection` 和 `MOZRobot`。连接层延迟连接 `MOZ1Robot`，只暴露读取快照、提交整机命令、复位、启停 external-following、关闭等公开操作；补齐 SDK 缺失的公开安全停止接口，不访问私有 `_env`。

- 新增 `MozPickLift-v0`：右臂 action 为归一化 7 维 `[Δx, Δy, Δz, Δrx, Δry, Δrz, gripper]`；转换为 SDK 的绝对右臂 cartesian target 和夹爪位置。左臂、躯干保持最近安全目标，底盘始终不发送动作，并且每个周期只下发一个完整整机命令。

- 固定 policy observation schema：`right_tcp_xyz(3) + right_tcp_rotvec(3) + right_joint_pos(7) + right_gripper(1)`，共 14 维；图像键统一为 `head_rgb` 与 `right_wrist_rgb`，均为 RGB，CNN 配置 `image_num: 2`、`state_dim: 14`。启动时验证图像尺寸、颜色通道、帧时间戳和最大帧龄。

- 在 MOZ 本地实现唯一的 120 Hz `MOZCommandStreamer`。环境只在 10 Hz 提交经校验的最新目标；streamer 持续保持目标、处理动作超时，并避免 Ray 网络抖动导致控制器断流。遥操作、policy、reset 三种状态互斥，任何时刻只有一个命令写入者。

- 所有可动配置必须显式给出 home/reset 位姿、工作空间边界、单步平移/旋转上限、夹爪范围、命令和图像超时。默认 `motion_enabled: false`；缺少任一标定项或 frame/action 过期时保持并拒绝运动。关闭、异常、Ray worker 退出时先停止遥操作/外部跟随并进入安全保持，再清理 ROS、相机和子进程。

## 数据、奖励与训练

- 新增 `teleop: moz_native` 采集适配。它调用 SDK 的 `teleop_step(record_data=True, record_image=True)` 获取原生右臂目标；经同一动作归一化和安全过滤后，由本地 streamer 唯一执行并写入轨迹，绝不与零动作或 policy 动作竞争。

- demonstrations 使用现有 `start_end` 键盘流程：`a` 开始/中止录制、`c` 成功结束；仅保存成功轨迹。复用 RLinf 的 trajectory/replay 格式，确保其 7D action、14D state、双相机 observation 可直接作为 `demo_buffer.load_path`。目标为 50 条通过 schema 与回放检查的成功轨迹。

- 在线 RLPD 使用现有 `eval_control`：复位后机械臂保持等待，操作员摆放物体并按 `a` 开始；`b` 以 reward `0` 结束失败，`c` 以 reward `+1` 结束成功；150 step 超时也以失败结束。这样每次 reset 后不会在物体尚未摆放时开始运动。

- 基于现有异步 CNN RLPD 配置新增 MOZ 专用配置：H100 为 `node_rank: 0`、MOZ 为 `node_rank: 1`；actor 与 rollout colocate 在 H100 第 0 卡，env 只部署在 MOZ。记录 success rate、动作裁剪率、命令/图像帧龄、reset 时长、replay/demo 比例、SAC 更新和 checkpoint 指标。

## 验证与交付顺序

1. 用 mock SDK 完成单元测试：动作映射、14D/双图像 schema、命令单写入、超时保持、工作空间裁剪、幂等关闭和 hardware placement。
2. 使用 `is_dummy` 跑 real-world 环境与异步 RLPD e2e，验证 50 条 demo 的加载、replay 采样、更新和 checkpoint 恢复。
3. 在 MOZ 做无运动预检：ROS/相机/键盘设备、两路图像、SDK 来源、Ray 节点和命令所有权。
4. 操作员完成安全标定后，依次执行复位、保持、极小受限动作、原生遥操作采集，再完成 50 条 demonstrations。
5. 启动双机 RLPD：至少完成多轮人工门控 episode、replay 更新和 checkpoint 保存；物理真机测试不进入 CI，但提供可重复的 smoke-check 与部署文档。

## 已锁定的默认项

- 不在首版做 Pi0.5 在线微调、双臂/底盘控制、自动视觉奖励或多 GPU 扩展。
- Pi0.5 推理和 RLinf 环境不得同时占用 MOZ 控制器。
- H100 与 MOZ 的 SSH 只用于部署和运维；训练数据与动作通信由 Ray 承担。
- 安全标定完成前只能运行 dummy、只读探测和无运动检查。
