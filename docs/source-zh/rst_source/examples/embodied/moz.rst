MOZ 真机上的 RLPD
==================

本指南用于把已经能够运行 π0.5 推理的 MOZ 接入首个真机 RLPD 流程：先采集右臂桌面抓取并抬升的成功示教，再由 H100 节点训练 CNN policy，MOZ 在本地以 10 Hz 执行动作。本流程只覆盖固定物体和右臂控制，不适用于底盘、左臂或 π0.5 的在线微调。

概览
----

训练计算不会直接进入机器人控制器。H100 主机只运行同置的 actor 和 rollout GPU；MOZ 主机运行唯一的环境 worker、ROS SDK 会话、原生遥操作、图像采集，以及 120 Hz 的本地保持命令流。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      CNN policy · ResNet-10 encoder

   .. grid-item-card:: 算法
      :text-align: center

      SAC / RLPD

   .. grid-item-card:: 任务
      :text-align: center

      右臂固定物体抓取并抬升

   .. grid-item-card:: 硬件
      :text-align: center

      MOZ · 头部 RGB · 右腕 RGB · H100

仓库中的配置都是安全模板。只有私有站点配置提供经过复核的复位关节、Cartesian 工作空间和夹爪范围后，系统才会打开 external following。

任务流程
--------

先通过 dummy 任务验证 RL 链路，再到真机采集示教，最后启动在线 RLPD。

.. list-table::
   :header-rows: 1
   :widths: 28 38 34

   * - 阶段
     - 配置 / 入口
     - 结果
   * - Dummy RLPD smoke test
     - ``realworld_moz_dummy_sac_cnn.yaml``
     - 不导入厂商 SDK，验证 14 维 state、双图像和 7 维 action 的完整 contract。
   * - 原生遥操作采集
     - ``realworld_moz_collect_data.yaml`` + ``collect_real_data.py``
     - 只保存由 MOZ 操作员明确标为成功的 episode。
   * - 在线训练
     - ``realworld_moz_pick_lift_rlpd_cnn_async.yaml`` + ``train_async.py``
     - 将成功示教与在线经验一起用于训练。

观测与动作
----------

policy 每 10 Hz 接收一个状态快照。state 顺序、相机顺序和夹爪范围必须保持不变；任一项改变都会使既有示教和 checkpoint 的输入语义失效。

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - 字段
     - Contract
   * - State
     - 共 14 个值：``right_gripper`` （1）、``right_joint_positions`` （7）和 ``right_tcp_pose`` （6）。RLinf 按排序后的 key 连接，因此顺序固定为这三项。
   * - Images
     - ``head_rgb`` 是主视角，``right_wrist_rgb`` 是额外视角；两者均为 ``[224, 224, 3]`` 的 uint8 图像。
   * - Action
     - 共 7 个归一化值：三维平移增量、三维 rotation-vector 增量和一个夹爪命令。初始尺度为每步 1 cm 和 0.05 rad；夹爪动作从 ``[-1, 1]`` 映射到标定后的物理范围。
   * - Reward
     - 操作员标注在线结果：踏板 ``c`` 表示成功并返回 1，``b`` 表示失败并返回 0。

命令归属与安全边界
--------------------

真机环境 ``MozPickLiftEnv`` 只生成经过边界检查的 7 维 action，不会直接调用厂商 SDK。运行在 MOZ 环境 worker 内的 ``MOZConnection`` 独占 SDK 会话，是控制器唯一的写入者。

连接建立时，SDK 使用 ``auto_reset=False`` 打开，following mode 保持关闭。真机复位会依次调用 ``reset_to_home()``、等待设定的稳定时间、再由 ``enable_motion()`` 启动 120 Hz 本地流。每个 10 Hz action 只替换完整目标；传输延迟或更新缺失时，流保持上一个目标，不会产生新的命令来源。复位、退出和流错误都会关闭 following mode。

以下条件缺一不可，物理运动才会解除锁定：

- ``motion_enabled: true`` 与 ``safety_calibrated: true``；
- MOZ hardware 配置中的左臂、右臂、躯干复位关节分别为 7/7/6 个值，并配置两个夹爪复位位置；
- 右 TCP 的六个有限 ``workspace_min`` / ``workspace_max`` 值；
- 有限的 ``gripper_position_min`` / ``gripper_position_max`` 值。

.. warning::

   只能使用已在当前机器人上复核过的标定值。不要复用其他 MOZ、推理配置或仿真场景中的数值。真机运行时应始终保持急停可达、安排操作员在场；标定变更后先采用更小的 Cartesian 动作尺度。

准备两台机器
------------

先保留 MOZ 上已经能够运行 π0.5 的运行时。厂商 ROS 包及其 ``draccus`` 依赖不会由通用 RLinf 安装脚本或 dummy Docker 镜像安装。

在 H100 主机安装与控制器无关的 RLinf 依赖：

.. code:: bash

   cd /path/to/RLinf
   bash requirements/install.sh embodied --env moz
   source .venv/bin/activate

在 MOZ 主机激活已经验证过的 π0.5 runtime，并让其中的 Python 导入当前 checkout 的共享 SDK 源码。这样 RLinf 与推理栈会使用同一份 ``packages/mozrobot``。

.. code:: bash

   source /path/to/the/pi05/runtime/bin/activate
   export RLINF_PATH=/path/to/RLinf
   export PYTHONPATH=${RLINF_PATH}:${RLINF_PATH}/packages:${PYTHONPATH}
   python -c "import mozrobot; print(mozrobot.MOZ1Robot.__name__)"

如果这条导入命令失败，应先解决现有 MOZ runtime 的不匹配问题，不要立即添加或升级依赖。特别是，当 RLinf 环境已经拥有机器人控制权时，不要同时启动 WebSocket 推理入口 ``python3 examples/moz1_real/main.py``。

为两份已提交的配置创建私有标定副本。替换 ``cluster.node_groups[].hardware`` 中全部 ``CALIBRATE_*`` 串和 ``null`` 标定字段，再设置 ``env.*.override_cfg`` 中的运行参数。站点标定值不应进入版本控制。

键盘 wrapper 读取 Linux evdev 输入。自动发现选错设备时，应在 MOZ runtime 中、启动采集或训练之前设置 ``RLINF_KEYBOARD_DEVICE=/dev/input/event<N>``。在机器人仍处于禁动状态时，先确认该设备能产生 ``a``、``b``、``c``。

采集原生遥操作示教
--------------------

示教采集是仅在 MOZ 上运行的单节点会话，与双节点训练分开。``realworld_moz_collect_data.yaml`` 中的 hardware 配置会从 ``teleop_config`` 构建现有的 MOZ 遥操作适配器。``moz_native`` 读取适配器目标，转换为 policy 使用的归一化 action，再交给 ``MOZConnection`` 下发；它不会成为第二个 SDK 写入者。

在已激活的 MOZ runtime 中启动本地 Ray head，再运行标定后的配置副本：

.. code:: bash

   export RLINF_NODE_RANK=0
   ray start --head --port=6379 --node-ip-address=<MOZ_IP>
   cd ${RLINF_PATH}
   python examples/embodiment/collect_real_data.py \
     --config-name realworld_moz_collect_data_local

踏板 ``a`` 用于开始录制或放弃当前录制，``b`` 用于添加分段边界，``c`` 用于标记成功 episode。目标是采集 50 条成功 episode。replay buffer 的输出目录为 ``<logger.log_path>/demos``；训练配置的 ``algorithm.demo_buffer.load_path`` 必须指向这里，而不是只指向导出的 ``collected_data`` 目录。

在两节点上运行在线 RLPD
--------------------------

采集结束后，停止单节点 Ray 会话，再创建两节点 cluster。H100 主机运行 head，MOZ 加入该 cluster。必须在每台机器启动 Ray 之前设置 ``RLINF_NODE_RANK``。

在 H100 主机上：

.. code:: bash

   source /path/to/RLinf/.venv/bin/activate
   export RLINF_NODE_RANK=0
   ray start --head --port=6379 --node-ip-address=<H100_IP>

在已准备好的 π0.5 runtime 中：

.. code:: bash

   export RLINF_NODE_RANK=1
   ray start --address=<H100_IP>:6379

只在 H100 head 上启动训练：

.. code:: bash

   cd /path/to/RLinf
   source .venv/bin/activate
   python examples/embodiment/train_async.py \
     --config-name realworld_moz_pick_lift_rlpd_cnn_async_local

训练配置会将 actor 和 rollout 放在 H100 节点的 GPU 0，将环境放在 MOZ hardware resource。初始 episode 最多为 150 个 policy step，即 10 Hz 下的 15 秒。在线采集时，``eval_control`` 会在每次复位后等待：踏板 ``a`` 开始 rollout，``b`` 以失败结束，``c`` 以成功结束。复位间隔给操作员重新放置桌面物体的时间。

验证首个流程
------------

在移动真机之前，先在 RLinf 环境中运行 dummy e2e 配置。它验证异步 SAC/RLPD 循环能够消费双图像、从 replay 更新并保存 checkpoint，同时不导入 MOZ ROS 包。

.. code:: bash

   export REPO_PATH=$(pwd)
   bash tests/e2e_tests/embodied/run.sh realworld_moz_dummy_sac_cnn

真机运行时需要同时观察 ``env/success_once``、replay buffer 的增长、actor 更新和 checkpoint 生成。首版任务中的非零 reward 是操作员结果标签，不是自动抓取检测器。因此首个验收标准是稳定完成采集 → replay → update → checkpoint 闭环，而不是预设抓取成功率。
