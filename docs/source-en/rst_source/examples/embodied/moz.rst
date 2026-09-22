Real-World RLPD on MOZ
======================

This guide brings a MOZ robot from its existing π0.5 inference runtime to a safe, first real-world RLPD experiment: collect successful right-arm pick-and-lift demonstrations, then train a CNN policy on an H100 node while MOZ executes 10 Hz actions locally. It covers the initial fixed-tabletop task only; do not use this recipe for mobile-base control, left-arm control, or online π0.5 fine-tuning.

Overview
--------

The workflow keeps learning traffic away from the robot controller. The H100 machine hosts one colocated actor and rollout GPU. The MOZ computer hosts the only environment worker, the ROS SDK session, native teleoperation, image capture, and a 120 Hz local command hold loop.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Model
      :text-align: center

      CNN policy · ResNet-10 encoder

   .. grid-item-card:: Algorithm
      :text-align: center

      SAC / RLPD

   .. grid-item-card:: Task
      :text-align: center

      Right-arm fixed-object pick and lift

   .. grid-item-card:: Hardware
      :text-align: center

      MOZ · head RGB · right-wrist RGB · H100

The shipped files are deliberately safe templates. They open no external following mode until a private site configuration supplies reviewed reset joints, a Cartesian workspace, and gripper limits.

Tasks
-----

Use the dummy task first to verify the RL path, then collect demonstrations on the robot, and finally start online RLPD.

.. list-table::
   :header-rows: 1
   :widths: 28 38 34

   * - Stage
     - Config / entry point
     - Result
   * - Dummy RLPD smoke test
     - ``realworld_moz_dummy_sac_cnn.yaml``
     - Runs the 14-D state, two-image, 7-D action contract without the vendor SDK.
   * - Native demonstration collection
     - ``realworld_moz_collect_data.yaml`` + ``collect_real_data.py``
     - Saves only episodes ended successfully with the MOZ operator controls.
   * - Online training
     - ``realworld_moz_pick_lift_rlpd_cnn_async.yaml`` + ``train_async.py``
     - Combines the successful demonstrations with online experience.

Observation and Action
----------------------

The policy receives one snapshot per 10 Hz step. Keeping this contract fixed is important: changing state order, camera order, or the gripper range invalidates both demonstrations and a checkpoint trained from them.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Contract
   * - State
     - 14 values: ``right_gripper`` (1), ``right_joint_positions`` (7), and ``right_tcp_pose`` (6). RLinf concatenates these sorted state keys in that order.
   * - Images
     - ``head_rgb`` is the main image and ``right_wrist_rgb`` is the extra view. Both are uint8 frames with shape ``[224, 224, 3]``.
   * - Action
     - Seven normalized values: three translation deltas, three rotation-vector deltas, and one gripper command. The initial scale is 1 cm and 0.05 rad per step; the gripper maps from ``[-1, 1]`` into its calibrated physical interval.
   * - Reward
     - The operator labels online outcomes: pedal ``c`` returns 1 for success and pedal ``b`` returns 0 for failure.

How Command Ownership Stays Safe
--------------------------------

The real-world environment, ``MozPickLiftEnv``, produces a bounded seven-value action. It does not call the vendor SDK directly. ``MOZConnection`` owns the SDK session on the MOZ environment worker and is the only writer to the robot controller.

On connection, the SDK is opened with ``auto_reset=False`` and following mode remains disabled. A physical reset first calls ``reset_to_home()``, waits for the configured settle time, and then ``enable_motion()`` starts the 120 Hz local stream. Each 10 Hz action replaces the stream's complete target; a delayed or missing update holds the last target instead of creating a new command source. Reset, shutdown, and any stream error leave following mode disabled.

The configuration must set all of the following before it can move hardware:

- ``motion_enabled: true`` and ``safety_calibrated: true``;
- 7/7/6 calibrated left-arm, right-arm, and torso home joints plus two gripper homes in the MOZ hardware block;
- six finite ``workspace_min`` / ``workspace_max`` values for the right TCP pose;
- finite ``gripper_position_min`` / ``gripper_position_max`` values.

.. warning::

   Fill these values only from a reviewed on-robot calibration. Do not copy them from another MOZ, an inference config, or a simulation scene. Keep the emergency stop reachable, leave an operator at the robot, and start with reduced Cartesian scales if a calibration changes.

Prepare the Two Machines
------------------------

Start by preserving the runtime that already performs π0.5 inference on MOZ. The vendor ROS packages and their ``draccus`` dependency are not installed by the generic RLinf installer or the dummy Docker image.

On the H100 machine, install controller-independent RLinf dependencies:

.. code:: bash

   cd /path/to/RLinf
   bash requirements/install.sh embodied --env moz
   source .venv/bin/activate

On the MOZ machine, activate the known-good π0.5 runtime and make this checkout's shared SDK source importable. This makes RLinf and the inference stack use the same ``packages/mozrobot`` source.

.. code:: bash

   source /path/to/the/pi05/runtime/bin/activate
   export RLINF_PATH=/path/to/RLinf
   export PYTHONPATH=${RLINF_PATH}:${RLINF_PATH}/packages:${PYTHONPATH}
   python -c "import mozrobot; print(mozrobot.MOZ1Robot.__name__)"

If this import fails, stop here. Resolve the mismatch in the existing MOZ runtime before adding or upgrading dependencies. In particular, do not run the WebSocket inference entry point, ``python3 examples/moz1_real/main.py``, while an RLinf environment owns the robot.

Create a private calibrated copy of both shipped configs. Replace every ``CALIBRATE_*`` serial and every ``null`` calibration field under ``cluster.node_groups[].hardware``. Then set the runtime values in ``env.*.override_cfg``. Keep these site values outside version control.

The keyboard wrappers read Linux evdev input. If automatic discovery selects the wrong device, set ``RLINF_KEYBOARD_DEVICE=/dev/input/event<N>`` in the MOZ runtime before launching collection or training. Confirm that the configured device emits ``a``, ``b``, and ``c`` while the robot remains motion-locked.

Collect Native Teleoperation Demonstrations
-------------------------------------------

Demonstration collection is a one-node MOZ session, separate from two-node training. The hardware block in ``realworld_moz_collect_data.yaml`` constructs the existing MOZ teleoperation adapter from ``teleop_config``. ``moz_native`` reads the adapter target, converts it into the same normalized action that the policy uses, and sends it through ``MOZConnection``; it never becomes a second SDK writer.

Start a local Ray head in the activated MOZ runtime, then run your calibrated copy:

.. code:: bash

   export RLINF_NODE_RANK=0
   ray start --head --port=6379 --node-ip-address=<MOZ_IP>
   cd ${RLINF_PATH}
   python examples/embodiment/collect_real_data.py \
     --config-name realworld_moz_collect_data_local

Use pedal ``a`` to begin recording or abort the current recording, ``b`` to add a segment boundary, and ``c`` to mark a successful episode. The collector target is 50 successful episodes. Its replay-buffer output is ``<logger.log_path>/demos``; set the training config's ``algorithm.demo_buffer.load_path`` to that directory, not merely to the exported ``collected_data`` directory.

Run Online RLPD on Two Nodes
----------------------------

After collection, stop the one-node Ray session and bring up one two-node cluster. Start the head on the H100 machine and join it from MOZ. Set ``RLINF_NODE_RANK`` before starting Ray on each machine.

On the H100 machine:

.. code:: bash

   source /path/to/RLinf/.venv/bin/activate
   export RLINF_NODE_RANK=0
   ray start --head --port=6379 --node-ip-address=<H100_IP>

On the MOZ machine, in the π0.5 runtime prepared above:

.. code:: bash

   export RLINF_NODE_RANK=1
   ray start --address=<H100_IP>:6379

Launch training only from the H100 head:

.. code:: bash

   cd /path/to/RLinf
   source .venv/bin/activate
   python examples/embodiment/train_async.py \
     --config-name realworld_moz_pick_lift_rlpd_cnn_async_local

The training config places actor and rollout on GPU 0 of the H100 node and the environment on the MOZ hardware resource. The initial episode lasts at most 150 policy steps, or 15 seconds at 10 Hz. During online collection, ``eval_control`` waits after each reset: pedal ``a`` starts the rollout, ``b`` ends it as a failure, and ``c`` ends it as a success. The reset boundary gives the operator time to restore the tabletop object.

Verify the First Run
--------------------

Before moving the robot, run the dummy e2e configuration in an RLinf environment. It checks that the asynchronous SAC/RLPD loop can consume the two images, update from replay, and save the configured checkpoints without importing MOZ ROS packages.

.. code:: bash

   export REPO_PATH=$(pwd)
   bash tests/e2e_tests/embodied/run.sh realworld_moz_dummy_sac_cnn

For a physical run, watch ``env/success_once``, replay-buffer growth, actor updates, and checkpoint creation together. A nonzero success label is only an operator outcome label in this first task; it is not an automatic grasp detector. The first acceptance target is therefore a stable collection → replay → update → checkpoint loop, not a target grasp success rate.
