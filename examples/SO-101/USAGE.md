# SO-101 Bimanual GR00T Evaluation Guide

## Prerequisites

1. **Install the bi_so101_follower package**:
   ```bash
   cd /Users/himankhanda/Downloads/lerobot_robot_bi_so101_follower-0.1.1
   pip install -e .
   ```

2. **Verify installation**:
   ```bash
   python -c "from lerobot.robots.config import RobotConfig; print('bi_so101_follower registered:', 'bi_so101_follower' in RobotConfig._registry)"
   ```
   Should print: `bi_so101_follower registered: True`

## Hardware Setup

1. Connect left arm to USB port (e.g., `/dev/ttyACM1`)
2. Connect right arm to USB port (e.g., `/dev/ttyACM0`)
3. Connect cameras:
   - Camera 1: Right view
   - Camera 2: Left view  
   - Camera 3: Top/depth view

## Running the Evaluation

### Step 1: Start the Policy Server

In one terminal:
```bash
cd ~/Isaac-GR00T
conda activate gr00t

python scripts/inference_service.py --server \
    --model_path ./so101-checkpoints \
    --embodiment-tag new_embodiment \
    --data-config so101_bimanual \
    --denoising-steps 4
```

### Step 2: Run the Eval Script

In another terminal:
```bash
cd ~/Isaac-GR00T
conda activate gr00t

python examples/SO-101/eval_gr00t_so101_new.py \
    --robot.type=bi_so101_follower \
    --robot.left_arm_port=/dev/ttyACM1 \
    --robot.right_arm_port=/dev/ttyACM0 \
    --robot.id=follower \
    --robot.cameras="{ right: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}, left: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, top_depth: {type: opencv, index_or_path: 3, width: 640, height: 480, fps: 30}}" \
    --policy_host=192.168.0.110 \
    --policy_port=5555 \
    --action_horizon=12 \
    --lang_instruction="pour matcha"
```

**Adjust these parameters:**
- `--robot.left_arm_port`: Your left arm's USB port
- `--robot.right_arm_port`: Your right arm's USB port
- `--robot.cameras`: Update `index_or_path` to match your camera indices
- `--policy_host`: IP address where inference server is running
- `--lang_instruction`: Task description from your training dataset

## Understanding the Motor Order

The script expects motors in this exact order (matching your dataset):

**Left Arm (indices 0-5):**
1. `left_shoulder_pan.pos`
2. `left_shoulder_lift.pos`
3. `left_elbow_flex.pos`
4. `left_wrist_flex.pos`
5. `left_wrist_roll.pos`
6. `left_gripper.pos`

**Right Arm (indices 6-11):**
7. `right_shoulder_pan.pos`
8. `right_shoulder_lift.pos`
9. `right_elbow_flex.pos`
10. `right_wrist_flex.pos`
11. `right_wrist_roll.pos`
12. `right_gripper.pos`

This matches your `info.json` and `modality.json`.

## Data Flow

```
Robot Observation (12 motors + 3 cameras)
    ↓
[left_shoulder_pan, ..., left_gripper,    ← indices 0-5
 right_shoulder_pan, ..., right_gripper]  ← indices 6-11
    ↓
GR00T Format (via modality.json):
    state.left_arm: [0:5]     (5 joints)
    state.gripper1: [5:6]     (1 gripper)
    state.right_arm: [6:11]   (5 joints)
    state.gripper2: [11:12]   (1 gripper)
    video.right, video.left, video.top_depth
    ↓
GR00T Model Inference
    ↓
Action (same structure as state)
    ↓
Convert back to robot format
    ↓
Send to bi_so101_follower
```

## Troubleshooting

### Motor key not found error
- Verify bi_so101_follower is installed correctly
- Check that both arms are connected to the correct ports

### Camera connection error
- Verify camera indices with `ls /dev/video*`
- Test cameras individually with `v4l2-ctl --list-devices`

### Inference server connection error
- Check that inference_service.py is running
- Verify network connectivity: `ping 192.168.0.110`
- Check firewall settings allow port 5555

### Wrong motor order warning
- This means the robot's motor names don't match the expected order
- Check your bi_so101_follower installation
- Verify your dataset's info.json has the correct motor order

## Control Rate

- **Action frequency**: 50 Hz (20ms per action)
- **Action horizon**: 12 (default)
- **Chunk execution time**: ~240ms per chunk
- **Inference time**: Varies (check output for timing)

## Stopping the Evaluation

Press `Ctrl+C` to safely stop the evaluation. The robot will disconnect gracefully.

