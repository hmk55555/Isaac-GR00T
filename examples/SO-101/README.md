# SO-101 Bimanual Robot Configuration

This directory contains the configuration files and evaluation scripts for the SO-101 bimanual robot setup with GR00T N1.

## Files

- `so101_bimanual__modality.json`: Modality configuration for SO-101 bimanual robot
- `custom_data_config.py`: Data configuration class for SO-101 bimanual
- `eval_gr00t_so101.py`: Evaluation script for running GR00T N1 on SO-101 bimanual robot
- `README.md`: This documentation file

## Modality Configuration

The `so101_bimanual__modality.json` file defines the observation and action space mappings for the SO-101 bimanual robot:

### State Space
- `left_arm`: 4 joint positions (indices 1-5)
- `gripper1`: 1 gripper position (index 5-6)
- `right_arm`: 4 joint positions (indices 1-5) 
- `gripper2`: 1 gripper position (index 5-6)

### Action Space
- `left_arm`: 4 joint position targets (indices 1-5)
- `gripper1`: 1 gripper position target (index 5-6)
- `right_arm`: 4 joint position targets (indices 1-5)
- `gripper2`: 1 gripper position target (index 5-6)

### Video Observations
- `right`: Right camera view (`observation.images.right`)
- `left`: Left camera view (`observation.images.left`)
- `top_depth`: Top depth camera view (`observation.images.top_depth`)

### Annotations
- `human.task_description`: Task description from `task_index`

## Data Configuration

The `So101BimanualDataConfig` class in `custom_data_config.py` provides:

- **Video keys**: `video.right`, `video.left`, `video.top_depth`
- **State keys**: `state.left_arm`, `state.gripper1`, `state.right_arm`, `state.gripper2`
- **Action keys**: `action.left_arm`, `action.gripper1`, `action.right_arm`, `action.gripper2`
- **Language keys**: `annotation.human.task_description`

The configuration includes standard video transforms (resize, crop, color jitter) and state/action normalization using min-max scaling.

## Usage

### 1. Dataset Preparation

Download your SO-101 bimanual dataset and copy the modality configuration:

```bash
# Copy the modality configuration to your dataset
cp examples/SO-101/so101_bimanual__modality.json /path/to/your/dataset/meta/modality.json
```

### 2. Model Fine-tuning

Use the custom data configuration for fine-tuning:

```bash
python scripts/gr00t_finetune.py \
    --dataset-path /path/to/your/so101_dataset/ \
    --data_config examples.SO-101.custom_data_config:So101BimanualDataConfig \
    --num-gpus 8 \
    --batch-size 90 \
    --output-dir /tmp/so101-checkpoints \
    --max-steps 60000
```

### 3. Model Evaluation

#### Start the Inference Service

```bash
python scripts/inference_service.py \
    --model_path /path/to/your/checkpoint \
    --server \
    --data_config examples.SO-101.custom_data_config:So101BimanualDataConfig \
    --embodiment_tag new_embodiment
```

#### Run Evaluation

```bash
python examples/SO-101/eval_gr00t_so101.py \
    --use_policy \
    --host localhost \
    --port 5555 \
    --cam_indices 1 2 3 \
    --lang_instruction "Pick up the fruits with both hands and place them on the plate." \
    --actions_to_execute 350 \
    --action_horizon 12
```

#### Dataset Playback

```bash
python examples/SO-101/eval_gr00t_so101.py \
    --dataset_path /path/to/your/so101_dataset \
    --cam_indices 1 2 3 \
    --actions_to_execute 350
```

## Robot Setup Notes

⚠️ **Important**: The evaluation script assumes a bimanual robot setup with:

1. **Two arms**: Left and right arms with independent control
2. **Three cameras**: Right view, left view, and top depth camera
3. **Motor buses**: Separate motor buses for left and right arms
4. **State concatenation**: States are concatenated as `[left_arm(4), gripper1(1), right_arm(4), gripper2(1)]`

The current implementation is a template that may need modification based on your specific SO-101 hardware configuration. Key areas that may need adjustment:

- Motor bus configuration and connection logic
- Camera indices and setup
- State/action splitting and concatenation
- Robot preset configurations
- Initial pose definitions

## Hardware Requirements

- SO-101 bimanual robot with two arms
- Three cameras (right, left, top depth)
- Dynamixel motors for both arms
- Appropriate USB connections for cameras and motors

## Troubleshooting

1. **Camera Issues**: Verify camera indices match your hardware setup
2. **Motor Connection**: Ensure both left and right motor buses are properly configured
3. **State Dimensions**: Verify state concatenation matches your robot's configuration
4. **Action Execution**: Check that action splitting correctly maps to left/right arms

## Example Commands

### Basic Policy Evaluation
```bash
python examples/SO-101/eval_gr00t_so101.py --use_policy --lang_instruction "Pick up the object with both hands"
```

### Record Evaluation Images
```bash
python examples/SO-101/eval_gr00t_so101.py --use_policy --record_imgs --actions_to_execute 100
```

### Custom Camera Setup
```bash
python examples/SO-101/eval_gr00t_so101.py --use_policy --cam_indices 0 1 2
```

For more detailed information about GR00T N1 fine-tuning and evaluation, refer to the main documentation in the `getting_started/` directory.
