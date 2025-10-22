# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
SO-101 Bimanual Gr00T policy eval script using bi_so101_follower from pip package.

Example command:

```shell
python eval_gr00t_so101_new.py \
    --robot.type=bi_so101_follower \
    --robot.left_arm_port=/dev/ttyACM1 \
    --robot.right_arm_port=/dev/ttyACM0 \
    --robot.id=follower \
    --robot.cameras="{ right: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}, left: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, top_depth: {type: opencv, index_or_path: 3, width: 640, height: 480, fps: 30}}" \
    --policy_host=192.168.0.110 \
    --lang_instruction="pour matcha"
```

Note: This script uses bi_so101_follower from your pip package. The robot config is automatically
registered via @RobotConfig.register_subclass("bi_so101_follower") decorator.
"""

import logging
import sys
import time
from dataclasses import asdict, dataclass
from pprint import pformat

import draccus
import matplotlib.pyplot as plt
import numpy as np
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401

# Import bi_so101_follower to trigger robot registration
try:
    import lerobot_robot_bi_so101_follower  # noqa: F401
except ImportError:
    print("Warning: lerobot-robot-bi-so101-follower package not found!")
    print("Install it with: pip install lerobot-robot-bi-so101-follower")

from lerobot.robots import (
    Robot,
    RobotConfig,
    make_robot_from_config,
)
from lerobot.utils.utils import init_logging, log_say

# NOTE:
# Sometimes we would like to abstract different env, or run this on a separate machine
# User can just move this single python class method gr00t/eval/service.py
# to their code or do the following line below
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
gr00t_eval_path = os.path.join(script_dir, '..', '..', 'gr00t', 'eval')
sys.path.insert(0, gr00t_eval_path)

from service import ExternalRobotInferenceClient

#################################################################################


class Gr00tBimanualRobotInferenceClient:
    """
    Bimanual inference client for SO-101.
    
    Modality structure (based on modality.json):
    - state: left_arm (5), gripper1 (1), right_arm (5), gripper2 (1) = 12 total
    - action: same as state
    - video: right, left, top_depth
    """

    def __init__(
        self,
        host="localhost",
        port=5555,
        camera_keys=[],
        robot_state_keys=[],
        show_images=False,
    ):
        self.policy = ExternalRobotInferenceClient(host=host, port=port)
        self.camera_keys = camera_keys
        self.robot_state_keys = robot_state_keys
        self.show_images = show_images
        
        # For bimanual: left_arm (5) + gripper1 (1) + right_arm (5) + gripper2 (1) = 12
        assert (
            len(robot_state_keys) == 12
        ), f"robot_state_keys should be size 12 for bimanual, but got {len(robot_state_keys)}"
        
        # Modality keys for bimanual setup
        self.modality_keys = ["left_arm", "gripper1", "right_arm", "gripper2"]

    def get_action(self, observation_dict, lang: str):
        # First add the images
        obs_dict = {f"video.{key}": observation_dict[key] for key in self.camera_keys}

        # Show images
        if self.show_images:
            view_img(obs_dict)

        # Make all single float value of dict[str, float] state into a single array
        # robot_state_keys order: [left_shoulder_pan, left_shoulder_lift, left_elbow_flex,
        #                           left_wrist_flex, left_wrist_roll, left_gripper,
        #                           right_shoulder_pan, right_shoulder_lift, right_elbow_flex,
        #                           right_wrist_flex, right_wrist_roll, right_gripper]
        state = np.array([observation_dict[k] for k in self.robot_state_keys])
        
        # Split state for bimanual according to modality.json:
        # state.left_arm: indices 0-4 (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll)
        # state.gripper1: index 5 (left_gripper)
        # state.right_arm: indices 6-10 (right motors)
        # state.gripper2: index 11 (right_gripper)
        obs_dict["state.left_arm"] = state[0:5].astype(np.float64)
        obs_dict["state.gripper1"] = state[5:6].astype(np.float64)
        obs_dict["state.right_arm"] = state[6:11].astype(np.float64)
        obs_dict["state.gripper2"] = state[11:12].astype(np.float64)
        obs_dict["annotation.human.task_description"] = lang

        # Add a dummy dimension of np.array([1, ...]) to all the keys (assume history is 1)
        for k in obs_dict:
            if isinstance(obs_dict[k], np.ndarray):
                obs_dict[k] = obs_dict[k][np.newaxis, ...]
            else:
                obs_dict[k] = [obs_dict[k]]

        # Get the action chunk via the policy server
        action_chunk = self.policy.get_action(obs_dict)

        # Convert the action chunk to a list of dict[str, float]
        lerobot_actions = []
        action_horizon = action_chunk[f"action.{self.modality_keys[0]}"].shape[0]
        for i in range(action_horizon):
            action_dict = self._convert_to_lerobot_action(action_chunk, i)
            lerobot_actions.append(action_dict)
        return lerobot_actions

    def _convert_to_lerobot_action(
        self, action_chunk: dict[str, np.array], idx: int
    ) -> dict[str, float]:
        """
        Convert the action chunk to a dict[str, float] for bimanual control.
        """
        concat_action = np.concatenate(
            [np.atleast_1d(action_chunk[f"action.{key}"][idx]) for key in self.modality_keys],
            axis=0,
        )
        assert len(concat_action) == len(self.robot_state_keys), (
            f"Action size mismatch: expected {len(self.robot_state_keys)}, got {len(concat_action)}"
        )
        
        # Convert the action to dict[str, float]
        action_dict = {key: concat_action[i] for i, key in enumerate(self.robot_state_keys)}
        return action_dict


#################################################################################


def view_img(img_dict, overlay_img=None):
    """
    Display camera views using matplotlib.
    """
    if isinstance(img_dict, dict):
        # Stack the images horizontally
        imgs = []
        for k in img_dict:
            img = img_dict[k]
            # Remove batch dimension if present
            if img.ndim == 4:
                img = img[0]
            imgs.append(img)
        img = np.concatenate(imgs, axis=1)
    else:
        img = img_dict

    plt.imshow(img)
    plt.title("Camera Views")
    plt.axis("off")
    plt.pause(0.001)  # Non-blocking show
    plt.clf()  # Clear the figure for the next frame


def print_yellow(text):
    print("\033[93m {}\033[00m".format(text))


@dataclass
class EvalConfig:
    robot: RobotConfig  # Bimanual robot configuration (bi_so101_follower)
    policy_host: str = "192.168.0.110"  # Host of the gr00t server
    policy_port: int = 5555  # Port of the gr00t server
    action_horizon: int = 12  # Number of actions to execute from the action chunk
    lang_instruction: str = "pour matcha"
    play_sounds: bool = False  # Whether to play sounds
    timeout: int = 60  # Timeout in seconds
    show_images: bool = False  # Whether to show images


@draccus.wrap()
def eval(cfg: EvalConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    # Step 1: Initialize the bimanual robot
    print_yellow("Initializing bimanual SO-101 robot...")
    robot = make_robot_from_config(cfg.robot)
    robot.connect()

    # Get camera keys from robot config
    camera_keys = list(cfg.robot.cameras.keys())
    print("camera_keys:", camera_keys)

    log_say("Initializing bimanual robot", cfg.play_sounds, blocking=True)

    language_instruction = cfg.lang_instruction

    # Get motor keys from bimanual robot in the EXACT order from dataset
    # This order must match info.json exactly!
    # From info.json: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
    robot_state_keys = [
        "left_shoulder_pan.pos",
        "left_shoulder_lift.pos",
        "left_elbow_flex.pos",
        "left_wrist_flex.pos",
        "left_wrist_roll.pos",
        "left_gripper.pos",
        "right_shoulder_pan.pos",
        "right_shoulder_lift.pos",
        "right_elbow_flex.pos",
        "right_wrist_flex.pos",
        "right_wrist_roll.pos",
        "right_gripper.pos",
    ]
    
    # Verify all keys exist in robot
    all_motor_keys = set(robot._motors_ft.keys())
    for key in robot_state_keys:
        if key not in all_motor_keys:
            raise ValueError(f"Motor key '{key}' not found in robot! Available: {all_motor_keys}")
    
    print("robot_state_keys (in dataset order):", robot_state_keys)
    print(f"Total state dimension: {len(robot_state_keys)}")

    # Step 2: Initialize the policy
    policy = Gr00tBimanualRobotInferenceClient(
        host=cfg.policy_host,
        port=cfg.policy_port,
        camera_keys=camera_keys,
        robot_state_keys=robot_state_keys,
        show_images=cfg.show_images,
    )
    log_say(
        "Initializing policy client with language instruction: " + language_instruction,
        cfg.play_sounds,
        blocking=True,
    )

    # Step 3: Run the Eval Loop
    print_yellow("\n" + "="*70)
    print_yellow("READY TO START!")
    print_yellow("="*70)
    print(f"Language instruction: '{language_instruction}'")
    print(f"Action horizon: {cfg.action_horizon}")
    print(f"Policy server: {cfg.policy_host}:{cfg.policy_port}")
    print_yellow("="*70 + "\n")
    print_yellow("Press Ctrl+C to stop\n")
    
    action_count = 0
    try:
        while True:
            # Get observations from bimanual robot (includes both arms and cameras)
            observation_dict = robot.get_observation()
            
            # Only print observation keys on first iteration
            if action_count == 0:
                print(f"Observation keys: {list(observation_dict.keys())}")
            
            # Get action chunk from policy
            start_time = time.perf_counter()
            action_chunk = policy.get_action(observation_dict, language_instruction)
            inference_time = (time.perf_counter() - start_time) * 1000
            
            print(f"\n[Chunk {action_count}] Inference: {inference_time:.1f}ms")

            # Execute actions
            chunk_start = time.perf_counter()
            for i in range(cfg.action_horizon):
                action_dict = action_chunk[i]
                
                # Send action to bimanual robot (handles both arms internally)
                robot.send_action(action_dict)
                
                time.sleep(0.02)  # 20ms per action = 50Hz control rate
            
            chunk_time = (time.perf_counter() - chunk_start) * 1000
            print(f"[Chunk {action_count}] Execution: {chunk_time:.1f}ms ({cfg.action_horizon} actions)")
            action_count += 1
                
    except KeyboardInterrupt:
        print_yellow(f"\n\nEvaluation stopped by user after {action_count} action chunks")
    finally:
        print_yellow("Disconnecting robot...")
        robot.disconnect()
        print_yellow("Done!")


if __name__ == "__main__":
    eval()
