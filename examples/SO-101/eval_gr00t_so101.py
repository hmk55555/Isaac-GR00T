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

# SO101 Bimanual Real Robot
import time
from contextlib import contextmanager

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.robot_devices.cameras.configs import OpenCVCameraConfig
from lerobot.common.robot_devices.motors.dynamixel import TorqueMode
from lerobot.common.robot_devices.robots.configs import So100RobotConfig
from lerobot.common.robot_devices.robots.utils import make_robot_from_config
from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError

# NOTE:
# Sometimes we would like to abstract different env, or run this on a separate machine
# User can just move this single python class method gr00t/eval/service.py
# to their code or do the following line below
# sys.path.append(os.path.expanduser("~/Isaac-GR00T/gr00t/eval/"))
from service import ExternalRobotInferenceClient

# Import tqdm for progress bar
from tqdm import tqdm

#################################################################################


class SO101BimanualRobot:
    def __init__(self, calibrate=False, enable_camera=False, cam_indices=[9, 10, 11]):
        self.config = So100RobotConfig()
        self.calibrate = calibrate
        self.enable_camera = enable_camera
        self.cam_indices = cam_indices  # [right_cam, left_cam, top_depth_cam]
        
        if not enable_camera:
            self.config.cameras = {}
        else:
            self.config.cameras = {
                "right": OpenCVCameraConfig(cam_indices[0], 30, 640, 480, "bgr"),
                "left": OpenCVCameraConfig(cam_indices[1], 30, 640, 480, "bgr"),
                "top_depth": OpenCVCameraConfig(cam_indices[2], 30, 640, 480, "bgr"),
            }
        self.config.leader_arms = {}

        # remove the .cache/calibration/so100 folder
        if self.calibrate:
            import os
            import shutil

            calibration_folder = os.path.join(os.getcwd(), ".cache", "calibration", "so100")
            print("========> Deleting calibration_folder:", calibration_folder)
            if os.path.exists(calibration_folder):
                shutil.rmtree(calibration_folder)

        # Create the robot
        self.robot = make_robot_from_config(self.config)
        # For bimanual, we need to handle two arms
        # This is a placeholder - actual implementation would depend on SO-101 hardware setup
        self.left_motor_bus = self.robot.follower_arms.get("left", None)
        self.right_motor_bus = self.robot.follower_arms.get("right", None)

    @contextmanager
    def activate(self):
        try:
            self.connect()
            self.move_to_initial_pose()
            yield
        finally:
            self.disconnect()

    def connect(self):
        if self.robot.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                "ManipulatorRobot is already connected. Do not run `robot.connect()` twice."
            )

        # Connect the arms
        if self.left_motor_bus:
            self.left_motor_bus.connect()
        if self.right_motor_bus:
            self.right_motor_bus.connect()

        # We assume that at connection time, arms are in a rest position, and torque can
        # be safely disabled to run calibration and/or set robot preset configurations.
        if self.left_motor_bus:
            self.left_motor_bus.write("Torque_Enable", TorqueMode.DISABLED.value)
        if self.right_motor_bus:
            self.right_motor_bus.write("Torque_Enable", TorqueMode.DISABLED.value)

        # Calibrate the robot
        self.robot.activate_calibration()

        self.set_so101_robot_preset()

        # Enable torque on all motors of the follower arms
        if self.left_motor_bus:
            self.left_motor_bus.write("Torque_Enable", TorqueMode.ENABLED.value)
        if self.right_motor_bus:
            self.right_motor_bus.write("Torque_Enable", TorqueMode.ENABLED.value)
        
        print("robot present position left:", self.left_motor_bus.read("Present_Position") if self.left_motor_bus else "N/A")
        print("robot present position right:", self.right_motor_bus.read("Present_Position") if self.right_motor_bus else "N/A")
        self.robot.is_connected = True

        self.cameras = {}
        if self.enable_camera:
            for cam_name in ["right", "left", "top_depth"]:
                if cam_name in self.robot.cameras:
                    self.cameras[cam_name] = self.robot.cameras[cam_name]
                    self.cameras[cam_name].connect()
        print("================> SO101 Bimanual Robot is fully connected =================")

    def set_so101_robot_preset(self):
        # Mode=0 for Position Control
        if self.left_motor_bus:
            self.left_motor_bus.write("Mode", 0)
            self.left_motor_bus.write("P_Coefficient", 10)
            self.left_motor_bus.write("I_Coefficient", 0)
            self.left_motor_bus.write("D_Coefficient", 32)
            self.left_motor_bus.write("Lock", 0)
            self.left_motor_bus.write("Maximum_Acceleration", 254)
            self.left_motor_bus.write("Acceleration", 254)
            
        if self.right_motor_bus:
            self.right_motor_bus.write("Mode", 0)
            self.right_motor_bus.write("P_Coefficient", 10)
            self.right_motor_bus.write("I_Coefficient", 0)
            self.right_motor_bus.write("D_Coefficient", 32)
            self.right_motor_bus.write("Lock", 0)
            self.right_motor_bus.write("Maximum_Acceleration", 254)
            self.right_motor_bus.write("Acceleration", 254)

    def move_to_initial_pose(self):
        current_state = self.robot.capture_observation()["observation.state"]
        # For bimanual, we need to set initial poses for both arms
        # This is a placeholder - actual implementation would depend on SO-101 hardware setup
        left_initial_state = torch.tensor([90, 90, 90, 90, -70, 30])  # 5 arm joints + 1 gripper
        right_initial_state = torch.tensor([90, 90, 90, 90, -70, 30])  # 5 arm joints + 1 gripper
        
        if self.left_motor_bus:
            self.robot.send_action(left_initial_state, arm="left")
        if self.right_motor_bus:
            self.robot.send_action(right_initial_state, arm="right")
        time.sleep(2)
        print("-------------------------------- moving to initial pose")

    def go_home(self):
        print("-------------------------------- moving to home pose")
        left_home_state = torch.tensor([88.0664, 156.7090, 135.6152, 83.7598, -89.1211, 16.5107])
        right_home_state = torch.tensor([88.0664, 156.7090, 135.6152, 83.7598, -89.1211, 16.5107])
        
        if self.left_motor_bus:
            self.set_target_state(left_home_state, arm="left")
        if self.right_motor_bus:
            self.set_target_state(right_home_state, arm="right")
        time.sleep(2)

    def get_observation(self):
        return self.robot.capture_observation()

    def get_current_state(self):
        obs = self.get_observation()["observation.state"].data.numpy()
        # For bimanual, we need to separate left and right arm states
        # This assumes the state is concatenated as [left_arm(5), gripper1(1), right_arm(5), gripper2(1)]
        left_arm_state = obs[1:6]  # indices 1-5 for left arm
        gripper1_state = obs[5:6]  # index 5 for gripper1
        right_arm_state = obs[1:6]  # indices 1-5 for right arm (assuming same structure)
        gripper2_state = obs[5:6]  # index 5 for gripper2
        
        return {
            "left_arm": left_arm_state,
            "gripper1": gripper1_state,
            "right_arm": right_arm_state,
            "gripper2": gripper2_state,
        }

    def get_current_imgs(self):
        imgs = {}
        for cam_name in ["right", "left", "top_depth"]:
            if cam_name in self.cameras:
                img = self.get_observation()[f"observation.images.{cam_name}"].data.numpy()
                # convert bgr to rgb
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                imgs[cam_name] = img
        return imgs

    def set_target_state(self, target_state: torch.Tensor, arm: str = "both"):
        if arm == "left" and self.left_motor_bus:
            self.robot.send_action(target_state, arm="left")
        elif arm == "right" and self.right_motor_bus:
            self.robot.send_action(target_state, arm="right")
        elif arm == "both":
            if self.left_motor_bus:
                self.robot.send_action(target_state, arm="left")
            if self.right_motor_bus:
                self.robot.send_action(target_state, arm="right")

    def enable(self):
        if self.left_motor_bus:
            self.left_motor_bus.write("Torque_Enable", TorqueMode.ENABLED.value)
        if self.right_motor_bus:
            self.right_motor_bus.write("Torque_Enable", TorqueMode.ENABLED.value)

    def disable(self):
        if self.left_motor_bus:
            self.left_motor_bus.write("Torque_Enable", TorqueMode.DISABLED.value)
        if self.right_motor_bus:
            self.right_motor_bus.write("Torque_Enable", TorqueMode.DISABLED.value)

    def disconnect(self):
        self.disable()
        self.robot.disconnect()
        self.robot.is_connected = False
        print("================> SO101 Bimanual Robot disconnected")

    def __del__(self):
        self.disconnect()


#################################################################################


class Gr00tBimanualRobotInferenceClient:
    def __init__(
        self,
        host="localhost",
        port=5555,
        language_instruction="Pick up the fruits with both hands and place them on the plate.",
    ):
        self.language_instruction = language_instruction
        # 480, 640
        self.img_size = (480, 640)
        self.policy = ExternalRobotInferenceClient(host=host, port=port)

    def get_action(self, imgs, state):
        obs_dict = {
            "video.right": imgs["right"][np.newaxis, :, :, :],
            "video.left": imgs["left"][np.newaxis, :, :, :],
            "video.top_depth": imgs["top_depth"][np.newaxis, :, :, :],
            "state.left_arm": state["left_arm"][np.newaxis, :].astype(np.float64),
            "state.gripper1": state["gripper1"][np.newaxis, :].astype(np.float64),
            "state.right_arm": state["right_arm"][np.newaxis, :].astype(np.float64),
            "state.gripper2": state["gripper2"][np.newaxis, :].astype(np.float64),
            "annotation.human.task_description": [self.language_instruction],
        }
        res = self.policy.get_action(obs_dict)
        return res

    def sample_action(self):
        obs_dict = {
            "video.right": np.zeros((1, self.img_size[0], self.img_size[1], 3), dtype=np.uint8),
            "video.left": np.zeros((1, self.img_size[0], self.img_size[1], 3), dtype=np.uint8),
            "video.top_depth": np.zeros((1, self.img_size[0], self.img_size[1], 3), dtype=np.uint8),
            "state.left_arm": np.zeros((1, 4)),  # 4 arm joints
            "state.gripper1": np.zeros((1, 1)),
            "state.right_arm": np.zeros((1, 4)),  # 4 arm joints
            "state.gripper2": np.zeros((1, 1)),
            "annotation.human.task_description": [self.language_instruction],
        }
        return self.policy.get_action(obs_dict)

    def set_lang_instruction(self, lang_instruction):
        self.language_instruction = lang_instruction


#################################################################################


def view_imgs(imgs, imgs2=None):
    """
    This is a matplotlib viewer for multiple camera views
    """
    fig, axes = plt.subplots(1, len(imgs), figsize=(15, 5))
    if len(imgs) == 1:
        axes = [axes]
    
    for i, (cam_name, img) in enumerate(imgs.items()):
        axes[i].imshow(img)
        axes[i].set_title(cam_name)
        axes[i].axis("off")
        
        if imgs2 and cam_name in imgs2:
            axes[i].imshow(imgs2[cam_name], alpha=0.5)
    
    plt.pause(0.001)  # Non-blocking show
    plt.clf()  # Clear the figure for the next frame


#################################################################################

if __name__ == "__main__":
    import argparse
    import os

    default_dataset_path = os.path.expanduser("~/datasets/so101_bimanual")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_policy", action="store_true"
    )  # default is to playback the provided dataset
    parser.add_argument("--dataset_path", type=str, default=default_dataset_path)
    parser.add_argument("--host", type=str, default="10.110.17.183")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--action_horizon", type=int, default=12)
    parser.add_argument("--actions_to_execute", type=int, default=350)
    parser.add_argument("--cam_indices", type=int, nargs=3, default=[1, 2, 3])
    parser.add_argument(
        "--lang_instruction", type=str, default="Pick up the fruits with both hands and place them on the plate."
    )
    parser.add_argument("--record_imgs", action="store_true")
    args = parser.parse_args()

    # print lang_instruction
    print("lang_instruction: ", args.lang_instruction)

    ACTIONS_TO_EXECUTE = args.actions_to_execute
    USE_POLICY = args.use_policy
    ACTION_HORIZON = (
        args.action_horizon
    )  # we will execute only some actions from the action_chunk of 16
    MODALITY_KEYS = ["left_arm", "gripper1", "right_arm", "gripper2"]
    
    if USE_POLICY:
        client = Gr00tBimanualRobotInferenceClient(
            host=args.host,
            port=args.port,
            language_instruction=args.lang_instruction,
        )

        if args.record_imgs:
            # create a folder to save the images and delete all the images in the folder
            os.makedirs("eval_images", exist_ok=True)
            for file in os.listdir("eval_images"):
                os.remove(os.path.join("eval_images", file))

        robot = SO101BimanualRobot(calibrate=False, enable_camera=True, cam_indices=args.cam_indices)
        image_count = 0
        with robot.activate():
            for i in tqdm(range(ACTIONS_TO_EXECUTE), desc="Executing actions"):
                imgs = robot.get_current_imgs()
                view_imgs(imgs)
                state = robot.get_current_state()
                action = client.get_action(imgs, state)
                start_time = time.time()
                
                for i in range(ACTION_HORIZON):
                    # Concatenate actions for both arms
                    concat_action = np.concatenate(
                        [np.atleast_1d(action[f"action.{key}"][i]) for key in MODALITY_KEYS],
                        axis=0,
                    )
                    assert concat_action.shape == (10,), concat_action.shape  # 4+1+4+1 = 10
                    
                    # Split action for left and right arms
                    left_action = torch.from_numpy(concat_action[:5])  # left_arm + gripper1
                    right_action = torch.from_numpy(concat_action[5:])  # right_arm + gripper2
                    
                    robot.set_target_state(left_action, arm="left")
                    robot.set_target_state(right_action, arm="right")
                    time.sleep(0.02)

                    # get the realtime images
                    imgs = robot.get_current_imgs()
                    view_imgs(imgs)

                    if args.record_imgs:
                        # resize and save images
                        for cam_name, img in imgs.items():
                            img_resized = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), (320, 240))
                            cv2.imwrite(f"eval_images/{cam_name}_img_{image_count}.jpg", img_resized)
                        image_count += 1

                    # 0.05*16 = 0.8 seconds
                    print("executing action", i, "time taken", time.time() - start_time)
                print("Action chunk execution time taken", time.time() - start_time)
    else:
        # Test Dataset Source - placeholder for SO-101 bimanual dataset
        dataset = LeRobotDataset(
            repo_id="",
            root=args.dataset_path,
        )

        robot = SO101BimanualRobot(calibrate=False, enable_camera=True, cam_indices=args.cam_indices)

        with robot.activate():
            print("Run replay of the dataset")
            actions = []
            for i in tqdm(range(ACTIONS_TO_EXECUTE), desc="Loading actions"):
                action = dataset[i]["action"]
                imgs = {}
                for cam_name in ["right", "left", "top_depth"]:
                    img = dataset[i][f"observation.images.{cam_name}"].data.numpy()
                    img = img.transpose(1, 2, 0)
                    imgs[cam_name] = img
                
                realtime_imgs = robot.get_current_imgs()
                view_imgs(imgs, realtime_imgs)
                actions.append(action)
                
                # Split action for both arms
                left_action = action[:5]  # left_arm + gripper1
                right_action = action[5:]  # right_arm + gripper2
                robot.set_target_state(torch.from_numpy(left_action), arm="left")
                robot.set_target_state(torch.from_numpy(right_action), arm="right")
                time.sleep(0.05)

            # plot the actions
            plt.plot(actions)
            plt.show()

            print("Done all actions")
            robot.go_home()
            print("Done home")
