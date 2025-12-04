"""Robot arm visual RL lab scaffold.

Implements the requirements for Лабораторная работа №6 with:

* PyBullet + Gymnasium env that provides grayscale framestack pixels and proprioception.
* Continuous tool-space actions, frame skipping, custom CNN feature extractor, and reward shaping.
* Stable-Baselines3 PPO training boilerplate with tensorboard logging and evaluation helper.

Usage:
    pip install pybullet gymnasium stable-baselines3 torch torchvision
    python main.py --train
    python main.py --evaluate
"""

from __future__ import annotations

import argparse
import math
import os
import random
from collections import deque

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

_STABLE_OBS_KEY = "pixels_proprio"


class RobotArmEnv(gym.Env):
    """PyBullet-based arm reaching environment with camera-only observations."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode: str | None = None,
        img_height: int = 64,
        img_width: int = 64,
        grayscale: bool = True,
        frame_stack: int = 4,
        frame_skip: int = 4,
        max_steps: int = 300,
        workspace_radius: float = 0.25,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.img_height = img_height
        self.img_width = img_width
        self.grayscale = grayscale
        self.frame_stack = frame_stack
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.workspace_radius = workspace_radius
        self.renderer = p.ER_BULLET_HARDWARE_OPENGL if render_mode == "human" else p.ER_TINY_RENDERER
        self.physics_client = p.connect(p.GUI if render_mode == "human" else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setPhysicsEngineParameter(fixedTimeStep=1 / 240.0, numSolverIterations=100)
        self._load_scene()
        channel_count = 1 if grayscale else 3
        self.pixel_channels = channel_count
        self.proprio_dim = len(self.control_joints)
        self.pixel_dim = frame_stack * channel_count * img_height * img_width
        self.observation_space = spaces.Dict(
            {
                "pixels": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(frame_stack, channel_count, img_height, img_width),
                    dtype=np.float32,
                ),
                "proprio": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(self.proprio_dim,),
                    dtype=np.float32,
                ),
            }
        )
        self.action_space = spaces.Box(low=-0.05, high=0.05, shape=(3,), dtype=np.float32)
        self.frame_buffer: deque[np.ndarray] = deque(maxlen=frame_stack)
        self.target_id: int | None = None
        self.step_counter = 0
        self.reward_weights = dict(distance=1.0, contact=50.0, time_penalty=0.1)
        self.success_threshold = 0.05
        self.success_bonus = 100.0
        self.reset()

    def _load_scene(self) -> None:
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        self.plane = p.loadURDF("plane.urdf")
        p.loadURDF("table/table.urdf", [0.5, 0, -0.63])
        self.robot_id = p.loadURDF(
            "kuka_iiwa/model.urdf", useFixedBase=True, basePosition=[0, 0, 0]
        )
        self.control_joints = [i for i in range(p.getNumJoints(self.robot_id)) if i < 7]
        self.tool_link = 6
        self.workspace_center = np.array([0.6, 0.0, 0.3])
        self.target_visual_shape = p.createVisualShape(
            p.GEOM_SPHERE, radius=0.03, rgbaColor=[1, 0, 0, 1]
        )
        self.target_collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.03)

    def _spawn_target(self) -> None:
        if self.target_id is not None:
            p.removeBody(self.target_id)
        angle = random.uniform(0, 2 * math.pi)
        radius = random.uniform(0.05, self.workspace_radius)
        target_xy = self.workspace_center[:2] + radius * np.array([math.cos(angle), math.sin(angle)])
        position = [target_xy[0], target_xy[1], 0.05]
        self.target_pos = np.array(position, dtype=np.float32)
        self.target_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=self.target_collision_shape,
            baseVisualShapeIndex=self.target_visual_shape,
            basePosition=position,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_counter = 0
        self.frame_buffer.clear()
        p.resetBasePositionAndOrientation(self.robot_id, [0, 0, 0], [0, 0, 0, 1])
        for joint in self.control_joints:
            p.resetJointState(self.robot_id, joint, targetValue=0.0)
        self._spawn_target()
        p.stepSimulation()
        frame = self._render_camera()
        for _ in range(self.frame_stack):
            self.frame_buffer.append(frame)
        observation = self._get_observation()
        return observation, {}

    def step(self, action: np.ndarray):
        clipped = np.clip(action, self.action_space.low, self.action_space.high)
        total_reward = 0.0
        done = False
        info = {}
        for _ in range(self.frame_skip):
            self._apply_action(clipped)
            p.stepSimulation()
            reward, success = self._compute_reward()
            total_reward += reward
            if success or self._is_contact():
                done = True
                info["is_success"] = success
                break
        self.step_counter += 1
        if self.step_counter >= self.max_steps:
            done = True
        frame = self._render_camera()
        self.frame_buffer.append(frame)
        observation = self._get_observation()
        return observation, total_reward, done, False, info

    def _apply_action(self, delta: np.ndarray) -> None:
        link_state = p.getLinkState(self.robot_id, self.tool_link)
        current_pos = np.array(link_state[0])
        goal_pos = current_pos + delta
        goal_pos = np.clip(
            goal_pos,
            self.workspace_center - self.workspace_radius,
            self.workspace_center + self.workspace_radius,
        )
        joint_positions = p.calculateInverseKinematics(
            self.robot_id,
            self.tool_link,
            goal_pos.tolist(),
            residualThreshold=1e-5,
            maxNumIterations=20,
        )
        target_positions = [joint_positions[i] for i in self.control_joints]
        p.setJointMotorControlArray(
            self.robot_id,
            self.control_joints,
            p.POSITION_CONTROL,
            targetPositions=target_positions,
            forces=[200] * len(self.control_joints),
        )

    def _render_camera(self) -> np.ndarray:
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=[0.6, 0, 1.0],
            cameraTargetPosition=[0.6, 0, 0.0],
            cameraUpVector=[0, 1, 0],
        )
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=float(self.img_width) / self.img_height, nearVal=0.1, farVal=2.5
        )
        img = p.getCameraImage(
            width=self.img_width,
            height=self.img_height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=self.renderer,
        )
        rgb = np.array(img[2], dtype=np.uint8).reshape(
            self.img_height, self.img_width, 4
        )[..., :3]
        if self.grayscale:
            rgb = np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])
            img_tensor = rgb.astype(np.float32) / 255.0
            img_tensor = np.expand_dims(img_tensor, 0)
        else:
            img_tensor = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)
        return img_tensor.astype(np.float32)

    def _get_proprioception(self) -> np.ndarray:
        joints = [p.getJointState(self.robot_id, j)[0] for j in self.control_joints]
        arr = np.array(joints, dtype=np.float32)
        arr = np.clip(arr / math.pi, -1.0, 1.0)
        return arr

    def _get_observation(self) -> np.ndarray:
        pixels = np.zeros(
            (self.frame_stack, self.pixel_channels, self.img_height, self.img_width),
            dtype=np.float32,
        )
        for idx, frame in enumerate(self.frame_buffer):
            frame_arr = np.asarray(frame, dtype=np.float32)
            if frame_arr.ndim == 2:
                frame_arr = frame_arr[np.newaxis]
            if frame_arr.shape != (self.pixel_channels, self.img_height, self.img_width):
                frame_arr = frame_arr.reshape(
                    self.pixel_channels,
                    self.img_height,
                    self.img_width,
                )
            pixels[idx] = frame_arr
        proprio = self._get_proprioception()
        return {"pixels": pixels, "proprio": proprio}

    def _compute_reward(self) -> tuple[float, bool]:
        tool_pos = np.array(p.getLinkState(self.robot_id, self.tool_link)[0])
        distance = np.linalg.norm(tool_pos - self.target_pos)
        reward = -self.reward_weights["distance"] * distance
        reward -= self.reward_weights["time_penalty"]
        success = distance <= self.success_threshold
        if self._is_contact():
            reward += self.reward_weights["contact"]
        if success:
            reward += self.success_bonus
        return float(reward), success

    def _is_contact(self) -> bool:
        if self.target_id is None:
            return False
        contacts = p.getContactPoints(self.target_id, self.robot_id)
        return len(contacts) > 0

    def render(self) -> np.ndarray | None:
        if self.render_mode == "rgb_array":
            return self._render_camera()
        return None

    def close(self) -> None:
        if p.isConnected():
            p.disconnect()


class CustomCNN(BaseFeaturesExtractor):
    """Extractor that fuses stacked camera frames and joint angles."""

    def __init__(self, observation_space: spaces.Space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        pixel_shape = observation_space.spaces["pixels"].shape
        proprio_shape = observation_space.spaces["proprio"].shape
        self.frame_stack, self.channels, self.height, self.width = pixel_shape
        proprio_dim = proprio_shape[0]
        stacked_channels = self.frame_stack * self.channels
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(stacked_channels, 32, kernel_size=3, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=2),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
        )
        self.proprio_net = torch.nn.Sequential(
            torch.nn.Linear(proprio_dim, 64),
            torch.nn.ReLU(),
        )
        with torch.no_grad():
            sample_pixels = torch.zeros(1, stacked_channels, self.height, self.width)
            flattened_size = self.cnn(sample_pixels).shape[1]
        concat_size = flattened_size + 64
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(concat_size, features_dim),
            torch.nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        pixels = observations["pixels"]
        batch_size = pixels.shape[0]
        pixels = pixels.reshape(batch_size, -1, self.height, self.width)
        features = self.cnn(pixels)
        proprio = observations["proprio"]
        proprio_features = self.proprio_net(proprio)
        concat = torch.cat([features, proprio_features], dim=1)
        return self.linear(concat)


def make_env(render_mode: str | None = None) -> gym.Env:
    env = RobotArmEnv(
        render_mode=render_mode,
        img_height=64,
        img_width=64,
        grayscale=True,
        frame_stack=4,
        frame_skip=4,
        max_steps=200,
    )
    return Monitor(env)


def train_model(total_timesteps: int = 150_000) -> None:
    env = DummyVecEnv([lambda: make_env()])
    env = VecMonitor(env)
    tensorboard_folder = os.path.join("runs", "robot_arm")
    policy_kwargs = {
        "features_extractor_class": CustomCNN,
        "features_extractor_kwargs": {"features_dim": 256},
    }
    model = PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=tensorboard_folder,
    )
    checkpoint_cb = CheckpointCallback(save_freq=5000, save_path="checkpoints", name_prefix="ppo_robot")
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_cb)
    model.save("ppo_robot_final")


def evaluate_model(episodes: int = 5) -> None:
    env = make_env(render_mode="human")
    model = PPO.load("ppo_robot_final")
    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
        print(f"Episode {episode + 1} reward: {total_reward:.2f}")
    env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the PPO agent")
    parser.add_argument("--evaluate", action="store_true", help="Run the trained agent")
    parser.add_argument("--timesteps", type=int, default=150_000, help="Number of train timesteps")
    args = parser.parse_args()
    if args.train:
        train_model(total_timesteps=args.timesteps)
    if args.evaluate:
        evaluate_model()
    if not (args.train or args.evaluate):
        parser.print_help()


if __name__ == "__main__":
    main()