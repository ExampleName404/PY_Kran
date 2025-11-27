import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
import cv2
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
import os


class RobotArmEnv(gym.Env):
    """Среда для обучения робота управлению по изображению"""
    
    def __init__(self, render_mode=None, use_gui=False, use_stereo=True):
        super().__init__()
        
        # Параметры камеры
        self.img_width = 64
        self.img_height = 64
        self.use_grayscale = True  # Для экономии ресурсов
        self.use_stereo = use_stereo  # Использовать две камеры для стереозрения
        
        # Параметры симуляции
        self.frame_skip = 4  # Повторять действие N раз
        self.max_steps = 100
        self.current_step = 0
        
        # Action space: смещение схвата (dx, dy, dz)
        self.action_space = spaces.Box(
            low=-0.05, 
            high=0.05, 
            shape=(3,), 
            dtype=np.float32
        )
        
        # Observation space: изображение + углы джоинтов
        n_channels = 1 if self.use_grayscale else 3
        # Если стереозрение - удваиваем количество каналов
        if self.use_stereo:
            n_channels *= 2
        
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0, high=255,
                shape=(self.img_height, self.img_width, n_channels),
                dtype=np.uint8
            ),
            'joints': spaces.Box(
                low=-np.pi, high=np.pi,
                shape=(7,),
                dtype=np.float32
            )
        })
        
        # Инициализация PyBullet
        self.use_gui = use_gui
        if self.use_gui:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Загрузка сцены
        self.plane_id = None
        self.robot_id = None
        self.object_id = None
        self.target_pos = None
        
        self._setup_scene()
    
    def _setup_scene(self):
        """Настройка сцены: стол, робот, объект"""
        # Плоскость (стол) - делаем темной для контраста с ярким объектом
        self.plane_id = p.loadURDF("plane.urdf")
        # Изменяем цвет плоскости на темный
        p.changeVisualShape(self.plane_id, -1, rgbaColor=[0.2, 0.2, 0.2, 1])
        
        # Робот Kuka IIWA
        self.robot_id = p.loadURDF(
            "kuka_iiwa/model.urdf",
            basePosition=[0, 0, 0],
            useFixedBase=True
        )
        
        # Настройка робота
        self.num_joints = p.getNumJoints(self.robot_id)
        self.ee_index = 6  # End-effector link index
        
        # Начальная конфигурация робота
        self.initial_joint_positions = [0, 0.5, 0, -1.5, 0, 1.0, 0]
        for i in range(len(self.initial_joint_positions)):
            p.resetJointState(self.robot_id, i, self.initial_joint_positions[i])
        
        # Целевой объект (куб)
        self.object_id = None  # Создается в reset()
    
    def _get_camera_image(self):
        """Получение изображения с виртуальной камеры (Eye-to-hand)"""
        if self.use_stereo:
            # Две камеры для стереозрения (левая и правая)
            images = []
            camera_positions = [
                [0.45, -0.1, 1.0],  # Левая камера
                [0.45, 0.1, 1.0]    # Правая камера
            ]
            
            for cam_pos in camera_positions:
                view_matrix = p.computeViewMatrix(
                    cameraEyePosition=cam_pos,
                    cameraTargetPosition=[0.5, 0, 0.3],
                    cameraUpVector=[0, 0, 1]
                )
                
                proj_matrix = p.computeProjectionMatrixFOV(
                    fov=60,
                    aspect=1.0,
                    nearVal=0.1,
                    farVal=3.0
                )
                
                img = p.getCameraImage(
                    self.img_width,
                    self.img_height,
                    view_matrix,
                    proj_matrix,
                    renderer=p.ER_TINY_RENDERER
                )
                
                # Обработка изображения
                rgb_array = np.array(img[2], dtype=np.uint8)
                rgb_array = rgb_array.reshape((self.img_height, self.img_width, 4))[:, :, :3]
                
                if self.use_grayscale:
                    gray = np.dot(rgb_array, [0.299, 0.587, 0.114])
                    images.append(gray.astype(np.uint8))
                else:
                    images.append(rgb_array.astype(np.uint8))
            
            # Объединение двух изображений по каналам
            if self.use_grayscale:
                return np.stack(images, axis=-1)  # (H, W, 2)
            else:
                return np.concatenate(images, axis=-1)  # (H, W, 6)
        else:
            # Одна камера (оригинальный вариант)
            view_matrix = p.computeViewMatrix(
                cameraEyePosition=[0.5, 0, 1.0],
                cameraTargetPosition=[0.5, 0, 0.3],
                cameraUpVector=[0, 0, 1]
            )
            
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=60,
                aspect=1.0,
                nearVal=0.1,
                farVal=3.0
            )
            
            img = p.getCameraImage(
                self.img_width,
                self.img_height,
                view_matrix,
                proj_matrix,
                renderer=p.ER_TINY_RENDERER
            )
            
            rgb_array = np.array(img[2], dtype=np.uint8)
            rgb_array = rgb_array.reshape((self.img_height, self.img_width, 4))[:, :, :3]
            
            if self.use_grayscale:
                gray = np.dot(rgb_array, [0.299, 0.587, 0.114])
                return gray.astype(np.uint8).reshape(self.img_height, self.img_width, 1)
            else:
                return rgb_array.astype(np.uint8)
    
    def _get_joint_states(self):
        """Получение углов джоинтов робота (проприоцепция)"""
        joint_states = []
        for i in range(7):
            state = p.getJointState(self.robot_id, i)
            joint_states.append(state[0])  # Угол
        return np.array(joint_states, dtype=np.float32)
    
    def _compute_reward(self):
        """Вычисление награды"""
        # Текущая позиция схвата
        ee_state = p.getLinkState(self.robot_id, self.ee_index)
        ee_pos = np.array(ee_state[0])
        
        # Расстояние до цели
        distance = np.linalg.norm(ee_pos - self.target_pos)
        
        # Dense reward: штраф за расстояние
        reward = -2.0 * distance
        
        # Sparse reward: бонус за касание
        contacts = p.getContactPoints(self.robot_id, self.object_id)
        if len(contacts) > 0:
            reward += 10.0
        
        # Штраф за каждый шаг (стимул действовать быстро)
        reward -= 0.01
        
        # Терминация при успехе
        done = distance < 0.05
        if done:
            reward += 50.0  # Большой бонус за успех
        
        # Терминация при превышении лимита шагов
        truncated = self.current_step >= self.max_steps
        
        return reward, done, truncated
    
    def reset(self, seed=None, options=None):
        """Сброс среды"""
        super().reset(seed=seed)
        
        self.current_step = 0
        
        # Удаление старого объекта
        if self.object_id is not None:
            p.removeBody(self.object_id)
        
        # Сброс робота в начальную позицию
        for i in range(len(self.initial_joint_positions)):
            p.resetJointState(self.robot_id, i, self.initial_joint_positions[i])
        
        # Случайная позиция объекта в рабочей зоне
        x = np.random.uniform(0.3, 0.7)
        y = np.random.uniform(-0.3, 0.3)
        z = 0.5
        self.target_pos = np.array([x, y, z])
        
        # Создание нового объекта (куб) - ЯРКИЙ КОНТРАСТНЫЙ цвет для лучшей видимости
        collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.03, 0.03, 0.03])
        visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.03, 0.03, 0.03],
            rgbaColor=[1, 1, 0, 1]  # ЖЕЛТЫЙ цвет - максимально контрастен на темном фоне
        )
        self.object_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=self.target_pos
        )
        
        # Стабилизация сцены
        for _ in range(10):
            p.stepSimulation()
        
        # Получение наблюдения
        observation = {
            'image': self._get_camera_image(),
            'joints': self._get_joint_states()
        }
        
        return observation, {}
    
    def step(self, action):
        """Выполнение действия"""
        self.current_step += 1
        
        # Применение действия: смещение end-effector
        ee_state = p.getLinkState(self.robot_id, self.ee_index)
        current_pos = np.array(ee_state[0])
        target_pos = current_pos + action
        
        # Inverse kinematics для расчета углов джоинтов
        joint_poses = p.calculateInverseKinematics(
            self.robot_id,
            self.ee_index,
            target_pos
        )
        
        # Применение углов к джоинтам
        for i in range(7):
            p.setJointMotorControl2(
                self.robot_id,
                i,
                p.POSITION_CONTROL,
                targetPosition=joint_poses[i],
                force=200
            )
        
        # Frame skipping: повторить действие N раз
        for _ in range(self.frame_skip):
            p.stepSimulation()
        
        # Получение наблюдения
        observation = {
            'image': self._get_camera_image(),
            'joints': self._get_joint_states()
        }
        
        # Вычисление награды
        reward, done, truncated = self._compute_reward()
        
        return observation, reward, done, truncated, {}
    
    def close(self):
        """Закрытие среды"""
        p.disconnect()


class NatureCNN(BaseFeaturesExtractor):
    """Легкая сверточная сеть для обработки изображений (Вариант А)"""
    
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        # Извлечение размерности изображения
        # VecTransposeImage может изменить форму на (C, H, W) вместо (H, W, C)
        img_shape = observation_space['image'].shape
        if img_shape[0] in [1, 2, 3, 4, 6] and img_shape[1] == 64 and img_shape[2] == 64:
            # Формат (C, H, W) после VecTransposeImage
            n_input_channels = img_shape[0]
        else:
            # Формат (H, W, C) - оригинальный
            n_input_channels = img_shape[2]
        
        # Легкая CNN архитектура с малыми ядрами (3x3)
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=2, padding=1),  # 64x64 -> 32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # 16x16 -> 16x16
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Вычисление размера выхода CNN
        with torch.no_grad():
            sample_input = torch.zeros(1, n_input_channels, 64, 64)
            n_flatten = self.cnn(sample_input).shape[1]
        
        # Размерность для углов джоинтов - используем фактический размер из observation_space
        # VecFrameStack может стекать joints, поэтому берем фактическую размерность
        n_joints = observation_space['joints'].shape[0]
        
        # Полносвязные слои
        self.linear = nn.Sequential(
            nn.Linear(n_flatten + n_joints, features_dim),
            nn.ReLU(),
        )
        
        self.n_joints = n_joints
    
    def forward(self, observations):
        # Обработка изображения
        # Нормализация и транспонирование из (B, H, W, C) в (B, C, H, W)
        image = observations['image'].float() / 255.0
        
        # Проверка и корректировка формата изображения
        if image.shape[1] == 2 and image.shape[2] == 64 and image.shape[3] == 64:
            # Уже в формате (B, C, H, W) - VecTransposeImage сработал
            pass
        elif image.shape[1] == 64 and image.shape[2] == 64 and image.shape[3] == 2:
            # Формат (B, H, W, C) - нужно транспонировать
            image = image.permute(0, 3, 1, 2)
        else:
            # Неожиданный формат - пробуем стандартное транспонирование
            if len(image.shape) == 4 and image.shape[-1] in [1, 2, 3, 4, 6]:
                image = image.permute(0, 3, 1, 2)
        
        # Пропуск через CNN
        cnn_features = self.cnn(image)
        
        # Объединение с углами джоинтов
        joints = observations['joints']
        combined = torch.cat([cnn_features, joints], dim=1)
        
        return self.linear(combined)


class TensorboardCallback(BaseCallback):
    """Callback для логирования в TensorBoard"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_reward = 0
        self.current_length = 0
    
    def _on_step(self):
        self.current_reward += self.locals['rewards'][0]
        self.current_length += 1
        
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_reward)
            self.episode_lengths.append(self.current_length)
            
            self.logger.record('rollout/ep_rew_mean', np.mean(self.episode_rewards[-100:]))
            self.logger.record('rollout/ep_len_mean', np.mean(self.episode_lengths[-100:]))
            
            self.current_reward = 0
            self.current_length = 0
        
        return True


def train_robot():
    """Основная функция обучения"""
    print("🤖 Запуск обучения Visual RL...")
    print("=" * 60)
    
    # Создание среды
    print("📦 Создание среды...")
    print("👁️👁️  Используется СТЕРЕОЗРЕНИЕ (две камеры) для лучшего распознавания глубины")
    env = RobotArmEnv(use_gui=False, use_stereo=True)  # use_stereo=True для двух камер
    env = DummyVecEnv([lambda: env])
    
    # Примечание: VecFrameStack не работает корректно с Dict observation space
    # Стереозрение уже дает достаточно информации о глубине
    print("ℹ️  Frame stacking отключен (несовместим с Dict obs space)")
    
    # Проверка доступности GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Используется устройство: {device.upper()}")
    if device == "cpu":
        print("⚠️  GPU не найден. Для использования GPU установите: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    
    # Настройка модели PPO с кастомной CNN
    print("🧠 Инициализация модели PPO с NatureCNN...")
    policy_kwargs = dict(
        features_extractor_class=NatureCNN,
        features_extractor_kwargs=dict(features_dim=256),
        normalize_images=False,  # Отключаем VecTransposeImage для Dict observation space
    )
    
    model = PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log="./logs/visual_rl/",
        device=device,  # Использовать GPU если доступен
    )
    
    print("\n✅ Модель готова к обучению!")
    print(f"📊 Observation space: {env.observation_space}")
    print(f"🎮 Action space: {env.action_space}")
    print("\n" + "=" * 60)
    print("🚀 Начинаем обучение...")
    print("=" * 60)
    print("\n💡 Советы:")
    print("  - Откройте TensorBoard: tensorboard --logdir ./logs/visual_rl/")
    print("  - Обучение займет 30-60 минут на CPU")
    print("  - Для визуализации: установите use_gui=True в RobotArmEnv")
    print("\n" + "=" * 60 + "\n")
    
    # Обучение
    callback = TensorboardCallback()
    model.learn(
        total_timesteps=500_000,
        callback=callback,
        progress_bar=True
    )
    
    # Сохранение модели
    print("\n💾 Сохранение модели...")
    os.makedirs("./models", exist_ok=True)
    model.save("./models/visual_rl_robot")
    
    print("\n✅ Обучение завершено!")
    print(f"📁 Модель сохранена в: ./models/visual_rl_robot.zip")
    print(f"📊 Логи TensorBoard: ./logs/visual_rl/")
    
    env.close()


def test_robot():
    """Тестирование обученной модели"""
    print("🧪 Запуск тестирования модели...")
    
    # Создание среды с визуализацией
    env = RobotArmEnv(use_gui=True, use_stereo=True)
    env = DummyVecEnv([lambda: env])
    
    # Загрузка модели
    model = PPO.load("./models/visual_rl_robot", env=env)
    
    # Тестирование на 10 эпизодах
    for episode in range(10):
        obs = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        print(f"\n📍 Эпизод {episode + 1}/10")
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            steps += 1
        
        print(f"   Награда: {episode_reward:.2f}, Шагов: {steps}")
    
    env.close()
    print("\n✅ Тестирование завершено!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Режим тестирования
        test_robot()
    else:
        # Режим обучения
        train_robot()