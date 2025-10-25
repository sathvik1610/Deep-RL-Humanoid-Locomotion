import gymnasium as gym
import pybullet as p
import pybullet_data
import numpy as np
import math


class HumanoidIKEnv(gym.Env):
    """
    Custom PyBullet environment using the standard humanoid.urdf,
    compatible with IK-derived initial poses.
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}

    def __init__(self, render_mode='human'):
        super().__init__()
        print("--- [Module 2 IK] Initializing HumanoidIKEnv ---")

        self._render_mode = render_mode
        self._physics_client_id = -1
        self.urdf_root = pybullet_data.getDataPath()

        # Connect temporarily to get model info
        temp_client = p.connect(p.DIRECT)
        try:
            p.setAdditionalSearchPath(self.urdf_root, physicsClientId=temp_client)
            urdf_path_local = "humanoid.urdf"
            self.robot_id_temp = p.loadURDF(urdf_path_local, [0, 0, 1], useFixedBase=False, physicsClientId=temp_client)
            self._get_humanoid_info(temp_client, self.robot_id_temp)
            print(
                f"  [Env Init] Standard humanoid.urdf loaded. Found {len(self.controllable_joint_indices)} controllable joints.")
        except Exception as e:
            print(f"❌ FATAL ERROR loading humanoid.urdf in Env Init: {e}")
            raise e
        finally:
            p.disconnect(temp_client)

        num_actions = len(self.controllable_joint_indices)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(num_actions,), dtype=np.float32)
        print(f"  [Env Init] Action Space defined: Box{self.action_space.shape}")

        obs_dim = num_actions * 2 + 3 + 4 + 3 + 3
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        print(f"  [Env Init] Observation Space defined: Box{self.observation_space.shape}")

        self.time_step = 1.0 / 240.0
        self.max_episode_steps = 1000
        self.current_step = 0
        self.fall_threshold_height = 0.6
        self.robot_id = -1

    def _get_humanoid_info(self, client_id, robot_id):
        self.num_total_joints = p.getNumJoints(robot_id, physicsClientId=client_id)
        self.controllable_joint_indices = []
        self.joint_names = []
        self.base_link_index = -1

        for i in range(self.num_total_joints):
            info = p.getJointInfo(robot_id, i, physicsClientId=client_id)
            joint_name = info[1].decode('UTF-8')
            joint_type = info[2]

            if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC, p.JOINT_SPHERICAL]:
                self.controllable_joint_indices.append(i)
                self.joint_names.append(joint_name)

        for i in range(self.num_total_joints):
            info = p.getJointInfo(robot_id, i, physicsClientId=client_id)
            link_name = info[12].decode('UTF-8')
            if link_name == 'chest':
                self.torso_link_index = i
                print(f"   - Found torso link ('chest') at index: {i}")
                break
        else:
            print("   - WARNING: Could not find 'chest' link index. Fall detection might be inaccurate.")
            self.torso_link_index = -1

    def reset(self, seed=None, options=None, initial_pose=None):
        super().reset(seed=seed)
        self.current_step = 0

        if self._physics_client_id >= 0: p.disconnect(self._physics_client_id)
        if self._render_mode == 'human':
            self._physics_client_id = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

            # --- START OF MODIFICATION 1 (CAMERA) ---
            # With new rotation, robot's "front" is World +Y. Look from -Y.
            p.resetDebugVisualizerCamera(
                cameraDistance=10,
                cameraYaw=90,  # Look from the front
                cameraPitch=-15,
                cameraTargetPosition=[0, 0, 2.0] # Adjust target to be lower
            )
            # --- END OF MODIFICATION 1 ---
        else:
            self._physics_client_id = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(self.urdf_root, physicsClientId=self._physics_client_id)
        p.setGravity(0, 0, -9.81, physicsClientId=self._physics_client_id)
        p.setTimeStep(self.time_step, physicsClientId=self._physics_client_id)
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self._physics_client_id)

        # Load Robot
        # --- START OF MODIFICATION 2 (ROTATION) ---
        start_pos = [0, 0, 6.0]  # Your height is correct
        # Add Z-axis rotation to align X/Y axes correctly
        start_orn = p.getQuaternionFromEuler([math.pi / 2, 0, 0])  # Simple 90° rotation around X

        # --- END OF MODIFICATION 2 ---

        urdf_path_local = "humanoid.urdf"
        self.robot_id = p.loadURDF(urdf_path_local, start_pos, start_orn, useFixedBase=False,
                                   physicsClientId=self._physics_client_id)
        self._get_humanoid_info(self._physics_client_id, self.robot_id)

        if initial_pose is not None:
            print("  [Env Reset] Applying initial pose from IK...")
            if len(initial_pose) != len(self.controllable_joint_indices):
                print(
                    f"⚠️ WARNING: initial_pose length ({len(initial_pose)}) doesn't match controllable joints ({len(self.controllable_joint_indices)}). Using default pose.")
            else:
                num_joints_to_set = len(self.controllable_joint_indices)
                target_poses = initial_pose[:num_joints_to_set]
                target_indices = self.controllable_joint_indices[:num_joints_to_set]

                for i, joint_index in enumerate(target_indices):
                    if 0 <= joint_index < self.num_total_joints:
                        p.resetJointState(
                            bodyUniqueId=self.robot_id,
                            jointIndex=joint_index,
                            targetValue=target_poses[i],
                            targetVelocity=0.0,
                            physicsClientId=self._physics_client_id
                        )
                    else:
                        print(f"⚠️ WARNING: Invalid joint index ({joint_index})...")

                for _ in range(30): p.stepSimulation(physicsClientId=self._physics_client_id)
        else:
            print("  [Env Reset] No initial_pose provided. Using default URDF pose.")

        observation = self._get_observation()
        info = {}
        return observation, info

    def step(self, action):
        max_force = 100
        scaled_action = action * max_force

        for i, joint_index in enumerate(self.controllable_joint_indices):
            p.setJointMotorControl2(
                bodyIndex=self.robot_id,
                jointIndex=joint_index,
                controlMode=p.TORQUE_CONTROL,
                force=scaled_action[i],
                physicsClientId=self._physics_client_id
            )

        p.stepSimulation(physicsClientId=self._physics_client_id)
        self.current_step += 1

        observation = self._get_observation()
        reward = self._compute_reward()
        terminated = self._is_terminated()
        truncated = self.current_step >= self.max_episode_steps
        info = {}

        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        joint_angles = []
        joint_velocities = []
        for i in self.controllable_joint_indices:
            state = p.getJointState(self.robot_id, i, physicsClientId=self._physics_client_id)
            joint_angles.append(state[0])
            joint_velocities.append(state[1])

        base_pos, base_orn_quat = p.getBasePositionAndOrientation(self.robot_id,
                                                                  physicsClientId=self._physics_client_id)
        base_lin_vel, base_ang_vel = p.getBaseVelocity(self.robot_id, physicsClientId=self._physics_client_id)

        observation = np.concatenate([
            joint_angles,
            joint_velocities,
            base_pos,
            base_orn_quat,
            base_lin_vel,
            base_ang_vel
        ]).astype(np.float32)

        return observation

    def _compute_reward(self, w_vel=1.0, w_live=0.1, w_energy=0.0001, fall_penalty=-5.0):
        base_pos, _ = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self._physics_client_id)
        base_lin_vel, _ = p.getBaseVelocity(self.robot_id, physicsClientId=self._physics_client_id)

        # With rotation [pi/2, 0, 0], forward is still +Y direction
        forward_velocity = base_lin_vel[1]
        r_vel = w_vel * forward_velocity

        is_alive = base_pos[2] > self.fall_threshold_height
        r_live = w_live if is_alive else fall_penalty

        applied_torques = []
        for i in self.controllable_joint_indices:
            joint_state = p.getJointState(self.robot_id, i, physicsClientId=self._physics_client_id)
            applied_torques.append(joint_state[3])
        r_energy = w_energy * np.sum(np.square(applied_torques))

        reward = r_vel + r_live - r_energy
        return reward

    def _is_terminated(self):
        base_pos, _ = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self._physics_client_id)
        return base_pos[2] < self.fall_threshold_height

    def render(self):
        if self._render_mode == 'rgb_array':
            base_pos, _ = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self._physics_client_id)
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=base_pos, distance=2.0, yaw=45, pitch=-30, roll=0, upAxisIndex=2,
                physicsClientId=self._physics_client_id)
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=60, aspect=1.0, nearVal=0.1, farVal=100.0,
                physicsClientId=self._physics_client_id)
            img_arr = p.getCameraImage(
                width=224, height=224, viewMatrix=view_matrix, projectionMatrix=proj_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL, physicsClientId=self._physics_client_id)
            w, h, rgb, _, _ = img_arr
            np_img_arr = np.array(rgb).reshape((h, w, 4))[:, :, :3]
            return np_img_arr
        elif self._render_mode == 'human':
            pass

    def close(self):
        if self._physics_client_id >= 0:
            p.disconnect(self._physics_client_id)
            self._physics_client_id = -1
        print("--- [Module 2 IK] Environment Closed ---")