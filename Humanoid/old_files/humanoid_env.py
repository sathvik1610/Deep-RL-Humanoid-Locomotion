# humanoid_env.py

import gymnasium as gym
import pybullet as p
import pybullet_data
import numpy as np
import math
import time # For time.sleep if needed

class HumanoidWalkEnv(gym.Env):
    """
    Custom PyBullet environment for the humanoid walking task,
    compliant with the Gymnasium API.
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 50}

    def __init__(self, urdf_path="biped.urdf", render_mode='human'):
        """
        Initializes the environment.

        Args:
            urdf_path (str): Path to the URDF file for the humanoid.
            render_mode (str): 'human' for GUI, 'None' for headless (DIRECT mode).
        """
        super().__init__()
        print("--- [Module 2] Initializing HumanoidWalkEnv ---")

        self.urdf_path = urdf_path
        self._render_mode = render_mode
        self._physics_client_id = -1 # Will be set in reset() or render()

        # Connect to PyBullet (will connect properly in reset/render)
        # This initial connection is just to load URDF info
        temp_client = p.connect(p.DIRECT)
        try:
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            plane_id = p.loadURDF("plane.urdf", physicsClientId=temp_client) # Load plane to prevent error
            self.robot_id = p.loadURDF(self.urdf_path, [0, 0, 0.8], useFixedBase=False, physicsClientId=temp_client)
            self._get_joint_info(temp_client) # Get joint info using the temp client
            print(f"  [Env Init] URDF loaded successfully. Found {len(self.controllable_joint_indices)} controllable joints.")
        except Exception as e:
            print(f"❌ FATAL ERROR loading URDF in Env Init: {e}")
            raise e
        finally:
            p.disconnect(temp_client)

        # --- Define Action Space (Continuous Torques) ---
        # Action is a vector of torques for each controllable joint.
        # Let's assume torques are between -1.0 and 1.0 (can be scaled later)
        action_low = -1.0 * np.ones(len(self.controllable_joint_indices))
        action_high = 1.0 * np.ones(len(self.controllable_joint_indices))
        self.action_space = gym.spaces.Box(low=action_low, high=action_high, dtype=np.float32)
        print(f"  [Env Init] Action Space defined: Box{self.action_space.shape}")

        # --- Define Observation Space ---
        # Includes: joint angles, joint velocities, torso orientation (quat), torso lin/ang velocity
        obs_dim = (
            len(self.controllable_joint_indices) +  # Joint angles
            len(self.controllable_joint_indices) +  # Joint velocities
            4 +  # Torso orientation (quaternion x,y,z,w)
            3 +  # Torso linear velocity (x,y,z)
            3    # Torso angular velocity (wx,wy,wz)
        )
        # Define reasonable bounds for observations
        obs_low = -np.inf * np.ones(obs_dim)
        obs_high = np.inf * np.ones(obs_dim)
        self.observation_space = gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        print(f"  [Env Init] Observation Space defined: Box{self.observation_space.shape}")

        # --- Simulation Parameters ---
        self.time_step = 1.0 / 240.0 # PyBullet default simulation step
        self.max_episode_steps = 1000 # Max steps before truncation
        self.current_step = 0
        self.fall_threshold_pitch = 1.0 # Radians (~57 degrees) - adjust as needed
        self.fall_threshold_height = 0.5 # Minimum height of torso base

        print("--- [Module 2] Environment Initialized ---")


    def _get_joint_info(self, client_id):
        """Helper to get joint indices, names, and limits."""
        self.controllable_joint_indices = []
        self.joint_names = []
        self.joint_lower_limits = []
        self.joint_upper_limits = []
        num_joints = p.getNumJoints(self.robot_id, physicsClientId=client_id)

        for i in range(num_joints):
            info = p.getJointInfo(self.robot_id, i, physicsClientId=client_id)
            joint_name = info[1].decode('utf-8')
            joint_type = info[2]
            if joint_type == p.JOINT_REVOLUTE:
                if joint_name == 'j_head': # Skip head
                    continue
                self.controllable_joint_indices.append(i)
                self.joint_names.append(joint_name)
                self.joint_lower_limits.append(info[8])
                self.joint_upper_limits.append(info[9])

    def reset(self, seed=None, options=None, initial_pose=None):
        """
        Resets the environment to start a new episode.

        Args:
            seed (int, optional): The seed for the random number generator.
            options (dict, optional): Additional options for resetting.
            initial_pose (np.ndarray, optional): The initial joint angles (theta_init)
                                                 from Module 1. If None, uses default pose.

        Returns:
            tuple: A tuple containing the initial observation and an empty info dict.
        """
        super().reset(seed=seed)
        self.current_step = 0

        # --- Connect to PyBullet ---
        # Disconnect if already connected
        if self._physics_client_id >= 0:
            p.disconnect(self._physics_client_id)

        if self._render_mode == 'human':
            self._physics_client_id = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0) # Disable default GUI controls
            # You can set camera distance, pitch, yaw here if needed
            p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=30, cameraPitch=-20, cameraTargetPosition=[0,0,0.5])

        else: # Headless mode
            self._physics_client_id = p.connect(p.DIRECT)

        # --- Setup Scene ---
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self._physics_client_id)
        p.setGravity(0, 0, -9.81, physicsClientId=self._physics_client_id)
        p.setTimeStep(self.time_step, physicsClientId=self._physics_client_id)
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self._physics_client_id)

        # --- Load Robot and Set Initial Pose ---
        start_pos = [0, 0, 0.8] # Default start height
        start_orn = p.getQuaternionFromEuler([0, 0, 0])
        self.robot_id = p.loadURDF(self.urdf_path, start_pos, start_orn, useFixedBase=False, physicsClientId=self._physics_client_id)
        self._get_joint_info(self._physics_client_id) # Refresh joint info for the new client ID

        # --- Apply initial_pose from Module 1 ---
        if initial_pose is not None:
            print("  [Env Reset] Applying initial pose from Module 1...")
            if len(initial_pose) != len(self.controllable_joint_indices):
                print(f"⚠️ WARNING: initial_pose length ({len(initial_pose)}) doesn't match controllable joints ({len(self.controllable_joint_indices)}). Using default pose.")
            else:
                for i, joint_index in enumerate(self.controllable_joint_indices):
                    p.resetJointState(
                        bodyUniqueId=self.robot_id,
                        jointIndex=joint_index,
                        targetValue=initial_pose[i],
                        targetVelocity=0.0,
                        physicsClientId=self._physics_client_id
                    )
                # Allow pybullet to settle the pose for a few steps
                for _ in range(20):
                    p.stepSimulation(physicsClientId=self._physics_client_id)
        else:
            print("  [Env Reset] No initial_pose provided. Using default zero pose.")
            # Default pose is usually all zeros, which resetJointState defaults to if not called

        observation = self._get_observation()
        info = {} # Gymnasium standard requires info dict

        return observation, info

    def step(self, action):
        """
        Applies an action, steps the simulation, and returns the results.

        Args:
            action (np.ndarray): The torques to apply to each controllable joint.

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # --- Apply Action (Torque Control) ---
        # Assuming action is normalized between -1 and 1. Scale it.
        # Max torque values can be tuned or taken from URDF effort limits.
        max_torque = 100.0 # Example max torque - TUNE THIS
        scaled_action = action * max_torque

        for i, joint_index in enumerate(self.controllable_joint_indices):
            p.setJointMotorControl2(
                bodyIndex=self.robot_id,
                jointIndex=joint_index,
                controlMode=p.TORQUE_CONTROL,
                force=scaled_action[i],
                physicsClientId=self._physics_client_id
            )

        # --- Step Simulation ---
        p.stepSimulation(physicsClientId=self._physics_client_id)
        self.current_step += 1

        # --- Get Results ---
        observation = self._get_observation()
        reward = self._compute_reward()
        terminated = self._is_terminated()
        truncated = self.current_step >= self.max_episode_steps
        info = {}

        # Slow down rendering if in GUI mode
        # if self._render_mode == 'human':
        #     time.sleep(self.time_step)

        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        """Gets the current state of the robot."""
        joint_angles = []
        joint_velocities = []
        for i in self.controllable_joint_indices:
            state = p.getJointState(self.robot_id, i, physicsClientId=self._physics_client_id)
            joint_angles.append(state[0])
            joint_velocities.append(state[1])

        torso_pos, torso_orn_quat = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self._physics_client_id)
        torso_lin_vel, torso_ang_vel = p.getBaseVelocity(self.robot_id, physicsClientId=self._physics_client_id)

        observation = np.concatenate([
            joint_angles,
            joint_velocities,
            torso_orn_quat,
            torso_lin_vel,
            torso_ang_vel
        ]).astype(np.float32)

        return observation

    def _compute_reward(self):
        """Computes the reward for the current state."""
        # --- Reward Weights (TUNE THESE) ---
        w_vel = 1.0     # Reward for forward velocity
        w_live = 0.1    # Small reward for not falling
        w_energy = 0.001 # Small penalty for high torques (energy use)

        # Get torso state
        torso_pos, torso_orn_quat = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self._physics_client_id)
        torso_lin_vel, _ = p.getBaseVelocity(self.robot_id, physicsClientId=self._physics_client_id)

        # 1. Forward Velocity Reward (r_vel)
        # Assuming x-axis is forward
        forward_velocity = torso_lin_vel[0]
        r_vel = w_vel * forward_velocity

        # 2. Alive Bonus (r_live)
        # Check if the robot has fallen (use the same logic as termination)
        is_alive = torso_pos[2] > self.fall_threshold_height # Check height
        if is_alive:
             torso_orn_euler = p.getEulerFromQuaternion(torso_orn_quat)
             pitch = torso_orn_euler[1] # Pitch is rotation around Y-axis
             if abs(pitch) > self.fall_threshold_pitch:
                 is_alive = False
        r_live = w_live if is_alive else 0.0

        # 3. Energy Penalty (r_energy) - Optional but good practice
        # Sum of squared applied torques (get actual applied torques)
        applied_torques = []
        for i in self.controllable_joint_indices:
            joint_state = p.getJointState(self.robot_id, i, physicsClientId=self._physics_client_id)
            applied_torques.append(joint_state[3]) # jointMotorTorque
        r_energy = w_energy * np.sum(np.square(applied_torques))

        # 4. Fall Penalty (incorporated into terminated check)
        fall_penalty = -10.0 if not is_alive else 0.0 # Large penalty for falling

        # --- Total Reward ---
        reward = r_vel + r_live - r_energy + fall_penalty

        return reward

    def _is_terminated(self):
        """Checks if the episode should terminate (e.g., robot fell)."""
        torso_pos, torso_orn_quat = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self._physics_client_id)

        # Check height
        if torso_pos[2] < self.fall_threshold_height:
            # print("Termination: Fell below height threshold.")
            return True

        # Check orientation (pitch)
        torso_orn_euler = p.getEulerFromQuaternion(torso_orn_quat)
        pitch = torso_orn_euler[1]
        if abs(pitch) > self.fall_threshold_pitch:
            # print("Termination: Exceeded pitch threshold.")
            return True

        return False

    def render(self):
        """ Renders the environment. """
        if self._render_mode == 'rgb_array':
            # --- Capture Image from PyBullet ---
            # Define camera parameters (adjust as needed)
            base_pos, _ = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self._physics_client_id)
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=base_pos,  # Follow the robot's base
                distance=1.5,
                yaw=30,
                pitch=-20,
                roll=0,
                upAxisIndex=2,
                physicsClientId=self._physics_client_id
            )
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=60,
                aspect=1.0,  # Assuming square aspect ratio for simplicity
                nearVal=0.1,
                farVal=100.0,
                physicsClientId=self._physics_client_id
            )
            # Get the image data
            img_arr = p.getCameraImage(
                width=224,  # Example width
                height=224,  # Example height
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL,  # Use hardware renderer if available
                physicsClientId=self._physics_client_id
            )

            # --- Process the Image Data ---
            width, height, rgb_pixels, _, _ = img_arr
            # Reshape the flat RGB pixel array into an image array (H, W, C)
            # PyBullet returns RGBA, so we take the first 3 channels
            np_image_array = np.array(rgb_pixels).reshape((height, width, 4))[:, :, :3]

            # Convert RGB to BGR if needed by other libraries (like OpenCV)
            # np_image_array = cv2.cvtColor(np_image_array, cv2.COLOR_RGB2BGR)

            return np_image_array  # Return the processed numpy array

        elif self._render_mode == 'human':
            # Human mode is handled by p.GUI connection, nothing needed here
            pass
        else:
            # Handle other modes or raise an error
            raise ValueError(f"Unsupported render mode: {self._render_mode}")


    def close(self):
        """Cleans up the environment."""
        if self._physics_client_id >= 0:
            p.disconnect(self._physics_client_id)
            self._physics_client_id = -1
        print("--- [Module 2] Environment Closed ---")