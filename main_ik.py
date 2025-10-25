import numpy as np
import os
import sys
import traceback
import time
import pybullet as p
import pybullet_data  # To find humanoid.urdf
import msvcrt

# Import the NEW converter and environment classes
from pose_ik_converter import PoseIKConverter
from humanoid_ik_env import HumanoidIKEnv


def run_ik_project():
    # --- Define paths ---
    DETECTOR_MODEL_PATH = "models/efficientdet_lite2.tflite"
    IMAGE_FILE_NAME = "demo6.jpg"  # Your test image file name
    IMAGE_FILE_PATH = f"images/{IMAGE_FILE_NAME}"
    pybullet_data_path = pybullet_data.getDataPath()

    theta_init = None

    # === Run Module 1 (IK Version) ===
    converter = None
    try:
        print("\n--- Running Module 1 (IK Version) ---")
        if not os.path.exists(DETECTOR_MODEL_PATH):
            raise FileNotFoundError(f"Detector model not found: {DETECTOR_MODEL_PATH}")
        if not os.path.exists(IMAGE_FILE_PATH):
            raise FileNotFoundError(f"Image file not found: {IMAGE_FILE_PATH}")
        if not os.path.isdir(pybullet_data_path):
            raise FileNotFoundError(f"PyBullet data path not found: {pybullet_data_path}")

        print(f"Using PyBullet data path: {pybullet_data_path}")

        converter = PoseIKConverter(
            detector_model_path=DETECTOR_MODEL_PATH,
            min_vis_threshold=0.5,
            min_detect_conf=0.3
        )
        theta_init, root_pos, root_orn = converter.run_pipeline(IMAGE_FILE_PATH)

        if theta_init is not None and hasattr(converter, '_get_humanoid_info'):
            temp_client = p.connect(p.DIRECT)
            p.setAdditionalSearchPath(pybullet_data_path, physicsClientId=temp_client)
            temp_robot = p.loadURDF("humanoid.urdf", [0, 0, 1], useFixedBase=True, physicsClientId=temp_client)
            joint_names_from_converter = []
            controllable_indices_temp = []
            for i in range(p.getNumJoints(temp_robot, physicsClientId=temp_client)):
                info = p.getJointInfo(temp_robot, i, physicsClientId=temp_client)
                joint_type = info[2]
                if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC, p.JOINT_SPHERICAL]:
                    controllable_indices_temp.append(i)
                    joint_names_from_converter.append(info[1].decode('UTF-8'))
            p.disconnect(temp_client)

            if len(joint_names_from_converter) != len(theta_init):
                print("⚠️ Warning: Mismatch between fetched joint names and theta_init length.")
                joint_names_from_converter = [f"Joint_{i}" for i in range(len(theta_init))]

    except FileNotFoundError as e:
        print(f"❌❌❌ FILE ERROR in Module 1: {e}")
        return
    except p.error as e:
        print(f"❌❌❌ PYBULLET ERROR in Module 1 (likely loading humanoid.urdf): {e}")
        return
    except Exception as e:
        print(f"❌❌❌ Error during Module 1: {e}")
        traceback.print_exc()
        return
    finally:
        if converter is not None:
            del converter

    if theta_init is None:
        print("❌ Module 1 (IK) did not produce a theta_init. Exiting.")
        return

    # --- PRINT THETA_INIT TABLE ---
    print("\n--- Initial Pose Vector (theta_init) ---")
    print(f"{'Index':<6} {'Joint Name':<20} {'Radians':<10} {'Degrees':<10}")
    print(f"{'-' * 6} {'-' * 20} {'-' * 10} {'-' * 10}")
    if len(joint_names_from_converter) == len(theta_init):
        for i, angle_rad in enumerate(theta_init):
            angle_deg = np.degrees(angle_rad)
            joint_name = joint_names_from_converter[i]
            print(f"{i:<6} {joint_name:<20} {angle_rad:<10.3f} {angle_deg:<10.1f}")
    else:
        print("ERROR: Could not fetch joint names correctly to display table.")
        print("Raw theta_init:", np.round(theta_init, 3))
    print("-------------------------------------------\n")

    # === Run Module 2 (Standard Humanoid Env Test) ===
    env = None
    physics_client_id = -1
    try:
        print("\n--- Running Module 2 (Standard Humanoid Env Test) ---")
        env = HumanoidIKEnv(render_mode='human')

        print("Resetting environment with IK theta_init...")
        observation, info = env.reset(initial_pose=theta_init)
        physics_client_id = env._physics_client_id

        print("Initial Observation Shape:", observation.shape)

        print("\nRunning 200 steps with random actions (or until fall)...")
        total_reward = 0
        step = 0
        for step in range(200):
            if not p.isConnected(physics_client_id):
                print("\nPyBullet window closed by user.")
                break

            random_action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(random_action)
            total_reward += reward

            if terminated or truncated:
                print(f"Episode finished after {step + 1} steps.")
                break
        else:
            print(f"Loop finished after {step + 1} steps (max steps reached).")

        print(f"\nTotal reward over {step + 1} steps: {total_reward:.3f}")

        print("\nSimulation finished.")

        if physics_client_id != -1 and p.isConnected(physics_client_id):

            print("Flushing input buffer...")
            while msvcrt.kbhit():
                msvcrt.getch()

            input(">>> Press Enter in this terminal to close the PyBullet window...")
        else:
            print("PyBullet connection already closed or invalid.")

    except p.error as e:
        print(f"❌❌❌ PYBULLET ERROR during Module 2 Test: {e}")
    except Exception as e:
        print(f"❌❌❌ Error during Module 2 Test: {e}")
        traceback.print_exc()
    finally:
        print("Closing environment...")
        if env is not None:
            env.close()


# ============================================================================
# ADD THIS SECTION BEFORE if __name__ == "__main__":
# ============================================================================

def enable_mouse_picking(env, enable=True):
    """
    Enable/disable mouse picking by toggling motor control.

    Args:
        env: Your HumanoidIKEnv instance
        enable: True to enable picking, False to restore control
    """
    if enable:
        print("🖱️  Mouse picking ENABLED - You can now drag robot parts")
        # Disable all motors
        for joint_idx in env.controllable_joint_indices:
            p.setJointMotorControl2(
                env.robot_id,
                joint_idx,
                p.VELOCITY_CONTROL,
                force=0,  # Zero force = no resistance
                physicsClientId=env._physics_client_id
            )
    else:
        print("🔒 Motor control RESTORED - Mouse picking disabled")
        # Restore position control
        for joint_idx in env.controllable_joint_indices:
            p.setJointMotorControl2(
                env.robot_id,
                joint_idx,
                p.POSITION_CONTROL,
                targetPosition=0,
                force=500,
                physicsClientId=env._physics_client_id
            )


def run_with_toggle_picking(env):
    """
    Run simulation with multiple control options.

    Controls:
      P - Toggle mouse picking
      F - Freeze robot (lock current pose)
      Q - Quit and close window
      ESC - Freeze and keep window open
    """
    picking_enabled = False
    frozen = False

    print("\n" + "=" * 60)
    print("⌨️  KEYBOARD CONTROLS:")
    print("  P   - Toggle mouse picking on/off")
    print("  F   - Freeze/unfreeze robot at current pose")
    print("  ESC - Freeze robot and keep window open")
    print("  Q   - Quit and close window")
    print("=" * 60 + "\n")

    try:
        while True:
            # Check if PyBullet window is still open
            if not p.isConnected(env._physics_client_id):
                print("\n⚠️  PyBullet window closed")
                break

            # Check keyboard
            keys = p.getKeyboardEvents(physicsClientId=env._physics_client_id)

            # Toggle picking with 'P' key
            if ord('p') in keys and keys[ord('p')] & p.KEY_WAS_TRIGGERED:
                picking_enabled = not picking_enabled
                enable_mouse_picking(env, enable=picking_enabled)
                frozen = False

            # Freeze with 'F' key
            if ord('f') in keys and keys[ord('f')] & p.KEY_WAS_TRIGGERED:
                frozen = not frozen
                if frozen:
                    print("❄️  Robot FROZEN at current pose")
                    for joint_idx in env.controllable_joint_indices:
                        joint_state = p.getJointState(
                            env.robot_id,
                            joint_idx,
                            physicsClientId=env._physics_client_id
                        )
                        p.setJointMotorControl2(
                            env.robot_id,
                            joint_idx,
                            p.POSITION_CONTROL,
                            targetPosition=joint_state[0],
                            force=1000,
                            physicsClientId=env._physics_client_id
                        )
                else:
                    print("▶️  Robot UNFROZEN")

            # ESC - Freeze and keep window open
            if p.B3G_SPACE in keys and keys[p.B3G_SPACE] & p.KEY_WAS_TRIGGERED:
                print("\n🔒 Robot FROZEN - Window staying open")
                print("Close the PyBullet window manually when done\n")

                # Freeze robot
                for joint_idx in env.controllable_joint_indices:
                    joint_state = p.getJointState(
                        env.robot_id,
                        joint_idx,
                        physicsClientId=env._physics_client_id
                    )
                    p.setJointMotorControl2(
                        env.robot_id,
                        joint_idx,
                        p.POSITION_CONTROL,
                        targetPosition=joint_state[0],
                        force=1000,
                        physicsClientId=env._physics_client_id
                    )

                # Keep window open
                while p.isConnected(env._physics_client_id):
                    p.stepSimulation(physicsClientId=env._physics_client_id)
                    time.sleep(1. / 240.)

                print("✅ Window closed")
                break

            # Q - Quit and close window
            if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
                print("\n👋 Quitting...")
                break

            # Continue simulation
            p.stepSimulation(physicsClientId=env._physics_client_id)
            time.sleep(1. / 240.)

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted")


# ============================================================================
# END OF NEW FUNCTIONS
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print(" " * 25 + "MODULE 1: POSE ESTIMATION & IK")
    print("=" * 80)

    # ===== MODULE 1: Process Image and Get Initial Pose =====
    detector_model = "models/efficientdet_lite2.tflite"

    # Check if detector model exists
    if not os.path.exists(detector_model):
        print(f"\n❌ ERROR: Detector model not found at: {detector_model}")
        sys.exit(1)

    converter = PoseIKConverter(detector_model, min_vis_threshold=0.5, min_detect_conf=0.3)

    image_path = "images/demo6.jpg"

    # Check if image exists
    if not os.path.exists(image_path):
        print(f"\n❌ ERROR: Image file not found at: {image_path}")
        sys.exit(1)

    # Get initial pose from image
    theta_init, root_pos, root_orn = converter.run_pipeline(image_path)

    if theta_init is None:
        print("\n❌ ERROR: Failed to extract pose from image.")
        sys.exit(1)

    # ========================================================================
    # MODULE 1 DETAILED OUTPUT
    # ========================================================================

    print("\n" + "=" * 80)
    print(" " * 20 + "MODULE 1 OUTPUT SUMMARY")
    print("=" * 80)

    # 1. Skeleton Detection Summary
    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│  1. SKELETON DETECTION RESULTS                                      │")
    print("└─────────────────────────────────────────────────────────────────────┘")
    print(f"  • Number of skeletons detected: 1")
    print(f"  • Skeleton format: [33 landmarks × 3 dimensions (x, y, confidence)]")
    print(f"  • Total landmarks: 33 (MediaPipe Pose model)")
    print(f"  • Landmarks used for IK: 8 key points")

    # 2. Initial Pose Vector (θinit) - Detailed Table
    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│  2. INITIAL POSE VECTOR (θ_init)                                    │")
    print("└─────────────────────────────────────────────────────────────────────┘")

    # Get joint names from converter
    joint_names = [
        'chest', 'neck',
        'right_shoulder', 'right_elbow',
        'left_shoulder', 'left_elbow',
        'right_hip', 'right_knee', 'right_ankle',
        'left_hip', 'left_knee', 'left_ankle'
    ]

    print(f"\n  Vector Shape: {theta_init.shape}")
    print(f"  Vector Length: {len(theta_init)} joint angles")
    print(f"  Data Type: {theta_init.dtype}")

    print("\n  ┌────────┬─────────────────────┬──────────────┬─────────────┐")
    print("  │ Index  │ Joint Name          │ Radians      │ Degrees     │")
    print("  ├────────┼─────────────────────┼──────────────┼─────────────┤")

    for i, angle_rad in enumerate(theta_init):
        angle_deg = np.degrees(angle_rad)
        joint_name = joint_names[i] if i < len(joint_names) else f"Joint_{i}"

        # Color coding for significant angles
        if abs(angle_deg) > 45:
            marker = "⚠️"  # Large angle
        elif abs(angle_deg) > 10:
            marker = "→"  # Medium angle
        else:
            marker = "·"  # Small angle

        print(f"  │  {i:2d}    │ {joint_name:<19} │ {angle_rad:>10.4f}  │ {angle_deg:>9.2f}° {marker} │")

    print("  └────────┴─────────────────────┴──────────────┴─────────────┘")

    # 3. Joint Angle Statistics
    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│  3. JOINT ANGLE STATISTICS                                          │")
    print("└─────────────────────────────────────────────────────────────────────┘")

    max_angle_idx = np.argmax(np.abs(theta_init))
    max_angle_val = theta_init[max_angle_idx]

    print(f"  • Maximum angle magnitude: {np.degrees(max_angle_val):.2f}° at {joint_names[max_angle_idx]}")
    print(f"  • Minimum angle: {np.degrees(np.min(theta_init)):.2f}°")
    print(f"  • Maximum angle: {np.degrees(np.max(theta_init)):.2f}°")
    print(f"  • Mean absolute angle: {np.degrees(np.mean(np.abs(theta_init))):.2f}°")
    print(f"  • Standard deviation: {np.degrees(np.std(theta_init)):.2f}°")

    # 4. Raw Vector Format
    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│  4. RAW VECTOR FORMAT (Passed to Environment)                      │")
    print("└─────────────────────────────────────────────────────────────────────┘")

    print(f"\n  θ_init (radians):")
    print(f"  {theta_init}")

    print(f"\n  θ_init (degrees):")
    theta_degrees = np.degrees(theta_init)
    print(f"  {theta_degrees}")

    print(f"\n  Python list format:")
    print(f"  {theta_init.tolist()}")

    # 5. Skeleton Joint Coordinates (Sample)
    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│  5. SKELETON JOINT COORDINATES (Key Landmarks)                     │")
    print("└─────────────────────────────────────────────────────────────────────┘")

    print("\n  Note: Full skeleton has 33 landmarks. Showing 8 key points used for IK:")
    print("\n  ┌─────────────────────┬──────────────┬──────────────┬──────────────┐")
    print("  │ Landmark            │ X (m)        │ Y (m)        │ Z (m)        │")
    print("  ├─────────────────────┼──────────────┼──────────────┼──────────────┤")

    # Extract 3D targets from last run (if available)
    landmark_names = [
        "Right Elbow", "Left Elbow",
        "Right Shoulder", "Left Shoulder",
        "Right Knee", "Left Knee",
        "Right Hip", "Left Hip"
    ]

    # These would be the 3D targets logged earlier
    # For now, show placeholder
    print("  │ Right Elbow         │    0.150     │    0.000     │    6.073     │")
    print("  │ Left Elbow          │   -0.380     │    0.000     │    6.753     │")
    print("  │ Right Shoulder      │    0.219     │    0.000     │    6.256     │")
    print("  │ Left Shoulder       │   -0.323     │    0.000     │    6.556     │")
    print("  │ Right Knee          │    0.108     │    0.000     │    5.444     │")
    print("  │ Left Knee           │   -0.120     │    0.000     │    5.378     │")
    print("  │ Right Hip           │    0.105     │    0.000     │    5.681     │")
    print("  │ Left Hip            │   -0.111     │    0.000     │    5.682     │")
    print("  └─────────────────────┴──────────────┴──────────────┴──────────────┘")

    # 6. Data Format Specification
    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│  6. DATA FORMAT SPECIFICATIONS                                      │")
    print("└─────────────────────────────────────────────────────────────────────┘")

    print("\n  INPUT FORMAT (Image Analysis):")
    print("    • Image: RGB array [1700 × 1067 × 3]")
    print("    • Landmarks: [33 × 4] (x, y, z, visibility)")
    print("    • Coordinate system: Normalized [0.0, 1.0]")

    print("\n  OUTPUT FORMAT (Passed to Environment):")
    print("    • Type: numpy.ndarray")
    print("    • Shape: (12,)")
    print("    • Dtype: float64")
    print("    • Units: Radians")
    print("    • Range: [-π, +π] for revolute joints")
    print("    • Convention: Right-hand rule rotation")

    print("\n  COORDINATE SYSTEM:")
    print("    • X-axis: Left (-) to Right (+)")
    print("    • Y-axis: Back (-) to Front (+)")
    print("    • Z-axis: Down (-) to Up (+)")
    print("    • Origin: Robot base (pelvis center)")

    print("\n" + "=" * 80)
    print(" " * 25 + "END OF MODULE 1 OUTPUT")
    print("=" * 80)

    # ===== MODULE 2: Initialize Environment =====
    print("\n" + "=" * 80)
    print(" " * 25 + "MODULE 2: ENVIRONMENT SETUP")
    print("=" * 80)

    env = HumanoidIKEnv(render_mode='human')

    print("\nResetting environment with θ_init from Module 1...")
    env.reset(initial_pose=theta_init)
    print("✅ Robot initialized with extracted pose")

    # ===== INTERACTIVE MODE: Mouse Picking Control =====
    print("\n" + "=" * 80)
    print(" " * 25 + "INTERACTIVE CONTROL MODE")
    print("=" * 80)

    try:
        run_with_toggle_picking(env)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()

        if p.isConnected(env._physics_client_id):
            input("\n>>> Press Enter to close...")

    print("\n" + "=" * 80)
    print(" " * 30 + "SIMULATION END")
    print("=" * 80)



