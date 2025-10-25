import cv2
import numpy as np
import os
import sys
import traceback
import time
import pybullet as p
import pybullet_data  # To find humanoid.urdf
import msvcrt

# Import the NEW converter and environment classes
# Ensure you are using pose_ik_converter (v5) and humanoid_ik_env (with crash fix)
from pose_ik_converter import PoseIKConverter
from humanoid_ik_env import HumanoidIKEnv

image_path = "images/demo10.jpg"
def run_ik_project():
    # --- Define paths ---
    DETECTOR_MODEL_PATH = "models/efficientdet_lite2.tflite"
    IMAGE_FILE_NAME = "demo.jpg"  # Your test image file name
    IMAGE_FILE_PATH = f"images/{IMAGE_FILE_NAME}"
    pybullet_data_path = pybullet_data.getDataPath()

    theta_init = None

    # === Run Module 1 (Geometric Version) ===
    converter = None
    try:
        # This section remains unchanged - fetches theta_init and joint names
        print("\n--- Running Module 1 (Geometric Version) ---")
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

        if theta_init is not None:
            temp_client = p.connect(p.DIRECT)
            p.setAdditionalSearchPath(pybullet_data_path, physicsClientId=temp_client)
            try:
                temp_robot = p.loadURDF("humanoid.urdf", [0, 0, 1], useFixedBase=True, physicsClientId=temp_client)
                joint_names_from_converter = []
                for i in range(p.getNumJoints(temp_robot, physicsClientId=temp_client)):
                    info = p.getJointInfo(temp_robot, i, physicsClientId=temp_client)
                    joint_type = info[2]
                    if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC, p.JOINT_SPHERICAL]:
                        joint_names_from_converter.append(info[1].decode('UTF-8'))
                p.disconnect(temp_client)

                if len(joint_names_from_converter) != len(theta_init):
                    print(
                        f"⚠️ Warning: Mismatch between fetched joint names ({len(joint_names_from_converter)}) and theta_init length ({len(theta_init)}).")
                    joint_names_from_converter = [f"Joint_{i}" for i in range(len(theta_init))]

            except p.error as e:
                print(f"❌ PYBULLET ERROR while fetching joint names: {e}")
                p.disconnect(temp_client)
                joint_names_from_converter = [f"Joint_{i}" for i in range(len(theta_init))]
        else:
            joint_names_from_converter = [f"Joint_{i}" for i in range(12)]


    except FileNotFoundError as e:
        print(f"❌❌❌ FILE ERROR in Module 1: {e}")
        return
    except p.error as e:
        print(f"❌❌❌ PYBULLET ERROR in Module 1: {e}")
        return
    except Exception as e:
        print(f"❌❌❌ Error during Module 1: {e}")
        traceback.print_exc()
        return
    finally:
        if converter is not None:
            del converter

    if theta_init is None:
        print("❌ Module 1 did not produce a theta_init. Exiting.")
        return

    # This function is not used by your __main__ block
    pass


# ============================================================================
# INTERACTIVE FUNCTIONS
# ============================================================================

def enable_mouse_picking(env, enable=True):
    """Enable/disable mouse picking by toggling motor control."""
    if enable:
        print("🖱️  Mouse picking ENABLED - You can now drag robot parts (Limp)")
        for joint_idx in env.controllable_joint_indices:
            p.setJointMotorControl2(
                env.robot_id, joint_idx, p.VELOCITY_CONTROL, force=0,
                physicsClientId=env._physics_client_id
            )
    else:
        print("🔒 Motor control RESTORED - Mouse picking disabled (Will go Limp)")
        # When disabling picking, don't restore position control immediately,
        # let the F key logic handle freezing/holding pose. Just disable motors.
        for joint_idx in env.controllable_joint_indices:
            p.setJointMotorControl2(
                env.robot_id, joint_idx, p.VELOCITY_CONTROL, force=0,
                physicsClientId=env._physics_client_id
            )


# *** THIS IS THE NEW F-KEY LOGIC ***
# *** THIS IS THE NEW F-KEY LOGIC (v2) ***
def run_with_toggle_picking(env, initial_pose_vector):
    """
    Run simulation with multiple control options.
    Starts FROZEN in the initial pose. 'F' toggles limp/frozen.
    """
    picking_enabled = False
    frozen = True  # START FROZEN

    print("\n" + "=" * 60)
    print("⌨️  KEYBOARD CONTROLS:")
    print("  P   - Toggle mouse picking on/off (Goes Limp)")
    print("  F   - Unfreeze (go Limp) / Freeze (in initial pose)")
    print("  ESC - Freeze robot (current pose) and keep window open")
    print("  Q   - Quit and close window")
    print("=" * 60 + "\n")

    # --- START OF FIX: APPLY INITIAL FREEZE ---
    # We must apply the freeze logic ONCE before the loop
    # This logic correctly checks joint types
    print("❄️  Robot is FROZEN in INITIAL pose. Press 'F' to unfreeze.")
    for i, joint_idx in enumerate(env.controllable_joint_indices):
        info = p.getJointInfo(
            env.robot_id, joint_idx, physicsClientId=env._physics_client_id
        )
        joint_type = info[2]
        target_pos = initial_pose_vector[i]

        if joint_type == p.JOINT_REVOLUTE or joint_type == p.JOINT_PRISMATIC:
            p.setJointMotorControl2(
                env.robot_id, joint_idx, p.POSITION_CONTROL,
                targetPosition=target_pos,
                force=1000,
                physicsClientId=env._physics_client_id
            )
        elif joint_type == p.JOINT_SPHERICAL:
            target_quat = p.getQuaternionFromEuler([0, 0, target_pos])
            p.setJointMotorControlMultiDof(
                env.robot_id, joint_idx, p.POSITION_CONTROL,
                targetPosition=target_quat,
                force=[1000, 1000, 1000],  # Force must be a 3D vector
                physicsClientId=env._physics_client_id
            )
    # --- END OF FIX: APPLY INITIAL FREEZE ---

    try:
        while True:
            if not p.isConnected(env._physics_client_id):
                print("\n⚠️  PyBullet window closed")
                break

            keys = p.getKeyboardEvents(physicsClientId=env._physics_client_id)

            # Toggle picking ('P') - goes limp
            if ord('p') in keys and keys[ord('p')] & p.KEY_WAS_TRIGGERED:
                picking_enabled = not picking_enabled
                enable_mouse_picking(env, enable=picking_enabled)
                frozen = False  # Always unfreeze and go limp if picking

            # Freeze/Unfreeze ('F')
            if ord('f') in keys and keys[ord('f')] & p.KEY_WAS_TRIGGERED:
                frozen = not frozen
                if frozen:
                    # --- START OF FIX: 'F' KEY FREEZE ---
                    print("❄️  Robot FROZEN in INITIAL pose")
                    for i, joint_idx in enumerate(env.controllable_joint_indices):
                        info = p.getJointInfo(
                            env.robot_id, joint_idx, physicsClientId=env._physics_client_id
                        )
                        joint_type = info[2]
                        target_pos = initial_pose_vector[i]

                        if joint_type == p.JOINT_REVOLUTE or joint_type == p.JOINT_PRISMATIC:
                            p.setJointMotorControl2(
                                env.robot_id, joint_idx, p.POSITION_CONTROL,
                                targetPosition=target_pos,
                                force=1000,
                                physicsClientId=env._physics_client_id
                            )
                        elif joint_type == p.JOINT_SPHERICAL:
                            target_quat = p.getQuaternionFromEuler([0, 0, target_pos])
                            p.setJointMotorControlMultiDof(
                                env.robot_id, joint_idx, p.POSITION_CONTROL,
                                targetPosition=target_quat,
                                force=[1000, 1000, 1000],
                                physicsClientId=env._physics_client_id
                            )
                    # --- END OF FIX: 'F' KEY FREEZE ---
                    else:
                        # --- START OF FIX: 'F' KEY UNFREEZE (LIMP) ---
                        print("▶️  Robot UNFROZEN (Limp)")
                        for joint_idx in env.controllable_joint_indices:
                            info = p.getJointInfo(
                                env.robot_id, joint_idx, physicsClientId=env._physics_client_id
                            )
                            joint_type = info[2]

                            if joint_type == p.JOINT_REVOLUTE or joint_type == p.JOINT_PRISMATIC:
                                p.setJointMotorControl2(
                                    env.robot_id, joint_idx, p.VELOCITY_CONTROL, force=0,
                                    physicsClientId=env._physics_client_id
                                )
                            elif joint_type == p.JOINT_SPHERICAL:
                                p.setJointMotorControlMultiDof(
                                    env.robot_id, joint_idx, p.VELOCITY_CONTROL,  # <-- Problematic mode
                                    targetVelocity=[0, 0, 0],
                                    force=[0, 0, 0],
                                    physicsClientId=env._physics_client_id
                                )
                    # --- END OF FIX: 'F' KEY UNFREEZE (LIMP) ---

            # ESC - Freeze and keep window open (Freezes in CURRENT pose)
            if p.B3G_SPACE in keys and keys[p.B3G_SPACE] & p.KEY_WAS_TRIGGERED:
                print("\n🔒 Robot FROZEN (Current Pose) - Window staying open")
                print("Close the PyBullet window manually when done\n")
                if not frozen:  # Apply freeze logic if it was limp
                    for i, joint_idx in enumerate(env.controllable_joint_indices):
                        info = p.getJointInfo(
                            env.robot_id, joint_idx, physicsClientId=env._physics_client_id
                        )
                        joint_type = info[2]

                        if joint_type == p.JOINT_REVOLUTE or joint_type == p.JOINT_PRISMATIC:
                            joint_state = p.getJointState(
                                env.robot_id, joint_idx, physicsClientId=env._physics_client_id
                            )
                            p.setJointMotorControl2(
                                env.robot_id, joint_idx, p.POSITION_CONTROL,
                                targetPosition=joint_state[0], force=1000,  # Freeze current
                                physicsClientId=env._physics_client_id
                            )
                        elif joint_type == p.JOINT_SPHERICAL:
                            joint_state = p.getJointStateMultiDof(
                                env.robot_id, joint_idx, physicsClientId=env._physics_client_id
                            )
                            p.setJointMotorControlMultiDof(
                                env.robot_id, joint_idx, p.POSITION_CONTROL,
                                targetPosition=joint_state[0], force=[1000, 1000, 1000],  # Freeze current
                                physicsClientId=env._physics_client_id
                            )

                    frozen = True  # Mark as frozen
                # Keep window open loop
                while p.isConnected(env._physics_client_id):
                    p.stepSimulation(physicsClientId=env._physics_client_id)
                    time.sleep(1. / 240.)
                print("✅ Window closed")
                break

            # Q - Quit and close window
            if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
                print("\n👋 Quitting...")
                break

            p.stepSimulation(physicsClientId=env._physics_client_id)
            time.sleep(1. / 240.)

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted")


# ============================================================================
# END OF INTERACTIVE FUNCTIONS
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print(" " * 25 + "MODULE 1: POSE ESTIMATION & IK")  # Keep title consistent
    print("=" * 80)

    # ===== MODULE 1: Process Image and Get Initial Pose =====
    detector_model = "models/efficientdet_lite2.tflite"

    if not os.path.exists(detector_model):
        print(f"\n❌ ERROR: Detector model not found at: {detector_model}")
        sys.exit(1)

    converter = PoseIKConverter(detector_model, min_vis_threshold=0.5, min_detect_conf=0.3)

      # Using demo.jpg from your latest log
    # image_path = "images/demo8.jpg"
    # image_path = "images/demo6.jpg"

    if not os.path.exists(image_path):
        print(f"\n❌ ERROR: Image file not found at: {image_path}")
        sys.exit(1)

    # Get initial pose from image using the converter instance
    theta_init, root_pos, root_orn = converter.run_pipeline(image_path)

    if theta_init is None:
        print("\n❌ ERROR: Failed to extract pose from image.")
        sys.exit(1)

    # ========================================================================
    # MODULE 1 DETAILED OUTPUT (Unchanged)
    # ========================================================================
    # This section remains the same, printing the details of theta_init

    print("\n" + "=" * 80)
    print(" " * 20 + "MODULE 1 OUTPUT SUMMARY")
    print("=" * 80)

    # 1. Skeleton Detection Summary
    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│  1. SKELETON DETECTION RESULTS                                      │")
    print("└─────────────────────────────────────────────────────────────────────┘")
    print(f"  • Skeletons detected: (See list in [Task 1.2 Output] above)")
    print(f"  • Skeleton format: [33 landmarks × 3 (x, y, visibility)]")
    print(f"  • Total landmarks: 33 (MediaPipe Pose model)")
    print(f"  • Landmarks used for angles: 12 key points")  # Updated placeholder

    # 2. Initial Pose Vector (θinit) - Detailed Table
    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│  2. INITIAL POSE VECTOR (θ_init)                                    │")
    print("└─────────────────────────────────────────────────────────────────────┘")

    # Fetch joint names again for the table in __main__
    joint_names = []
    try:
        pybullet_data_path = pybullet_data.getDataPath()
        temp_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data_path, physicsClientId=temp_client)
        temp_robot = p.loadURDF("humanoid.urdf", [0, 0, 1], useFixedBase=True, physicsClientId=temp_client)
        for i in range(p.getNumJoints(temp_robot, physicsClientId=temp_client)):
            info = p.getJointInfo(temp_robot, i, physicsClientId=temp_client)
            joint_type = info[2]
            if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC, p.JOINT_SPHERICAL]:
                joint_names.append(info[1].decode('UTF-8'))
        p.disconnect(temp_client)
        if len(joint_names) != len(theta_init):
            print("⚠️ Warning: Main block joint name mismatch. Using generic names.")
            joint_names = [f"Joint_{i}" for i in range(len(theta_init))]
    except Exception:
        print("⚠️ Warning: Could not load URDF for joint names. Using generic names.")
        joint_names = [f"Joint_{i}" for i in range(len(theta_init))]

    print(f"\n  Vector Shape: {theta_init.shape}")
    print(f"  Vector Length: {len(theta_init)} joint angles")
    print(f"  Data Type: {theta_init.dtype}")

    print("\n  ┌────────┬─────────────────────┬──────────────┬─────────────┐")
    print("  │ Index  │ Joint Name          │ Radians      │ Degrees     │")
    print("  ├────────┼─────────────────────┼──────────────┼─────────────┤")

    for i, angle_rad in enumerate(theta_init):
        angle_deg = np.degrees(angle_rad)
        joint_name = joint_names[i] if i < len(joint_names) else f"Joint_{i}"
        marker = "·"
        if abs(angle_deg) > 45:
            marker = "⚠️"
        elif abs(angle_deg) > 10:
            marker = "→"
        print(f"  │  {i:2d}    │ {joint_name:<19} │ {angle_rad:>10.4f}  │ {angle_deg:>9.2f}° {marker} │")

    print("  └────────┴─────────────────────┴──────────────┴─────────────┘")

    # 3. Joint Angle Statistics
    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│  3. JOINT ANGLE STATISTICS                                          │")
    print("└─────────────────────────────────────────────────────────────────────┘")
    max_angle_idx = np.argmax(np.abs(theta_init))
    max_angle_val = theta_init[max_angle_idx]
    max_angle_joint_name = joint_names[max_angle_idx] if max_angle_idx < len(joint_names) else "N/A"
    print(f"  • Maximum angle magnitude: {np.degrees(max_angle_val):.2f}° at {max_angle_joint_name}")
    print(f"  • Minimum angle: {np.degrees(np.min(theta_init)):.2f}°")
    print(f"  • Maximum angle: {np.degrees(np.max(theta_init)):.2f}°")
    print(f"  • Mean absolute angle: {np.degrees(np.mean(np.abs(theta_init))):.2f}°")
    print(f"  • Standard deviation: {np.degrees(np.std(theta_init)):.2f}°")

    # 4. Raw Vector Format
    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│  4. RAW VECTOR FORMAT (Passed to Environment)                      │")
    print("└─────────────────────────────────────────────────────────────────────┘")
    print(f"\n  θ_init (radians):\n  {theta_init}")
    print(f"\n  θ_init (degrees):\n  {np.degrees(theta_init)}")
    print(f"\n  Python list format:\n  {theta_init.tolist()}")

    # 6. Data Format Specification
    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│  6. DATA FORMAT SPECIFICATIONS                                      │")
    print("└─────────────────────────────────────────────────────────────────────┘")
    print("\n  INPUT FORMAT (Image Analysis):")
    # Get image dimensions dynamically if possible, else use placeholder
    try:
        img_h, img_w, _ = cv2.imread(image_path).shape
        img_dims_str = f"[{img_h} x {img_w} x 3]"
    except:
        img_dims_str = "(e.g., [Height x Width x 3])"
    print(f"    • Image: RGB array {img_dims_str}")
    print("    • Landmarks: [33 × 4] (x, y, z, visibility)")
    print("    • Coordinate system: Normalized [0.0, 1.0]")
    print("\n  OUTPUT FORMAT (Passed to Environment):")
    print("    • Type: numpy.ndarray")
    print(f"    • Shape: {theta_init.shape}")
    print(f"    • Dtype: {theta_init.dtype}")
    print("    • Units: Radians")
    print("    • Range: Specific to each joint, constrained")
    print("    • Convention: Relative angles based on URDF T-Pose")
    print("\n  COORDINATE SYSTEM (Geometric Angle Calculation):")
    print("    • Torso 'Down' is vector from mid-shoulder to mid-hip.")
    print("    • Hip angles relative to Torso Down.")
    print("    • Shoulder angles relative to Torso Horizontal (T-Pose).")
    print("    • Knee/Elbow angles relative to parent limb.")
    print("    • Origin: Robot base (pelvis center)")

    print("\n" + "=" * 80)
    print(" " * 25 + "END OF MODULE 1 OUTPUT")
    print("=" * 80)

    # ===== MODULE 2: Initialize Environment =====
    print("\n" + "=" * 80)
    print(" " * 25 + "MODULE 2: ENVIRONMENT SETUP")
    print("=" * 80)

    env = None
    try:
        env = HumanoidIKEnv(render_mode='human')

        print("\nResetting environment with θ_init from Module 1...")
        # env.reset() handles setting the initial pose via resetJointState
        env.reset(initial_pose=theta_init)
        print("✅ Robot initialized with extracted pose")
        # ==========================================================
        # ### DEBUGGING CODE TO ADD (START) ###
        # ==========================================================
        print("\n[DEBUG] Reading joint states back from PyBullet after reset:")
        print("  ┌────────┬─────────────────────┬──────────────┬─────────────┐")
        print("  │ Index  │ Joint Name          │ Target (Deg) │ Actual (Deg)│")
        print("  ├────────┼─────────────────────┼──────────────┼─────────────┤")

        # Get names from the environment instance
        actual_joint_names = env.joint_names

        for i, joint_idx in enumerate(env.controllable_joint_indices):
            # Read the actual state from PyBullet
            joint_state = p.getJointState(
                env.robot_id, joint_idx, physicsClientId=env._physics_client_id
            )
            actual_angle_rad = joint_state[0]
            actual_angle_deg = np.degrees(actual_angle_rad)

            # Get target angle from theta_init
            target_angle_deg = np.degrees(theta_init[i])

            joint_name = actual_joint_names[i] if i < len(actual_joint_names) else f"Joint_{i}"

            print(f"  │  {i:2d}    │ {joint_name:<19} │ {target_angle_deg:>10.2f}° │ {actual_angle_deg:>11.2f}° │")

        print("  └────────┴─────────────────────┴──────────────┴─────────────┘")
        # ==========================================================
        # ### DEBUGGING CODE TO ADD (END) ###
        # ==========================================================
        # ===== INTERACTIVE MODE: Mouse Picking Control =====
        print("\n" + "=" * 80)
        print(" " * 25 + "INTERACTIVE CONTROL MODE")
        print("=" * 80)

        # *** Pass theta_init to the interactive function ***
        run_with_toggle_picking(env, theta_init)

    except Exception as e:
        print(f"\n❌ Error during Module 2: {e}")
        traceback.print_exc()
        # Check connection before prompting
        client_id_check = -1
        if env is not None and hasattr(env, '_physics_client_id'):
            client_id_check = env._physics_client_id
        if client_id_check != -1 and p.isConnected(client_id_check):
            input("\n>>> Press Enter to close...")
    finally:
        if env is not None:
            env.close()

    print("\n" + "=" * 80)
    print(" " * 30 + "SIMULATION END")
    print("=" * 80)