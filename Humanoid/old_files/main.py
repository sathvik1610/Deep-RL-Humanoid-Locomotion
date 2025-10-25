# main.py

import numpy as np
import os
import traceback

# Import Module 1 class
from old_files.pose_converter import MultiPersonMediaPipeConverter
# Import Module 2 class
from humanoid_env import HumanoidWalkEnv
import pybullet as p # Import pybullet to check connection

def run_project():
    # --- Define paths ---
    URDF_FILE_PATH = "biped.urdf"
    DETECTOR_MODEL_PATH = "../models/efficientdet_lite2.tflite"  # Use Lite2 if downloaded
    IMAGE_FILE_NAME = "demo.jpg" # Your test image (man waving)
    IMAGE_FILE_PATH = f"images/{IMAGE_FILE_NAME}"

    theta_init = None # Initialize theta_init

    # === Run Module 1 ===
    try:
        print("\n--- Running Module 1 ---")
        if not os.path.exists(URDF_FILE_PATH): raise FileNotFoundError(f"URDF file not found: {URDF_FILE_PATH}")
        if not os.path.exists(DETECTOR_MODEL_PATH): raise FileNotFoundError(f"Detector model not found: {DETECTOR_MODEL_PATH}")
        if not os.path.exists(IMAGE_FILE_PATH): raise FileNotFoundError(f"Image file not found: {IMAGE_FILE_PATH}")

        converter = MultiPersonMediaPipeConverter(
            urdf_path=URDF_FILE_PATH,
            detector_model_path=DETECTOR_MODEL_PATH,
            min_vis_threshold=0.5,
            min_detect_conf=0.2
        )
        theta_init, _, _ = converter.run_pipeline(IMAGE_FILE_PATH) # Get theta_init

    except FileNotFoundError as e:
        print(f"❌❌❌ FILE ERROR in Module 1: {e}")
        return # Stop if Module 1 fails
    except Exception as e:
        print(f"❌❌❌ Error during Module 1: {e}")
        traceback.print_exc()
        return

    if theta_init is None:
        print("❌ Module 1 did not produce a theta_init. Exiting.")
        return

    # === Run Module 2 (Simple Test) ===
    env = None # Initialize env
    physics_client_id = -1 # Track client ID for safety
    try:
        print("\n--- Running Module 2 (Simple Test) ---")
        # --- Create Environment ---
        env = HumanoidWalkEnv(urdf_path=URDF_FILE_PATH, render_mode='human')

        # --- Reset with Initial Pose ---
        print("Resetting environment with theta_init...")
        observation, info = env.reset(initial_pose=theta_init)

        # ------------------------------------
        # --- FIX: Get Client ID *AFTER* Reset ---
        # ------------------------------------
        # env.reset() creates the actual connection, so get the ID now
        physics_client_id = env._physics_client_id
        # ------------------------------------

        print("Initial Observation:", np.round(observation[:10], 2), "...") # Print first 10 obs elements

        # --- Run a few steps with random actions ---
        print("\nRunning 100 steps with random actions (or until fall)...")
        total_reward = 0
        step = 0 # Initialize step count
        for step in range(100):
            random_action = env.action_space.sample() # Sample random torques
            observation, reward, terminated, truncated, info = env.step(random_action)
            total_reward += reward

            if terminated or truncated:
                print(f"Episode finished after {step+1} steps.")
                break
        else:
             print(f"Loop finished after {step+1} steps (max steps reached).")


        print(f"\nTotal reward over {step+1} steps: {total_reward:.3f}")

        # --- PAUSE TO KEEP WINDOW OPEN ---
        print("\nSimulation finished.")
        # Now this check should work correctly
        if p.isConnected(physics_client_id):
             input(">>> Press Enter in this terminal to close the PyBullet window...")
        else:
             print("PyBullet window was already closed (unexpected).") # Added warning

    except Exception as e:
        print(f"❌❌❌ Error during Module 2 Test: {e}")
        traceback.print_exc()
    finally:
        # --- Clean up ---
        print("Closing environment...")
        if env is not None:
            env.close()

# This makes the script runnable
if __name__ == "__main__":
    run_project()