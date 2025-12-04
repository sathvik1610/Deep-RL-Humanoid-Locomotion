"""
Run Humanoid Walking Policy

Usage:
    python run_policy.py                              # Use trained model
    python run_policy.py --baseline                   # Use baseline model
    python run_policy.py --image images/person.jpg   # Start from image pose
    python run_policy.py --image img.jpg --baseline  # Image pose + baseline
    python run_policy.py --random                     # Random actions
"""

import argparse
import os
import time
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import SB3
try:
    from stable_baselines3 import PPO
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("❌ Stable-Baselines3 required. Run: pip install stable-baselines3")

# Import environment
try:
    from pybullet_envs.gym_locomotion_envs import HumanoidBulletEnv
    BULLET_ENV_AVAILABLE = True
except ImportError:
    BULLET_ENV_AVAILABLE = False
    print("❌ pybullet_envs required.")

# Import pose converter
try:
    from humanoid_walk.perception.pose_converter import PoseConverter
    POSE_CONVERTER_AVAILABLE = True
except ImportError:
    POSE_CONVERTER_AVAILABLE = False

import pickle
import pybullet as p


# ============================================================================
# PATHS
# ============================================================================
TRAINED_MODEL = "models/ppo_walking_trained.zip"
TRAINED_NORMALIZER = "models/ppo_walking_normalizer.pkl"
BASELINE_MODEL = "models/trained_walking_policy.zip"
BASELINE_NORMALIZER = "models/trained_normalizer.pkl"
DETECTOR_MODEL = "models/efficientdet_lite2.tflite"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def load_model(model_path, normalizer_path=None):
    """Load SB3 model and normalizer."""
    print(f"🔄 Loading: {os.path.basename(model_path)}")
    model = PPO.load(model_path)
    print("✅ Model loaded")
    
    normalizer = None
    if normalizer_path and os.path.exists(normalizer_path):
        with open(normalizer_path, 'rb') as f:
            normalizer = pickle.load(f)
        print("✅ Normalizer loaded")
    
    return model, normalizer


def normalize_obs(obs, normalizer):
    """Apply observation normalization."""
    if normalizer is None:
        return obs
    try:
        rms = normalizer.obs_rms
        return (obs - rms.mean) / np.sqrt(rms.var + 1e-8)
    except:
        return obs


def extract_pose_from_image(image_path):
    """Extract pose from image using PoseConverter."""
    if not POSE_CONVERTER_AVAILABLE:
        print("❌ PoseConverter not available")
        return None
    
    if not os.path.exists(DETECTOR_MODEL):
        print(f"❌ Detector model not found: {DETECTOR_MODEL}")
        return None
    
    print(f"\n🖼️  Extracting pose from: {os.path.basename(image_path)}")
    
    converter = PoseConverter(DETECTOR_MODEL, min_vis_threshold=0.5, min_detect_conf=0.3)
    theta_init, _, _ = converter.run_pipeline(image_path)
    
    if theta_init is not None:
        print(f"✅ Extracted {len(theta_init)} joint angles")
    else:
        print("⚠️ Could not extract pose, using default")
    
    return theta_init


def map_pose_to_actions(pose_12):
    """Map 12-joint pose to 17-joint action space."""
    if pose_12 is None:
        return None
    
    actions_17 = np.zeros(17, dtype=np.float32)
    
    # Mapping: our joints → HumanoidBulletEnv joints
    # chest → abdomen_y (index 1)
    actions_17[1] = np.clip(pose_12[0] / (np.pi/2), -1, 1)
    # r_shoulder → right_shoulder1 (index 11)
    actions_17[11] = np.clip(pose_12[2] / (np.pi/2), -1, 1)
    # r_elbow → right_elbow (index 13)
    actions_17[13] = np.clip(pose_12[3] / (np.pi/2), -1, 1)
    # l_shoulder → left_shoulder1 (index 14)
    actions_17[14] = np.clip(pose_12[4] / (np.pi/2), -1, 1)
    # l_elbow → left_elbow (index 16)
    actions_17[16] = np.clip(pose_12[5] / (np.pi/2), -1, 1)
    # r_hip → right_hip_y (index 5)
    actions_17[5] = np.clip(pose_12[6] / (np.pi/2), -1, 1)
    # r_knee → right_knee (index 6)
    actions_17[6] = np.clip(pose_12[7] / (np.pi/2), -1, 1)
    # l_hip → left_hip_y (index 9)
    actions_17[9] = np.clip(pose_12[9] / (np.pi/2), -1, 1)
    # l_knee → left_knee (index 10)
    actions_17[10] = np.clip(pose_12[10] / (np.pi/2), -1, 1)
    
    return actions_17


# ============================================================================
# MAIN SIMULATION
# ============================================================================
def run_policy(args):
    """Run humanoid walking simulation."""
    print("\n" + "=" * 60)
    print(" HUMANOID WALKING - POLICY EVALUATION")
    print("=" * 60)
    
    if not SB3_AVAILABLE or not BULLET_ENV_AVAILABLE:
        print("❌ Required packages not installed!")
        return
    
    # Extract pose from image if provided
    initial_pose = None
    if args.image:
        if not os.path.exists(args.image):
            print(f"❌ Image not found: {args.image}")
            return
        initial_pose = extract_pose_from_image(args.image)
    
    # Select model
    model_path = None
    normalizer_path = None
    mode = None
    
    if args.random:
        mode = "Random Actions"
    elif args.baseline:
        if os.path.exists(BASELINE_MODEL):
            model_path = BASELINE_MODEL
            normalizer_path = BASELINE_NORMALIZER
            mode = "Baseline Model"
        else:
            print(f"❌ Baseline model not found: {BASELINE_MODEL}")
            return
    else:
        if os.path.exists(TRAINED_MODEL):
            model_path = TRAINED_MODEL
            normalizer_path = TRAINED_NORMALIZER
            mode = "Trained Model"
        else:
            print("\n" + "=" * 60)
            print("⚠️  NO TRAINED MODEL FOUND")
            print("=" * 60)
            print(f"\nExpected: {TRAINED_MODEL}")
            print("\nOptions:")
            print("  1. Train a model:")
            print("     python train.py --use-sb3 --timesteps 500000")
            print("\n  2. Use baseline model:")
            print("     python run_policy.py --baseline")
            print("\n  3. Test with random actions:")
            print("     python run_policy.py --random")
            return
    
    # Add image info to mode
    if args.image:
        mode += f" + Image Pose"
    
    print(f"\n🤖 Mode: {mode}")
    
    # Create environment
    print("📦 Creating environment...")
    env = HumanoidBulletEnv(render=True)
    obs = env.reset()
    
    # Apply initial pose from image if available
    if initial_pose is not None:
        print("🎯 Applying pose from image...")
        pose_actions = map_pose_to_actions(initial_pose)
        if pose_actions is not None:
            # Apply pose for a few steps to set initial position
            for _ in range(30):
                obs, _, _, _ = env.step(pose_actions)
            print("✅ Pose applied")
    
    # Setup camera
    try:
        p.resetDebugVisualizerCamera(
            cameraDistance=5.0, cameraYaw=0, cameraPitch=-20,
            cameraTargetPosition=[0, 0, 1]
        )
    except:
        pass
    
    # Load walking model
    model, normalizer = None, None
    if model_path:
        model, normalizer = load_model(model_path, normalizer_path)
    
    print("\n🚀 Starting walking simulation... (Close window to quit)\n")
    
    episode_reward = 0
    step_count = 0
    episode_num = 0
    
    # Camera settings
    cam_distance = 5.0
    cam_yaw = 0
    cam_pitch = -20
    
    try:
        while True:
            # Get action from policy
            if model is None:
                action = env.action_space.sample()
            else:
                norm_obs = normalize_obs(obs, normalizer)
                action, _ = model.predict(norm_obs, deterministic=True)
            
            # Step
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            step_count += 1
            
            # Update camera to follow humanoid
            try:
                if hasattr(env, 'robot') and env.robot is not None:
                    robot_pos = env.robot.body_xyz
                    p.resetDebugVisualizerCamera(
                        cameraDistance=cam_distance,
                        cameraYaw=cam_yaw,
                        cameraPitch=cam_pitch,
                        cameraTargetPosition=[robot_pos[0], robot_pos[1], 1.0]
                    )
            except:
                pass
            
            if done:
                episode_num += 1
                print(f"📊 Episode {episode_num}: Reward = {episode_reward:.2f}, Steps = {step_count}")
                obs = env.reset()
                
                # Re-apply initial pose if provided
                if initial_pose is not None:
                    pose_actions = map_pose_to_actions(initial_pose)
                    if pose_actions is not None:
                        for _ in range(30):
                            obs, _, _, _ = env.step(pose_actions)
                
                episode_reward = 0
                step_count = 0
            
            time.sleep(1./30.)
    
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted")
    except:
        pass
    finally:
        try:
            env.close()
        except:
            pass
    
    print("\n✅ Done")


def main():
    parser = argparse.ArgumentParser(description="Run humanoid walking policy")
    parser.add_argument("--baseline", action="store_true",
                        help="Use baseline model")
    parser.add_argument("--random", action="store_true",
                        help="Random actions")
    parser.add_argument("--image", type=str, default=None,
                        help="Start from pose extracted from image")
    args = parser.parse_args()
    run_policy(args)


if __name__ == "__main__":
    main()
