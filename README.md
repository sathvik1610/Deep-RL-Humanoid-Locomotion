
<h1 align="center">Deep Reinforcement Learning for Humanoid Locomotion from an Image-Defined Initial Pose</h1>

<p align="center">
Developed as an end-to-end pipeline using a custom Python library (with <code>setup.py</code>) that enables a simulated humanoid agent to learn stable walking using Deep Reinforcement Learning. The agent’s initial pose for each episode is determined by a 25-keypoint skeleton extracted from a user-provided static image.
</p>

---

<h2 align="center">Expected Inputs & Outputs</h2>

**Inputs:**  
1. A static image file (e.g., `.jpg`, `.png`) provided by the user. The image can contain a single person or multiple people in various poses.  
2. A pre-defined humanoid model specified in a URDF or MJCF file.  

**Outputs:**  
1. A trained Deep Q-Network (DQN) representing a robust walking policy that can be initiated from numerous starting configurations.  
2. Multi-person keypoint extraction: accurately extract a 25-keypoint skeleton for every person detected in the user’s image.  
3. Selection mechanism for the target skeleton if multiple people are detected using template matching (ORB/SIFT/SURF/Haar Cascades).  
4. A custom Gymnasium-compliant physics environment with a reset function capable of initializing humanoid joint angles from the extracted pose, and additional poses generated via GANs.  
5. A DQN agent for the locomotion task with high-dimensional continuous states and discretized action space.  
6. A reward function encouraging forward velocity and stability, allowing dynamic walking learning from varied initial poses.  

---

<h2 align="center">Module 1: Pose Estimation & Initial Pose Extraction</h2>

<h3>Task 1.1: Image Acquisition and Preprocessing</h3>

**Description:** Load an image provided by the user.  
**Tools:** OpenCV (`cv2`) or Pillow (`PIL`)  
**Implementation:**  
- Load the image from file path  
- Convert it into a format suitable for pose estimation (e.g., NumPy array)  

---

<h3>Task 1.2: Multi-Person 25-Keypoint Extraction</h3>

**Description:** Detect multiple people and output a 25-keypoint skeleton for each.  
**Tools:** OpenPose (BODY 25 model)  
**Output:**  
- List of detected skeletons, each as a `[25, 3]` array representing (x, y, confidence) for each joint  

---

<h3>Task 1.3: Target Selection and Kinematic Conversion</h3>

**Description:**  
- Select a skeleton if multiple people are detected  
- Convert Cartesian coordinates to initial joint angles for the humanoid  

**Implementation:**  
1. **Selection:** Choose the person with the largest bounding box area  
2. **Conversion:**  
   - Define vectors between adjacent joints (e.g., hip→knee, knee→ankle)  
   - Use `atan2` to calculate joint angles  

**Output:**  
- A single NumPy vector representing the Initial Pose Vector (`θ_init`) to pass to the environment’s `reset` function  

---

<h2 align="center">Module 2: Simulation (Physics Environment)</h2>

<h3>Task 2.1: Humanoid Asset Definition (Rigging)</h3>

**Description:** Define physical properties of the humanoid agent.  
**Tools:** URDF or MJCF  
**Implementation:**  
- XML-based `.urdf` file specifies links (mass, inertia, collision geometry), joints (axes, limits), and actuators (torque limits)  

---

<h3>Task 2.2: Environment Instantiation and API</h3>

**Description:** Create a custom Python environment class that loads the humanoid and manages the simulation state.  
**Tools:** PyBullet, Gymnasium  

**Implementation (`HumanoidWalkEnv` class):**  
1. `__init__()` – Initializes PyBullet, loads URDF, defines observation and action spaces  
2. `reset(initial_pose=None)` – Resets simulation; if `initial_pose` provided, sets joint states accordingly; returns initial observation  
3. `step(action)` – Steps physics, calculates reward, checks termination, returns results  

---

<h2 align="center">Module 3: Control (DQN Agent)</h2>

<h3>Task 3.1: Network Architecture</h3>

**Description:** Define the Q-Network for DQN.  
**Tools:** PyTorch or TensorFlow/Keras  

**Implementation (MLP):**  
- **Input Layer:** Dimension equals state vector (joint angles, velocities, torso orientation, etc.)  
- **Hidden Layers:** 2–3 fully connected layers (256–512 neurons) with ReLU activations  
- **Output Layer:** Total number of discrete actions  

---

<h3>Task 3.2: Action Space Discretization</h3>

**Description:** Discretize continuous torque values for DQN compatibility.  

**Implementation:**  
- Each joint has torque bins (e.g., 5 bins: [-1.0, -0.5, 0, 0.5, 1.0])  
- For 8 joints × 5 bins → 40 output neurons  
- Reshape output to `[8,5]` and select `argmax` per joint for action  

---

<h3>Task 3.3: Reward Function Engineering</h3>

**Description:** Design a reward function for stable, forward walking.  

**Reward Formula:**  

```

R_t = w_vel * r_vel + w_live * r_live - w_energy * r_energy

```

- `r_vel`: Reward for forward velocity of torso center of mass  
- `r_live`: Small positive reward for remaining upright  
- `r_energy`: Penalty for sum of squared torques (efficiency)  
- Large negative reward if the agent falls  

---
