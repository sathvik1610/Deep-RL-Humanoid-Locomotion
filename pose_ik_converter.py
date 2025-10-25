import cv2
import mediapipe as mp
import numpy as np
import math
import pybullet as p
import pybullet_data
import time
import os
import sys
from copy import deepcopy

# MediaPipe Tasks imports
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# MediaPipe legacy drawing (optional, for visualization)
try:
    import mediapipe.python.solutions.pose as pose_solution
    import mediapipe.python.solutions.drawing_utils as drawing_utils
except ImportError:
    drawing_utils = None  # Handle if legacy solutions are not available


class PoseIKConverter:
    """
    Module 1 using MediaPipe for detection/pose and PyBullet's IK
    for the standard humanoid.urdf.
    Includes fixes for IK stability and target generation.
    """

    def __init__(self, detector_model_path, min_vis_threshold=0.5, min_detect_conf=0.3):
        print("\n--- Initializing PoseIKConverter (v2 Stability Fixes) ---")
        self.MIN_VISIBILITY = min_vis_threshold
        self.MIN_DETECT_CONF = min_detect_conf
        self.MP_LANDMARKS = mp.solutions.pose.PoseLandmark

        # --- Stage 1: Detector ---
        try:
            base_options = python.BaseOptions(model_asset_path=detector_model_path)
            options = vision.ObjectDetectorOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                score_threshold=self.MIN_DETECT_CONF,
                category_allowlist=['person'],
                max_results=10
            )
            self.mp_detector = vision.ObjectDetector.create_from_options(options)
            print("  [Init] ✅ MediaPipe ObjectDetector loaded.")
        except Exception as e:
            print(f"❌ FATAL: Could not load ObjectDetector model. Error: {e}")
            raise e

        # --- Stage 2: Pose Estimator ---
        self.mp_pose = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=self.MIN_DETECT_CONF
        )
        print("  [Init] ✅ MediaPipe Pose model loaded.")

        # --- PyBullet IK Setup (DIRECT mode) ---
        self.ik_client = p.connect(p.DIRECT)
        if self.ik_client < 0:
            raise RuntimeError("Could not connect PyBullet DIRECT.")

        # --- Load Humanoid URDF for IK ---
        urdf_path_local = "humanoid.urdf"
        if not os.path.exists(urdf_path_local):
            print(f"   - Local humanoid.urdf not found, checking pybullet_data...")
            p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.ik_client)
        else:
            print(f"   - Using local humanoid.urdf")

        try:
            # --- START OF MODIFICATION 1 (ROTATION) ---
            start_pos_ik = [0, 0, 6.0]  # Your height is correct
            # Add Z-axis rotation to align X/Y axes correctly
            start_orn_ik = p.getQuaternionFromEuler([math.pi / 2, 0, 0])  # Simple 90° rotation around X
            # --- END OF MODIFICATION 1 ---

            self.humanoid_id = p.loadURDF(
                "humanoid.urdf",
                start_pos_ik,
                start_orn_ik,
                useFixedBase=True,
                physicsClientId=self.ik_client
            )

            self._get_humanoid_info()
            print(f"  [Init] ✅ Standard humanoid.urdf loaded for IK (ID: {self.humanoid_id}).")
        except Exception as e:
            print(f"❌ FATAL: Could not load humanoid.urdf for IK. Error: {e}")
            p.disconnect(self.ik_client)
            raise e

        # --- START OF MODIFICATION 2 (TARGET MAP - THE REAL BUG) ---
        # We must target links that are children of MOVABLE joints.
        # 'left_wrist' is a FIXED joint, so targeting it fails.
        # We retarget landmarks to the parent link.
        self.target_link_map = {
            # Arms (swap L↔R because of viewer perspective)
            self.MP_LANDMARKS.LEFT_SHOULDER: self.link_name_to_index.get('right_shoulder', -1),
            self.MP_LANDMARKS.LEFT_ELBOW: self.link_name_to_index.get('right_elbow', -1),
            self.MP_LANDMARKS.LEFT_WRIST: self.link_name_to_index.get('right_wrist', -1),
            # use wrist link (fixed is OK)
            self.MP_LANDMARKS.RIGHT_SHOULDER: self.link_name_to_index.get('left_shoulder', -1),
            self.MP_LANDMARKS.RIGHT_ELBOW: self.link_name_to_index.get('left_elbow', -1),
            self.MP_LANDMARKS.RIGHT_WRIST: self.link_name_to_index.get('left_wrist', -1),

            # Legs (swap L↔R)
            self.MP_LANDMARKS.LEFT_HIP: self.link_name_to_index.get('right_hip', -1),
            self.MP_LANDMARKS.LEFT_KNEE: self.link_name_to_index.get('right_knee', -1),
            self.MP_LANDMARKS.LEFT_ANKLE: self.link_name_to_index.get('right_ankle', -1),
            self.MP_LANDMARKS.RIGHT_HIP: self.link_name_to_index.get('left_hip', -1),
            self.MP_LANDMARKS.RIGHT_KNEE: self.link_name_to_index.get('left_knee', -1),
            self.MP_LANDMARKS.RIGHT_ANKLE: self.link_name_to_index.get('left_ankle', -1),
        }
        # --- END OF MODIFICATION 2 ---

        self.target_link_indices = [idx for idx in self.target_link_map.values() if idx != -1]
        print(f"  [Init] ✅ Mapped MediaPipe landmarks to {len(self.target_link_indices)} target link indices.")

        self.reference_height_m = 1.75
        self.robot_base_z_offset = 6.0  # Your height is correct
        print(f"   - Using defined base Z offset: {self.robot_base_z_offset:.2f}m")

        print("--- Converter Initialized Successfully ---")

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'mp_detector'):
            self.mp_detector.close()
        if hasattr(self, 'ik_client') and self.ik_client >= 0:
            if p.isConnected(self.ik_client):
                p.disconnect(self.ik_client)

    def _get_humanoid_info(self):
        """Gets joint and link info for the standard humanoid."""
        self.num_joints = p.getNumJoints(self.humanoid_id, physicsClientId=self.ik_client)
        self.joint_indices = []
        self.link_name_to_index = {}
        self.joint_name_to_index = {}
        self.joint_lower_limits = []
        self.joint_upper_limits = []
        self.joint_ranges = []

        base_info = p.getBodyInfo(self.humanoid_id, physicsClientId=self.ik_client)
        base_link_name = base_info[0].decode('UTF-8')
        self.link_name_to_index[base_link_name] = -1

        for i in range(self.num_joints):
            info = p.getJointInfo(self.humanoid_id, i, physicsClientId=self.ik_client)
            joint_name = info[1].decode('UTF-8')
            link_name = info[12].decode('UTF-8')
            joint_type = info[2]
            self.link_name_to_index[link_name] = i

            if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC, p.JOINT_SPHERICAL]:
                self.joint_indices.append(i)
                self.joint_name_to_index[joint_name] = i
                lower = info[8]
                upper = info[9]
                if lower >= upper:
                    lower, upper = -np.pi * 2, np.pi * 2
                self.joint_lower_limits.append(lower)
                self.joint_upper_limits.append(upper)
                self.joint_ranges.append(upper - lower)

        print(f"   - Found {self.num_joints} total joints.")
        print(f"   - Found {len(self.joint_indices)} controllable joint indices.")
        if not (len(self.joint_lower_limits) == len(self.joint_upper_limits) ==
                len(self.joint_ranges) == len(self.joint_indices)):
            print("   - ⚠️ WARNING: Mismatch in joint limit/range list lengths!")

    def _get_pixel_box(self, detection, img_width, img_height):
        box = detection.bounding_box
        x = max(0, box.origin_x)
        y = max(0, box.origin_y)
        w = min(box.width, img_width - x)
        h = min(box.height, img_height - y)
        return (x, y, w, h)

    def _select_largest_skeleton(self, detected_skeletons):
        print(f"    [Select] Evaluating {len(detected_skeletons)} detected skeletons...")
        largest_area = -1
        selected_landmarks = None
        selected_bbox = (0, 0, 0, 0)

        for (landmarks, pixel_box) in detected_skeletons:
            (x, y, w, h) = pixel_box
            area = w * h
            if area > largest_area:
                largest_area, selected_landmarks, selected_bbox = area, landmarks, pixel_box

        if selected_landmarks:
            print(f"    [Select] ✅ Selected Skeleton (Area: {largest_area:.0f})")
        else:
            print(f"    [Select] ⚠️ No suitable skeleton found.")

        return selected_landmarks, selected_bbox

    def _convert_landmarks_to_full_image(self, landmarks, x_crop, y_crop, w_crop, h_crop,
                                         img_width, img_height):
        full_image_landmarks = deepcopy(landmarks)
        for landmark in full_image_landmarks.landmark:
            x_pix_full = (landmark.x * w_crop) + x_crop
            y_pix_full = (landmark.y * h_crop) + y_crop
            landmark.x = x_pix_full / img_width
            landmark.y = y_pix_full / img_height
        return full_image_landmarks

    def _get_scale_and_ref(self, landmarks, img_width_px, img_height_px):
        left_hip = landmarks.landmark[self.MP_LANDMARKS.LEFT_HIP]
        right_hip = landmarks.landmark[self.MP_LANDMARKS.RIGHT_HIP]
        left_shoulder = landmarks.landmark[self.MP_LANDMARKS.LEFT_SHOULDER]
        right_shoulder = landmarks.landmark[self.MP_LANDMARKS.RIGHT_SHOULDER]

        if not all(lm.visibility > self.MIN_VISIBILITY for lm in
                   [left_hip, right_hip, left_shoulder, right_shoulder]):
            print("    [Scale] ⚠️ Missing core landmarks for scaling.")
            return None, None, None

        mid_hip_x = (left_hip.x + right_hip.x) / 2
        mid_hip_y = (left_hip.y + right_hip.y) / 2
        mid_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        torso_height_px = abs(mid_hip_y - mid_shoulder_y) * img_height_px

        if torso_height_px < 10:
            print("    [Scale] ⚠️ Torso pixel height too small.")
            return None, None, None

        reference_torso_height_m = 0.5
        scale = reference_torso_height_m / torso_height_px
        print(f"    [Scale] Estimated scale: {scale:.4f} m/px")

        ref_point_3d = np.array([0.0, 0.0, self.robot_base_z_offset])
        ref_point_px = np.array([mid_hip_x * img_width_px, mid_hip_y * img_height_px])

        return scale, ref_point_3d, ref_point_px

    # ========== FIX 2: CORRECT COORDINATE TRANSFORMATION ==========
    # In _generate_3d_targets method, replace MODIFICATION 3 with:

    def _generate_3d_targets(self, landmarks, scale, ref_point_3d, ref_point_px,
                             img_width, img_height):
        """Generates target 3D positions for IK."""
        target_positions = {}

        for mp_index, link_index in self.target_link_map.items():
            if link_index == -1:
                continue

            landmark = landmarks.landmark[mp_index]
            if landmark.visibility < self.MIN_VISIBILITY:
                continue

            lm_point_px = np.array([landmark.x * img_width, landmark.y * img_height])
            pixel_vector = lm_point_px - ref_point_px

            # FIXED: Correct coordinate mapping
            # Original URDF has Y-axis up. After rotation [pi/2, 0, 0]:
            # - Image X → World X (horizontal/width)
            # - Image Y → World Z (vertical/height) with negation (image Y goes down)
            # - World Y = 0 (depth, frontal plane)
            target_vector_3d = np.array([
                pixel_vector[0] * scale,  # Image X → World X (left/right)
                0.0,  # World Y (depth) = 0 for 2D pose
                -pixel_vector[1] * scale  # Image Y → World -Z (up/down, negative because image Y goes down)
            ])

            target_pos = ref_point_3d + target_vector_3d
            target_positions[link_index] = np.round(target_pos, 3)

        print(f"    [IK Targets] Generated {len(target_positions)} 3D targets for IK.")
        return target_positions

    def _solve_ik(self, target_positions):
        if not target_positions:
            print("    [IK Solve] No valid targets.")
            return None

        target_link_indices = list(target_positions.keys())
        target_pos_values = list(target_positions.values())

        num_all_joints = self.num_joints
        num_controllable = len(self.joint_indices)

        standing_rest_pose = [0.0] * num_all_joints
        try:
            standing_rest_pose[self.joint_name_to_index['right_knee']] = -0.1
            standing_rest_pose[self.joint_name_to_index['left_knee']] = -0.1
            print("   - Using defined standing rest pose.")
        except KeyError as e:
            print(f"   - WARNING: Could not find joint '{e}' for standing rest pose. Using zeros.")
            standing_rest_pose = [0.0] * num_all_joints

        full_lower_limits = [-np.pi * 2] * num_all_joints
        full_upper_limits = [np.pi * 2] * num_all_joints
        full_joint_ranges = [np.pi * 4] * num_all_joints

        for j_idx in range(num_all_joints):
            info = p.getJointInfo(self.humanoid_id, j_idx, physicsClientId=self.ik_client)
            if info[2] != p.JOINT_FIXED:
                lower, upper = info[8], info[9]
                if lower >= upper:
                    lower, upper = -np.pi * 2, np.pi * 2
                full_lower_limits[j_idx] = lower
                full_upper_limits[j_idx] = upper
                full_joint_ranges[j_idx] = upper - lower

        try:
            all_joint_poses = p.calculateInverseKinematics2(
                bodyUniqueId=self.humanoid_id,
                endEffectorLinkIndices=target_link_indices,
                targetPositions=target_pos_values,
                lowerLimits=full_lower_limits,
                upperLimits=full_upper_limits,
                jointRanges=full_joint_ranges,
                restPoses=standing_rest_pose,
                maxNumIterations=200,
                residualThreshold=0.01,
                physicsClientId=self.ik_client
            )

            print(f"    [IK Solve] ✅ IK solution found. Full result length: {len(all_joint_poses)}")

            theta_init = [all_joint_poses[i] for i in self.joint_indices]

            if len(theta_init) != num_controllable:
                print(f"    [IK Solve] ⚠️ Final theta_init length mismatch!")
                return None

            return np.array(theta_init)

        except Exception as e:
            print(f"    [IK Solve] ❌ PyBullet IK failed: {e}")
            return None

    def run_pipeline(self, image_path):
        print(f"\n--- [Module 1 IK] Processing Image: {os.path.basename(image_path)} ---")
        start_time = time.time()

        try:
            image_cv = cv2.imread(image_path)
            img_height, img_width, _ = image_cv.shape
            print(f"  [Step 1] ✅ Image loaded ({img_width}x{img_height}).")
            image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
            image_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        except Exception as e:
            print(f"  [Step 1] ❌ Image Load Fail: {e}")
            return None, None, None

        print("  [Stage 1] Running Person Detector...")
        detector_results = self.mp_detector.detect(image_mp)
        person_detections = detector_results.detections

        if not person_detections:
            print("  [Stage 1] ⚠️ No 'Person' found.")
            return None, None, None

        print(f"  [Stage 1] ✅ Found {len(person_detections)} potential boxes.")

        print("  [Stage 2] Running Pose Estimator...")
        detected_skeletons, all_pixel_boxes = [], []

        for detection in person_detections:
            pixel_box = self._get_pixel_box(detection, img_width, img_height)
            all_pixel_boxes.append(pixel_box)
            (x, y, w, h) = pixel_box
            pad_x = int(w * 0.1)
            pad_y = int(h * 0.1)
            x_crop = max(0, x - pad_x)
            y_crop = max(0, y - pad_y)
            w_crop = min(w + 2 * pad_x, img_width - x_crop)
            h_crop = min(h + 2 * pad_y, img_height - y_crop)
            cropped_image_rgb = image_rgb[y_crop:y_crop + h_crop, x_crop:x_crop + w_crop]

            if cropped_image_rgb.size == 0:
                continue

            pose_results = self.mp_pose.process(cropped_image_rgb)
            if pose_results.pose_landmarks:
                full_landmarks = self._convert_landmarks_to_full_image(
                    pose_results.pose_landmarks, x_crop, y_crop, w_crop, h_crop,
                    img_width, img_height
                )
                detected_skeletons.append((full_landmarks, pixel_box))

        print(f"  [Stage 2] ✅ Extracted {len(detected_skeletons)} skeletons.")

        if not detected_skeletons:
            return None, None, None

        selected_landmarks, selected_bbox = self._select_largest_skeleton(detected_skeletons)
        if selected_landmarks is None:
            return None, None, None

        print("  [Stage 4] Calculating scale...")
        scale, ref_3d, ref_px = self._get_scale_and_ref(
            selected_landmarks, img_width, img_height
        )
        if scale is None:
            print("  [Stage 4] ⚠️ Scale failed.")
            return None, None, None

        print("  [Stage 5] Generating targets...")
        targets = self._generate_3d_targets(
            selected_landmarks, scale, ref_3d, ref_px, img_width, img_height
        )
        if not targets:
            print("  [Stage 5] ⚠️ Targets failed.")
            return None, None, None

        # --- START OF MODIFICATION 4 (USER OUTPUT) ---
        print(f"  [Step 5] ✅ 3D Targets (Link Index: [X, Y, Z]): {targets}")
        # --- END OF MODIFICATION 4 ---

        print("  [Stage 6] Solving IK...")
        ik_start = time.time()
        theta_init = self._solve_ik(targets)
        ik_dur = time.time() - ik_start
        success = theta_init is not None

        if not success:
            theta_init = np.zeros(len(self.joint_indices))
            print("  [Stage 6] ⚠️ Using fallback pose.")
        else:
            print(f"  [Stage 6] ✅ IK successful ({ik_dur:.2f}s).")

        print("  [Stage 7] Visualization...")
        vis_img = self._draw_poses(image_cv, all_pixel_boxes, selected_landmarks, selected_bbox)

        if vis_img is not None:
            print("  [VIS] Displaying...")
            max_h = 800
            h, w = vis_img.shape[:2]
            if h > max_h:
                scale_vis = max_h / h
                vis_img = cv2.resize(vis_img, (int(w * scale_vis), int(h * scale_vis)))
            cv2.imshow("IK Output", vis_img)
            print("  >>> Press any key ON THE IMAGE WINDOW to continue...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        total_time = time.time() - start_time
        print(f"\n--- [MODULE 1 {'OK' if success else 'FALLBACK'}] ({total_time:.2f}s) ---")
        print(f"  θinit shape: {theta_init.shape}")
        print("----------------------\n")

        def_pos = [0, 0, self.robot_base_z_offset]
        def_orn = [0, 0, 0, 1]

        return theta_init, def_pos, def_orn

    def _draw_poses(self, image, all_boxes, selected_landmarks, selected_box):
        img_drawn = image.copy()

        for box in all_boxes:
            if box == selected_box:
                continue
            (x, y, w, h) = box
            cv2.rectangle(img_drawn, (x, y), (x + w, y + h), (255, 0, 0), 2)

        (x_s, y_s, w_s, h_s) = selected_box
        cv2.rectangle(img_drawn, (x_s, y_s), (x_s + w_s, y_s + h_s), (0, 0, 255), 3)
        cv2.putText(
            img_drawn, "SELECTED", (x_s, y_s - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2
        )

        if selected_landmarks and drawing_utils:
            drawing_utils.draw_landmarks(
                image=img_drawn,
                landmark_list=selected_landmarks,
                connections=mp.solutions.pose.POSE_CONNECTIONS,
                landmark_drawing_spec=drawing_utils.DrawingSpec(
                    color=(0, 255, 0), thickness=2, circle_radius=2
                ),
                connection_drawing_spec=drawing_utils.DrawingSpec(
                    color=(0, 255, 0), thickness=2, circle_radius=2
                )
            )

        return img_drawn