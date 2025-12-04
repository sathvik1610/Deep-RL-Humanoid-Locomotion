import cv2
import mediapipe as mp
import numpy as np
import math
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


class PoseConverter:
    """
    Module 1 using MediaPipe for detection/pose and direct geometric
    angle calculation (v5 - Joint-Specific References).
    """

    def __init__(self, detector_model_path, min_vis_threshold=0.5, min_detect_conf=0.3):
        print("\n--- Initializing PoseConverter (Geometric Version) ---")
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
        print("--- Converter Initialized Successfully ---")

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'mp_detector'):
            self.mp_detector.close()

    # ========================================================================
    # ### CORE FIX (v5) ###
    # This function is RE-WRITTEN to use the correct "zero" reference
    # for each joint type (Hips = vertical, Shoulders = horizontal).
    # This fixes the "face in hip" / "T-Rex arm" bugs.
    # ========================================================================
    def _calculate_joint_angles_from_landmarks(self, landmarks):
        """
        Calculates joint angles using relative angles and joint-specific
        zero-references (T-Pose).
        """

        # --- Landmark indices ---
        LM = self.MP_LANDMARKS

        # --- Helper functions ---
        def get_point(idx):
            """Get landmark as a 2D numpy array [x, -y]"""
            lm = landmarks.landmark[idx]
            if lm.visibility < self.MIN_VISIBILITY:
                return None  # Return None if not visible
            # We use -y because image Y-axis is inverted
            return np.array([lm.x, -lm.y])

        def get_vector(p1, p2):
            """Get vector from p1 to p2"""
            if p1 is None or p2 is None:
                return None
            return p2 - p1

        def get_signed_angle(v_ref, v_limb):
            """Get the signed angle from v_ref to v_limb in radians"""
            if v_ref is None or v_limb is None:
                return 0.0  # Default to 0 angle

            # Check for zero vectors
            if np.linalg.norm(v_ref) < 1e-6 or np.linalg.norm(v_limb) < 1e-6:
                return 0.0

            # Angle in radians
            angle = math.atan2(v_limb[1], v_limb[0]) - math.atan2(v_ref[1], v_ref[0])
            # Normalize to [-pi, pi]
            if angle > math.pi:
                angle -= 2 * math.pi
            elif angle <= -math.pi:
                angle += 2 * math.pi
            return angle

        # --- Get all key points ---
        p_l_shoulder = get_point(LM.LEFT_SHOULDER)
        p_r_shoulder = get_point(LM.RIGHT_SHOULDER)
        p_l_elbow = get_point(LM.LEFT_ELBOW)
        p_r_elbow = get_point(LM.RIGHT_ELBOW)
        p_l_wrist = get_point(LM.LEFT_WRIST)
        p_r_wrist = get_point(LM.RIGHT_WRIST)
        p_l_hip = get_point(LM.LEFT_HIP)
        p_r_hip = get_point(LM.RIGHT_HIP)
        p_l_knee = get_point(LM.LEFT_KNEE)
        p_r_knee = get_point(LM.RIGHT_KNEE)
        p_l_ankle = get_point(LM.LEFT_ANKLE)
        p_r_ankle = get_point(LM.RIGHT_ANKLE)

        # --- Handle missing critical points ---
        if p_l_hip is None or p_r_hip is None or p_l_shoulder is None or p_r_shoulder is None:
            print("    [Angles] ⚠️ Missing critical torso landmarks. Using default pose.")
            return np.zeros(12)  # Return a 0-pose

        # --- Define Torso Reference Vectors ---
        p_mid_hip = (p_l_hip + p_r_hip) / 2
        p_mid_shoulder = (p_l_shoulder + p_r_shoulder) / 2

        # v_torso_down: Reference for HIPS (0 angle = leg straight down)
        v_torso_down = get_vector(p_mid_shoulder, p_mid_hip)

        # v_torso_right: Reference for RIGHT SHOULDER (0 angle = arm straight out)
        if v_torso_down is None:
            v_torso_down = np.array([0, -1])  # Default
        v_torso_right = np.array([1.0, 0.0])

        # v_torso_left: Reference for LEFT SHOULDER (0 angle = arm straight out)
        v_torso_left = np.array([-1.0, 0.0])

        # --- Get Limb Vectors (NO L/R SWAP - 1:1 MAPPING) ---
        v_l_upper_arm = get_vector(p_l_shoulder, p_l_elbow)
        v_l_forearm = get_vector(p_l_elbow, p_l_wrist)
        v_l_thigh = get_vector(p_l_hip, p_l_knee)
        v_l_calf = get_vector(p_l_knee, p_l_ankle)

        v_r_upper_arm = get_vector(p_r_shoulder, p_r_elbow)
        v_r_forearm = get_vector(p_r_elbow, p_r_wrist)
        v_r_thigh = get_vector(p_r_hip, p_r_knee)
        v_r_calf = get_vector(p_r_knee, p_r_ankle)

        # --- Calculate Relative Angles (in Radians) ---

        # === Robot's Right Arm (uses Person's RIGHT Arm) ===
        # Shoulder: Angle between torso's "right" and upper arm
        angle_r_shoulder = get_signed_angle(v_torso_right, v_r_upper_arm)
        # Elbow: Angle between upper arm and forearm
        angle_r_elbow = get_signed_angle(v_r_upper_arm, v_r_forearm)

        # === Robot's Left Arm (uses Person's LEFT Arm) ===
        # Shoulder: Angle between torso's "left" and upper arm
        angle_l_shoulder = get_signed_angle(v_torso_left, v_l_upper_arm)
        # Elbow: Angle between upper arm and forearm
        angle_l_elbow = get_signed_angle(v_l_upper_arm, v_l_forearm)

        # === Robot's Right Leg (uses Person's RIGHT Leg) ===
        # Hip: Angle between torso's "down" and thigh
        angle_r_hip = get_signed_angle(v_torso_down, v_r_thigh)
        # Knee: Angle between thigh and calf
        angle_r_knee = get_signed_angle(v_r_thigh, v_r_calf)

        # === Robot's Left Leg (uses Person's LEFT Leg) ===
        # Hip: Angle between torso's "down" and thigh
        angle_l_hip = get_signed_angle(v_torso_down, v_l_thigh)
        # Knee: Angle between thigh and calf
        angle_l_knee = get_signed_angle(v_l_thigh, v_l_calf)

        # --- Apply Constraints ---

        # For now, temporarily commented, might uncomment later based on RL Training.

        # angle_r_elbow = np.clip(abs(angle_r_elbow), 0, np.radians(150))
        # angle_l_elbow = np.clip(abs(angle_l_elbow), 0, np.radians(150))
        # angle_r_knee = np.clip(abs(angle_r_knee), 0, np.radians(150))
        # angle_l_knee = np.clip(abs(angle_l_knee), 0, np.radians(150))
        #
        # # Hips and Shoulders are clamped to sane ranges
        # angle_r_hip = np.clip(angle_r_hip, np.radians(-90), np.radians(90))
        # angle_l_hip = np.clip(angle_l_hip, np.radians(-90), np.radians(90))
        # angle_r_shoulder = np.clip(angle_r_shoulder, np.radians(-150), np.radians(150))
        # angle_l_shoulder = np.clip(angle_l_shoulder, np.radians(-150), np.radians(150))


        # --- Assemble Final Vector ---
        # Indices: [chest, neck, r_shoulder, r_elbow, l_shoulder, l_elbow,
        #           r_hip, r_knee, r_ankle, l_hip, l_knee, l_ankle]
        theta_geometric = np.array([
            0.0,  # chest
            0.0,  # neck
            angle_r_shoulder,  # <-- Back to R
            angle_r_elbow,  # <-- Back to R
            angle_l_shoulder,  # <-- Back to L
            angle_l_elbow,  # <-- Back to L
            angle_r_hip,  # <-- Back to R
            angle_r_knee,  # <-- Back to R
            0.0,  # r_ankle (too unstable to calculate)
            angle_l_hip,  # <-- Back to L
            angle_l_knee,  # <-- Back to L
            0.0  # l_ankle (too unstable to calculate)
        ])

        return theta_geometric

    # ========================================================================
    # END OF NEW GEOMETRY LOGIC
    # ========================================================================

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

    def run_pipeline(self, image_path):
        print(f"\n--- [Module 1 Geometric] Processing Image: {os.path.basename(image_path)} ---")
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

        print("    [Detections]:")
        for i, detection in enumerate(person_detections):
            box = detection.bounding_box
            conf = detection.categories[0].score
            print(f"      - Person {i}: (x={box.origin_x}, y={box.origin_y}), Confidence: {conf:.2f}")

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

        print("    [Task 1.2 Output]: List of detected skeletons")
        for i, (landmarks, bbox) in enumerate(detected_skeletons):
            # Convert landmarks to a numpy array (x, y, visibility)
            # MediaPipe has 33 landmarks
            skeleton_array = np.array(
                [[lm.x, lm.y, lm.visibility] for lm in landmarks.landmark]
            )
            print(f"      - Skeleton {i}: NumPy array, shape {skeleton_array.shape}")

        if not detected_skeletons:
            return None, None, None

        selected_landmarks, selected_bbox = self._select_largest_skeleton(detected_skeletons)
        if selected_landmarks is None:
            return None, None, None

        print("  [Stage 5] Calculating angles from landmarks...")
        ik_start = time.time()
        theta_init = self._calculate_joint_angles_from_landmarks(selected_landmarks)
        ik_dur = time.time() - ik_start
        success = theta_init is not None

        if not success:
            theta_init = np.zeros(12)
            print("  [Stage 5] ⚠️ Using fallback pose.")
        else:
            print(f"  [Stage 5] ✅ Geometric angles calculated ({ik_dur:.4f}s).")

        print("  [Stage 7] Visualization...")
        vis_img = self._draw_poses(image_cv, all_pixel_boxes, selected_landmarks, selected_bbox)

        if vis_img is not None:
            print("  [VIS] Displaying...")

            max_width = 800
            max_height = 600
            h, w = vis_img.shape[:2]
            scale = min(max_width / w, max_height / h)
            if scale < 1.0:
                vis_img = cv2.resize(vis_img, (int(w * scale), int(h * scale)))

            cv2.imshow("Geometric Pose Output", vis_img)
            print("  >>> Press any key ON THE IMAGE WINDOW to continue...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        total_time = time.time() - start_time
        print(f"\n--- [MODULE 1 {'OK' if success else 'FALLBACK'}] ({total_time:.2f}s) ---")
        print(f"  θinit shape: {theta_init.shape}")
        print("----------------------\n")

        def_pos = [0, 0, 6.0]
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
