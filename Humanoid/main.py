import numpy as np
import os
import sys
import traceback

# Import the converter class from the other file
from pose_converter import MultiPersonMediaPipeConverter


def run_module_1():
    # --- Define paths ---
    URDF_FILE_PATH = "biped.urdf"
    DETECTOR_MODEL_PATH = "models/efficientdet_lite2.tflite"

    # --- Put your test image in the 'images/' folder ---
    IMAGE_FILE_NAME = "demo6.jpg"
    IMAGE_FILE_PATH = f"images/{IMAGE_FILE_NAME}"

    # --- Create Converter Instance ---
    try:
        if not os.path.exists(URDF_FILE_PATH):
            raise FileNotFoundError(f"URDF file not found at {URDF_FILE_PATH}.")
        if not os.path.exists(DETECTOR_MODEL_PATH):
            raise FileNotFoundError(f"Detector model not found at {DETECTOR_MODEL_PATH}.")
        if not os.path.exists(IMAGE_FILE_PATH):
            raise FileNotFoundError(f"Image file not found at {IMAGE_FILE_PATH}. Please add it to the 'images' folder.")

        # Tune the confidence here
        converter = MultiPersonMediaPipeConverter(
            urdf_path=URDF_FILE_PATH,
            detector_model_path=DETECTOR_MODEL_PATH,
            min_vis_threshold=0.5,
            min_detect_conf=0.2  # Lowered to find more people
        )

        # --- Run on the single local image ---
        theta_init, root_pos, root_orn = converter.run_pipeline(IMAGE_FILE_PATH)

        # --- Print Final Results ---
        print("\n========= FINAL MODULE 1 OUTPUT =========")
        print(f"  theta_init (10 angles, radians):\n  {np.round(theta_init, 3)}")
        print(f"  root_pos: {root_pos}")
        print(f"  root_orn: {root_orn}")
        print("=========================================")

    except FileNotFoundError as e:
        print(f"❌❌❌ FILE ERROR: {e}")
    except Exception as e:
        print(f"❌❌❌ An unexpected error occurred: {e}")
        traceback.print_exc()


# This makes the script runnable
if __name__ == "__main__":
    run_module_1()