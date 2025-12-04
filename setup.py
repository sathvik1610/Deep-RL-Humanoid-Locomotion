from setuptools import setup, find_packages

setup(
    name="humanoid_locomotion",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "gymnasium",
        "pybullet",
        "mediapipe",
        "stable-baselines3",
        "torch",
        "opencv-python",
        "numpy",
    ],
)
