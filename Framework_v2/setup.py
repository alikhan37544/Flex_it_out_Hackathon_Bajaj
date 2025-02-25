from setuptools import setup, find_packages

setup(
    name="motion_detector",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "opencv-python",
        "mediapipe",
        "numpy"
    ],
    author="Ali Khan",
    author_email="alikhan37544@gmail.com",
    description="A motion detector module for hand gestures",
    # long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/motion_detector",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)