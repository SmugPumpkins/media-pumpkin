from setuptools import setup, find_packages
setup(
    name='media_pumpkin',
    packages=find_packages(),
    version='0.1.0',
    license='MIT',
    description='Computer Vision Helping Library',
    author='SmugPumpkins',
    url='https://github.com/SmugPumpkins/media-py',
    keywords=['ComputerVision', 'HandTracking', 'FaceTracking', 'PoseEstimation'],
    install_requires=[
        'opencv-python',
        'numpy',
        'mediapipe',
        'protobuf<4',
    ],
    python_requires='>=3.9,<3.12',  # Requires any version >= 3.6

    classifiers=[
        'Development Status :: 3 - Alpha',
        # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',  # Specify which python versions that you want to support
    ],
)
