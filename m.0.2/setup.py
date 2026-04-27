from setuptools import setup, find_packages

setup(
    name="muscle-sensor-processor",
    version="0.1.0",
    description="Muscle sensor signal processing system for world engine control",
    author="Your Name",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "pandas>=2.0.0",
    ],
)
