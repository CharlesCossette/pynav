from setuptools import setup, find_packages

setup(
    name="navlie",
    version="0.0.1",
    description="A collection of common state estimation algorithms in robotics.",
    packages=find_packages(),
    extras_require={"test": ["pytest"]},
    install_requires=[
        "numpy>=1.21.2",
        "scipy>=1.7.1",
        "matplotlib>=3.4.3",
        "joblib>=1.2.0",
        "pylie @ git+https://github.com/decargroup/pylie@main",
        "tqdm>=4.64.1",
        "seaborn>=0.11.2",
    ],
)
