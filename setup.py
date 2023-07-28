from setuptools import setup, find_packages

with open("README.rst", "r") as f:
    readme = f.read()

setup(
    name="navlie",
    version="0.0.1",
    description="A collection of common state estimation algorithms in robotics.",
    long_description=readme,
    packages=find_packages(),
    extras_require={"test": ["pytest"]},
    install_requires=[
        "numpy>=1.21.2",
        "scipy>=1.7.1",
        "matplotlib>=3.4.3",
        "joblib>=1.2.0",
        "tqdm>=4.64.1",
        "seaborn>=0.11.2",
    ],
    url="https://github.com/decargroup/navlie",
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    project_urls={
        "Source": "https://github.com/decargroup/navlie",
        "Tracker": "https://github.com/decargroup/navlie/issues"
    },
)
