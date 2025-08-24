"""
Setup script for Airline Operations Environment
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="airline-operations-env",
    version="1.0.0",
    author="Hasnain Fareed",
    author_email="",
    description="A comprehensive Gymnasium environment for airline operations management with hub-and-spoke network simulation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/airline-operations-env",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Games/Entertainment :: Simulation",
    ],
    python_requires=">=3.7",
    install_requires=[
        "gymnasium>=0.28.0",
        "numpy>=1.21.0",
        "pygame>=2.1.0",
        "matplotlib>=3.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
        ],
    },
    entry_points={
        "gymnasium.envs": [
            "AirlineOperations-v0 = airline_operations_env:AirlineOperationsEnv",
        ],
    },
    include_package_data=True,
    license="MIT",
    keywords="reinforcement-learning airline operations management gymnasium environment simulation",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/airline-operations-env/issues",
        "Source": "https://github.com/yourusername/airline-operations-env",
    },
)




