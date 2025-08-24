"""
Setup script for the Agricultural Farm Management Environment.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="agricultural-farm-env",
    version="1.0.0",
    author="Hasnain Fareed",
    author_email="",
    description="A comprehensive agricultural farm management environment for reinforcement learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/agricultural-farm-env",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Agriculture"
    ],
    python_requires=">=3.7",
    install_requires=[
        "gymnasium>=0.28.0",
        "numpy>=1.21.0",
        "pygame>=2.1.0",
        "matplotlib>=3.3.0"
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950"
        ]
    },
    entry_points={
        "gymnasium.envs": [
            "AgriculturalFarm-v0 = agricultural_farm_env:AgriculturalFarmEnv"
        ]
    },
    include_package_data=True,
    zip_safe=False,
    license="MIT",
    keywords="reinforcement-learning gymnasium agriculture farming simulation",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/agricultural-farm-env/issues",
        "Source": "https://github.com/yourusername/agricultural-farm-env",
    },
)

