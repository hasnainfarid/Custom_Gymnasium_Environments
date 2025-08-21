"""Setup script for Smart Manufacturing Environment package"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="smart_manufacturing_env",
    version="1.0.0",
    author="Manufacturing AI Lab",
    author_email="contact@manufacturingai.lab",
    description="A comprehensive smart manufacturing environment for reinforcement learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/manufacturing-ai/smart_manufacturing_env",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7",
    install_requires=[
        "gym>=0.21.0",
        "numpy>=1.19.0",
        "pygame>=2.0.0",
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
        "console_scripts": [
            "test-manufacturing=smart_manufacturing_env.test_manufacturing:main",
        ],
    },
    include_package_data=True,
    package_data={
        "smart_manufacturing_env": ["assets/*"],
    },
    zip_safe=False,
)