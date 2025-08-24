"""
Setup script for Emergency Response Environment package

Author: Hasnain Fareed
License: MIT (2025)
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Emergency Response Environment for Reinforcement Learning"

setup(
    name="emergency_response_env",
    version="1.0.0",
    author="Hasnain Fareed",
    author_email="",
    description="A comprehensive emergency response coordination environment for reinforcement learning",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Games/Entertainment :: Simulation",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "gymnasium>=0.28.0",
        "numpy>=1.21.0",
        "pygame>=2.1.0",
        "matplotlib>=3.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
    },
    include_package_data=True,
    package_data={
        "emergency_response_env": [
            "scenarios/*.json",
        ],
    },
    entry_points={
        "console_scripts": [
            "emergency-response-test=emergency_response_env.test_emergency:main",
        ],
    },
    license="MIT",
    keywords=[
        "reinforcement learning",
        "emergency response",
        "disaster management",
        "gymnasium",
        "simulation",
        "multi-agent",
        "emergency services",
        "crisis management",
    ],
    project_urls={
        "Bug Reports": "",
        "Source": "",
        "Documentation": "",
    },
)




