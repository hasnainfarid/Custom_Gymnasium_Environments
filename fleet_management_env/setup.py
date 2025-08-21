"""
Setup script for Fleet Management Environment package
"""

from setuptools import setup, find_packages
import os

# Read README file for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Fleet Management Environment for multi-agent reinforcement learning"

# Read version from __init__.py
def read_version():
    init_path = os.path.join(os.path.dirname(__file__), 'fleet_management_env', '__init__.py')
    with open(init_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip('"').strip("'")
    return "1.0.0"

setup(
    name="fleet_management_env",
    version=read_version(),
    author="Fleet Management Environment Team",
    author_email="fleet-env@example.com",
    description="A comprehensive multi-agent reinforcement learning environment for urban delivery logistics",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/example/fleet_management_env",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Games/Entertainment :: Simulation",
    ],
    python_requires=">=3.8",
    install_requires=[
        "gymnasium>=0.26.0",
        "numpy>=1.21.0",
        "pygame>=2.1.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "pandas>=1.3.0",
        "dataclasses; python_version<'3.7'",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.910",
            "jupyter>=1.0.0",
            "notebook>=6.0.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "sphinxcontrib-napoleon>=0.7",
        ],
        "testing": [
            "pytest>=6.0",
            "pytest-benchmark>=3.4.0",
            "pytest-xdist>=2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "fleet-test=fleet_management_env.test_fleet:main",
        ],
    },
    include_package_data=True,
    package_data={
        "fleet_management_env": [
            "*.md",
            "*.txt",
            "*.yml",
            "*.yaml",
        ],
    },
    zip_safe=False,
    keywords=[
        "reinforcement learning",
        "multi-agent",
        "logistics",
        "fleet management",
        "urban delivery",
        "gymnasium",
        "environment",
        "simulation",
        "AI",
        "machine learning",
    ],
    project_urls={
        "Bug Reports": "https://github.com/example/fleet_management_env/issues",
        "Source": "https://github.com/example/fleet_management_env",
        "Documentation": "https://fleet-management-env.readthedocs.io/",
    },
)