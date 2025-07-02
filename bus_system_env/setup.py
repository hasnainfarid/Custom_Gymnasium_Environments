from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="bus-system-env",
    version="1.0.0",
    author="Hasnain Fareed",
    author_email="Hasnainfarid7@yahoo.com",
    description="A custom OpenAI Gymnasium environment for urban bus system simulation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hasnainfarid/bus-system-env",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
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
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=[
        "gymnasium>=0.28.0",
        "numpy>=1.19.0",
        "pygame>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
        ],
    },
    keywords="reinforcement-learning gymnasium environment bus-system simulation",
    project_urls={
        "Bug Reports": "https://github.com/hasnainfarid/bus-system-env/issues",
        "Source": "https://github.com/hasnainfarid/bus-system-env",
    },
) 