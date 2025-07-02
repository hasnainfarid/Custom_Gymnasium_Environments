from setuptools import setup, find_packages

setup(
    name='smartclimate_rl',
    version='0.1.0',
    description='SmartClimate RL Environment for HVAC Control',
    author='Hasnain Fareed',
    packages=find_packages(),
    install_requires=[
        'gymnasium',
        'ray[rllib]',
        'pygame',
        'numpy',
    ],
    entry_points={
        'console_scripts': [
            'smartclimate-demo=smartclimate.examples.demo:main',
            'smartclimate-manual=smartclimate.examples.manual_test:main',
        ],
    },
    python_requires='>=3.8',
    include_package_data=True,
) 
