from setuptools import setup, find_packages

setup(
    name="eeg-lead-pipeline",
    version="0.1.0",
    description="Train EEGConformer from scratch on the LEAD EEG dataset",
    packages=find_packages(where="."),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "scipy>=1.10",
        "scikit-learn>=1.3",
        "tqdm>=4.65",
        "PyYAML>=6.0",
        "python-dotenv>=1.0",
        "torch>=2.1",
        "braindecode>=1.3.2",
        "wandb>=0.17",
        "matplotlib>=3.7",
    ],
)
