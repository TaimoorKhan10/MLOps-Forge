from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Core dependencies
requirements = [
    "scikit-learn>=1.3.0",
    "pandas>=2.1.0",
    "numpy>=1.24.3",
    "torch>=2.0.1",
    "feast>=0.30.1",
    "fastapi>=0.103.1",
    "uvicorn>=0.23.2",
    "pydantic>=2.4.2",
    "python-dotenv>=1.0.0",
    "mlflow>=2.7.1",
    "dvc>=3.21.1",
    "prometheus-client>=0.17.1",
    "loguru>=0.7.0"
]

setup(
    name="mlops-forge",  # Using hyphen for PyPI compatibility
    version="1.0.0",
    author="Taimoor Khan",
    author_email="contact@taimoorkhan.dev",
    description="A complete production-ready MLOps framework with built-in distributed training, monitoring, and CI/CD.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TaimoorKhan10/MLOps-Forge",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "mlops-forge-train=mlops_forge.cli.train:main",
            "mlops-forge-serve=mlops_forge.cli.serve:main",
            "mlops-forge-monitor=mlops_forge.cli.monitor:main",
        ],
    },
)
